#!/usr/bin/env python3
"""External pilot: register provider → submit workload → Autopilot → result + cost.

Reproducible same-host demonstration of the Prompt 4 workflow. Does **not**
claim multi-region fleet; that is covered by the Linode distributed proof.

Writes ``artifacts/external_pilot.json``.
"""

from __future__ import annotations

import importlib
import json
import os
import socket
import tempfile
import threading
import time
from pathlib import Path

import uvicorn

from tron_pilot.client import TronClient
from tron_pilot.provider import ProviderRegistry
from tron_pilot.provider_stats import ProviderStatsStore
from tron_pilot.usage import UsageLedger


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_master(tmp: Path, port: int):
    os.environ["TRON_SPINE_DIR"] = str(tmp / "spine")
    os.environ["TRON_SUBMIT_AUTH_TOKEN"] = "pilot-submit-secret"
    import queue_server

    if hasattr(queue_server, "BACKGROUND_STOP"):
        queue_server.BACKGROUND_STOP.set()
        time.sleep(0.1)
    importlib.reload(queue_server)
    queue_server.BACKGROUND_STOP.clear()
    queue_server.HEARTBEAT_TIMEOUT_SECONDS = 2.0
    queue_server.MATCH_INTERVAL_SECONDS = 0.2

    config = uvicorn.Config(
        queue_server.app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 15
    while not getattr(server, "started", False) and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("master failed to start")
    match_t = threading.Thread(target=queue_server._match_loop, daemon=True)
    watch_t = threading.Thread(target=queue_server._watchdog_loop, daemon=True)
    match_t.start()
    watch_t.start()
    return server, queue_server, t, match_t, watch_t


def _start_autopilot(port: int):
    os.environ["TRON_AUTOPILOT_API_TOKEN"] = "pilot-api-secret"
    from tron_autopilot import api as ap_api

    importlib.reload(ap_api)
    # Reload resets module-level API_TOKEN from env
    ap_api.API_TOKEN = os.environ["TRON_AUTOPILOT_API_TOKEN"]
    config = uvicorn.Config(
        ap_api.app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 15
    while not getattr(server, "started", False) and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("autopilot api failed to start")
    return server, t


def _start_worker(master_url: str, tmp: Path, name: str = "pilot-w0"):
    os.environ["TRON_MASTER_URL"] = master_url
    os.environ["TRON_WORKER_NAME"] = name
    os.environ["TRON_AUTH_TOKEN_FILE"] = str(tmp / f"{name}_auth.json")
    os.environ["TRON_WORKER_SKU"] = "g6-nanode-1"
    os.environ["TRON_WORKER_HOURLY_RATE"] = "0.0075"
    import worker as worker_mod

    importlib.reload(worker_mod)
    worker_mod._MP_CONTEXT = None
    token = worker_mod.register_worker()
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            try:
                worker_mod.heartbeat(token)
                job = worker_mod.fetch_next_job(token)
                if not job:
                    time.sleep(0.05)
                    continue
                t0 = time.time()
                result = worker_mod.execute_job(job)
                worker_mod.report_complete(token, job["id"], result, time.time() - t0)
            except Exception:
                time.sleep(0.05)

    threading.Thread(target=loop, daemon=True).start()
    return stop, name


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="tron-pilot-"))
    master_port = _free_port()
    api_port = _free_port()
    master_url = f"http://127.0.0.1:{master_port}"
    api_url = f"http://127.0.0.1:{api_port}"

    server, qs, st, mt, wt = _start_master(tmp, master_port)
    ap_server, ap_t = _start_autopilot(api_port)
    stop_w, worker_name = _start_worker(master_url, tmp)
    time.sleep(0.4)

    usage_path = tmp / "usage.jsonl"
    stats_path = tmp / "stats.jsonl"
    registry_path = tmp / "providers.json"

    client = TronClient(
        autopilot_url=api_url,
        master_url=master_url,
        customer_id="pilot-dev-1",
        project_id="external-pilot",
        usage_ledger=UsageLedger(usage_path),
        provider_stats=ProviderStatsStore(stats_path),
        provider_registry=ProviderRegistry(registry_path),
    )

    # Provider already registered via worker; also create durable provider_id.
    # Use register_with_master=False to avoid duplicate worker name conflict.
    rec, _ = client.providers.register(
        worker_name=worker_name,
        master_url=master_url,
        contact="pilot@example.com",
        region="local-lab",
        sku="g6-nanode-1",
        hourly_rate_usd=0.0075,
        memory_gb=1.0,
        register_with_master=False,
    )
    challenge = client.providers.issue_challenge(rec.provider_id)
    client.providers.verify(
        rec.provider_id, verified_by="pilot-harness", challenge=challenge
    )
    health = client.sync_health = client.providers.sync_health_from_master(master_url)
    fleet = client.fleet()
    trust = client.trust_model()

    # 1) Callable via pool with submit auth
    def work(x, y):
        return {"sum": x + y, "product": x * y}

    pooled = client.submit_callable(work, 6, 7, wait_timeout_seconds=60)

    # 2) Autopilot plan + run (synthetic local — efficiency=False needs callable;
    #    API /run defaults efficiency=True which needs model — use plan only +
    #    local distributed-style map via partition if workers alive, else plan)
    plan = client.plan(
        model_name="pilot/demo",
        hourly_rate_usd=0.5,
        expected_runtime_seconds=0.2,
        reliability="tolerant",
        strategy="autopilot",
    )

    # 3) Distributed map if ≥1 worker (coordinator needs 2 for distributed
    #    candidate; with 1 worker Autopilot stays local/pooled). Force pooled
    #    path evidence via submit_callable above; also run a second worker.
    stop_w2, worker_name2 = _start_worker(master_url, tmp, name="pilot-w1")
    time.sleep(0.5)
    items = list(range(40))
    run = client.run(
        model_name="pilot/distributed",
        hourly_rate_usd=0.5,
        expected_runtime_seconds=8.0,
        parallelizable=True,
        partition_items=items,
        map_op="sum_squares",
        bytes_per_item=8.0,
        efficiency=False,
        wait_timeout_seconds=90,
        strategy="autopilot",
    )

    report = client.usage_report()
    artifact = {
        "pilot": "external_developer_workflow",
        "success": bool(pooled and pooled.get("sum") == 13)
        and bool((run.get("outcome") or {}).get("success")),
        "customer_id": "pilot-dev-1",
        "provider": rec.to_dict(),
        "provider_verified": True,
        "health_sync": health,
        "fleet": fleet,
        "trust": trust,
        "pooled_callable_result": pooled,
        "plan_chosen": (plan.get("chosen") or {}).get("target"),
        "run_outcome": {
            "success": (run.get("outcome") or {}).get("success"),
            "target": (run.get("outcome") or {}).get("target"),
            "actual_runtime_seconds": (run.get("outcome") or {}).get(
                "actual_runtime_seconds"
            ),
            "actual_cost_usd": (run.get("outcome") or {}).get("actual_cost_usd"),
            "metrics": (run.get("outcome") or {}).get("metrics"),
            "error": (run.get("outcome") or {}).get("error"),
        },
        "usage_report": {
            k: v for k, v in report.items() if k != "rows"
        },
        "usage_events": report.get("events"),
        "honesty": [
            "Same-host pilot for control-plane workflow reproducibility",
            "Multi-machine execution proven separately in distributed_multi_machine_proof.json",
            "No financial custody — usage costs are estimates for external billing",
            "Worker trust = operator-verified providers; not a multi-tenant sandbox",
        ],
    }

    out = Path("artifacts/external_pilot.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    md = Path("artifacts/external_pilot.md")
    md.write_text(
        "# External pilot\n\n"
        f"- success: `{artifact['success']}`\n"
        f"- provider_id: `{rec.provider_id}` verified\n"
        f"- pooled callable: `{pooled}`\n"
        f"- run target: `{artifact['run_outcome']['target']}` "
        f"cost=`{artifact['run_outcome']['actual_cost_usd']}`\n"
        f"- usage events: `{artifact['usage_events']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, default=str)[:2500])

    stop_w.set()
    stop_w2.set()
    qs.BACKGROUND_STOP.set()
    server.should_exit = True
    ap_server.should_exit = True
    return 0 if artifact["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
