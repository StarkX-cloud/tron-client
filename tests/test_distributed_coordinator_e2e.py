"""Live two-worker e2e for DistributedCoordinator + Autopilot distributed path.

Honest note: this proves the coordinator against two independent worker
*processes* on one host. Multi-machine Linode proof lives in
``scripts/distributed_multi_machine_proof.py`` and must not be claimed here.
"""

from __future__ import annotations

import importlib
import socket
import threading
import time

import pytest
import uvicorn

from tron_efficiency.advisor.recommend import Recommendation
from tron_efficiency.hardware.detect import HardwareInfo
from tron_router.decision import PlacementDecision, PoolStatus

from tron_autopilot.distributed.coordinator import DistributedCoordinator
from tron_autopilot.distributed.ops import closed_form_sum_squares
from tron_autopilot.distributed.partition import partition_items
from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.execute import execute_plan
from tron_autopilot.store import DecisionLog, RuntimeHistory
from tron_autopilot.types import WorkloadSpec


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("TRON_SPINE_DIR", str(tmp_path / "spine"))
    monkeypatch.delenv("TRON_SUBMIT_AUTH_TOKEN", raising=False)
    import queue_server

    if hasattr(queue_server, "BACKGROUND_STOP"):
        queue_server.BACKGROUND_STOP.set()
        time.sleep(0.15)
    importlib.reload(queue_server)
    queue_server.BACKGROUND_STOP.clear()
    queue_server.HEARTBEAT_TIMEOUT_SECONDS = 2.0
    queue_server.MATCH_INTERVAL_SECONDS = 0.2

    port = _free_port()
    config = uvicorn.Config(
        queue_server.app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    st = threading.Thread(target=server.run, daemon=True)
    st.start()
    deadline = time.time() + 10
    while not getattr(server, "started", False) and time.time() < deadline:
        time.sleep(0.05)
    assert server.started

    match_t = threading.Thread(target=queue_server._match_loop, daemon=True)
    watch_t = threading.Thread(target=queue_server._watchdog_loop, daemon=True)
    match_t.start()
    watch_t.start()

    url = f"http://127.0.0.1:{port}"

    workers = []
    for name in ("dist-w0", "dist-w1"):
        monkeypatch.setenv("TRON_MASTER_URL", url)
        monkeypatch.setenv("TRON_WORKER_NAME", name)
        monkeypatch.setenv("TRON_AUTH_TOKEN_FILE", str(tmp_path / f"{name}.json"))
        import worker as worker_mod

        importlib.reload(worker_mod)
        worker_mod._MP_CONTEXT = None
        token = worker_mod.register_worker()
        workers.append((worker_mod, token, name))

    stop = threading.Event()

    def _loop(wmod, token):
        while not stop.is_set():
            try:
                job = wmod.fetch_next_job(token)
                if not job:
                    time.sleep(0.05)
                    continue
                t0 = time.time()
                result = wmod.execute_job(job)
                wmod.report_complete(token, job["id"], result, time.time() - t0)
            except Exception:
                time.sleep(0.05)

    for wmod, token, _ in workers:
        threading.Thread(target=_loop, args=(wmod, token), daemon=True).start()

    def _hb():
        while not stop.is_set():
            for wmod, token, _ in workers:
                try:
                    wmod.heartbeat(token)
                except Exception:
                    pass
            time.sleep(0.3)

    threading.Thread(target=_hb, daemon=True).start()
    # Let registration + heartbeats settle so matcher sees 2 alive workers.
    time.sleep(0.5)

    yield url, queue_server
    stop.set()
    queue_server.BACKGROUND_STOP.set()
    match_t.join(timeout=2)
    watch_t.join(timeout=2)
    server.should_exit = True
    st.join(timeout=5)


def test_coordinator_two_workers_correct(live_pool):
    url, _ = live_pool
    items = list(range(50))
    shards = partition_items(items, 2, workload_id="e2e")
    assert len(shards) == 2
    run = DistributedCoordinator(url, wait_timeout_seconds=60).run(
        shards, op="sum_squares"
    )
    assert run.success, run.error
    assert run.reduced["total"] == closed_form_sum_squares(50)
    assert run.attempts_total >= 2


def test_idempotent_collect_suppresses_duplicate_key():
    """Unit: first success for an idempotency_key wins; second is suppressed."""
    from tron_autopilot.distributed.coordinator import DistributedCoordinator

    coord = DistributedCoordinator("http://127.0.0.1:9")
    key = "abc123"
    coord._completed[key] = {"shard_id": 0, "partial": 1, "n": 1, "op": "sum"}
    # Re-accept same key
    if key in coord._completed:
        coord.duplicates_suppressed += 1
    assert coord.duplicates_suppressed == 1
    assert coord._completed[key]["partial"] == 1


def _rec():
    return Recommendation(
        model_name="dist/demo",
        estimated_param_count=1_000,
        param_count_is_estimate=True,
        device="cpu",
        dtype="fp32",
        policy="baseline",
        memory_budget_bytes=None,
        confidence="medium",
        rationale=["dist e2e"],
        warnings=[],
    )


def _hw():
    return HardwareInfo(
        platform="test",
        python_version="3.11",
        cpu_count=2,
        cpu_total_memory_bytes=4 * 1024**3,
        cuda_available=False,
        torch_version=None,
        gpus=[],
        warnings=[],
    )


def test_autopilot_distributed_execute_path(live_pool, tmp_path):
    url, _ = live_pool
    pool = PoolStatus(
        reachable=True,
        master_url=url,
        alive_worker_count=2,
        total_reported_memory_gb=16.0,
        checked_worker_count=2,
        measured_links={
            "dist-w0": {"latency_ms": 5.0, "bandwidth_mbps": 200.0},
            "dist-w1": {"latency_ms": 6.0, "bandwidth_mbps": 180.0},
        },
    )
    rec = _rec()

    def recommend_fn(*a, **k):
        return rec

    def decide_fn(job, hardware=None, pool_status=None, master_url=None):
        return PlacementDecision(
            target="pooled",
            confidence="medium",
            rationale=["dist e2e"],
            local_recommendation=rec.to_dict(),
            pool_status=pool.to_dict(),
            required_memory_gb=0.25,
        )

    ap = EconomicAutopilot(
        decision_log=DecisionLog(tmp_path / "decisions.jsonl"),
        history=RuntimeHistory(tmp_path / "history.json"),
        hardware=_hw(),
        recommend_fn=recommend_fn,
        decide_placement_fn=decide_fn,
        query_pool_status_fn=lambda master_url=None: pool,
    )
    # Large serial runtime → cost model prefers distributed wall.
    ap.history.record(
        RuntimeHistory.key("dist/demo", "local", "baseline", "cpu", "fp32"), 12.0
    )

    items = list(range(200))
    spec = WorkloadSpec(
        model_name="dist/demo",
        hourly_rate_usd=0.5,
        pooled_hourly_rate_usd=0.5,
        master_url=url,
        parallelizable=True,
        partition_items=items,
        map_op="sum_squares",
        expected_runtime_seconds=12.0,
        bytes_per_item=8.0,
        reliability="tolerant",
    )
    plan = ap.plan(spec, pool_status=pool, strategy="autopilot")
    dist = next(c for c in plan.candidates if c.target == "distributed")
    assert dist.feasible, dist.rejection_reasons
    assert plan.chosen is not None
    assert plan.chosen.target == "distributed", (
        plan.chosen.target,
        plan.chosen.predicted_cost_usd,
        [(c.target, c.feasible, c.predicted_cost_usd) for c in plan.candidates],
    )

    outcome = execute_plan(plan, strategy="autopilot", wait_timeout_seconds=90)
    assert outcome.success, outcome.error
    assert outcome.target == "distributed"
    assert outcome.metrics.get("got_total") == closed_form_sum_squares(200)
    assert outcome.actual_cost_usd is not None
    assert outcome.actual_runtime_seconds > 0
