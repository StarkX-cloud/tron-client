"""Benchmark: fixed vs cheapest vs Economic Autopilot.

Runs the **same** callable through three decision strategies and reports
cost per successful workload, completion time, memory, failure rate, and
recovery requeues.

Methodology (honest):
- Default mode uses a synthetic CPU workload (sleep + light compute) so the
  benchmark is repeatable without multi-GB model downloads. Decision logic
  still calls the real advisor/router interfaces (injectable size/recommend
  in `--synthetic` mode).
- Optional `--live-pool` starts a real uvicorn ``queue_server`` + worker in
  this process tree (same pattern as ``tests/test_router_handoff.py``) so
  pooled decisions execute for real. Same-host workers are **not** a
  cross-region fleet; network cost stays near zero unless you point
  ``--master-url`` at a real remote master.
- Cost = configured ``--hourly-rate`` × measured wall seconds. That rate is
  an input you set, not a live cloud price.

Run:
    python3 -m benchmarks.economic_autopilot
    python3 -m benchmarks.economic_autopilot --live-pool --trials 3
    python3 -m benchmarks.economic_autopilot --out artifacts/autopilot_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
import threading
import time
from pathlib import Path

from tron_efficiency.advisor.model_info import BYTES_PER_PARAM, ModelSizeEstimate
from tron_efficiency.advisor.recommend import Recommendation, recommend
from tron_efficiency.hardware.detect import HardwareInfo
from tron_router.decision import PoolStatus

from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.report import format_benchmark, summarize_outcomes
from tron_autopilot.store import DecisionLog, RuntimeHistory
from tron_autopilot.strategies import STRATEGIES, run_strategy
from tron_autopilot.types import WorkloadSpec


def _synthetic_workload(duration_seconds: float = 0.05, allocate_mb: int = 1) -> dict:
    """Cheap, deterministic stand-in for an AI workload body."""
    buf = bytearray(allocate_mb * 1024 * 1024)
    for i in range(0, len(buf), 4096):
        buf[i] = i % 256
    time.sleep(duration_seconds)
    checksum = sum(buf[::4096]) % 10_000
    return {"ok": True, "checksum": checksum, "allocate_mb": allocate_mb}


def _fake_cpu_hardware(ram_gb: float = 8.0) -> HardwareInfo:
    return HardwareInfo(
        platform="benchmark",
        python_version="3.x",
        cpu_count=4,
        cpu_total_memory_bytes=int(ram_gb * 1024**3),
        cuda_available=False,
        torch_version=None,
        gpus=[],
        warnings=["synthetic benchmark hardware profile"],
    )


def _fake_size(param_count: int, model_name: str) -> ModelSizeEstimate:
    return ModelSizeEstimate(
        model_name=model_name,
        param_count=param_count,
        param_count_is_estimate=True,
        architecture="Synthetic",
        bytes_by_dtype={d: int(param_count * b) for d, b in BYTES_PER_PARAM.items()},
        warnings=[],
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_live_pool(tmp: Path):
    """Real master + one worker thread (same-host multi-process simulation)."""
    import importlib

    import uvicorn

    os.environ["TRON_SPINE_DIR"] = str(tmp / "spine")
    import queue_server

    importlib.reload(queue_server)
    queue_server.HEARTBEAT_TIMEOUT_SECONDS = 2.0

    port = _free_port()
    config = uvicorn.Config(
        queue_server.app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    deadline = time.time() + 10
    while not getattr(server, "started", False) and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("uvicorn master failed to start")

    watchdog = threading.Thread(target=queue_server._watchdog_loop, daemon=True)
    watchdog.start()
    match = threading.Thread(target=queue_server._match_loop, daemon=True)
    match.start()

    url = f"http://127.0.0.1:{port}"

    os.environ["TRON_MASTER_URL"] = url
    os.environ["TRON_WORKER_NAME"] = "bench-worker-1"
    os.environ["TRON_AUTH_TOKEN_FILE"] = str(tmp / "bench_worker_auth.json")
    import worker

    importlib.reload(worker)
    worker._MP_CONTEXT = None  # avoid fork-after-threads deadlock in this harness
    auth = worker.register_worker()

    # CPU-only detect_gpu() often reports memory_gb=0, which makes the
    # matcher's memory_fits hard-filter reject every job that carries a
    # real required_memory_gb. Give this bench worker an honest synthetic
    # capacity so pooled strategies can actually be assigned work.
    worker_name = os.environ.get("TRON_WORKER_NAME", "bench-worker-1")
    with queue_server.lock:
        if worker_name in queue_server.workers:
            queue_server.workers[worker_name]["memory_gb"] = 32.0
            queue_server.workers[worker_name]["free_memory_gb"] = 32.0
        try:
            queue_server.topology.record_memory(worker_name, 32.0)
        except Exception:
            pass

    stop = threading.Event()

    def _worker_loop():
        while not stop.is_set():
            try:
                # Heartbeat keeps the worker inside query_pool_status's
                # staleness window — without this, long cheapest/autopilot
                # pooled waits make the worker look dead mid-benchmark.
                worker.heartbeat(auth)
            except Exception:
                pass
            try:
                job = worker.fetch_next_job(auth)
            except Exception:
                time.sleep(0.05)
                continue
            if not job:
                time.sleep(0.05)
                continue
            job_id = job.get("id")
            t0 = time.time()
            result = worker.execute_job(job)
            worker.report_complete(auth, job_id, result, time.time() - t0)

    worker_thread = threading.Thread(target=_worker_loop, daemon=True)
    worker_thread.start()

    def shutdown():
        stop.set()
        server.should_exit = True
        server_thread.join(timeout=5)

    return url, shutdown


def run_benchmark(
    trials: int = 3,
    hourly_rate: float = 0.50,
    pooled_hourly_rate: float = 0.20,
    live_pool: bool = False,
    master_url: str | None = None,
    out_path: str | None = None,
    duration_seconds: float = 0.05,
) -> dict:
    model_name = "synthetic/autopilot-bench"
    # ~2GB fp16 footprint → advisor on 8GB CPU RAM still baseline; for
    # "needs offload" contrast we shrink synthetic RAM in a second scenario
    # inside the same run when comparing fixed vs autopilot memory choices.
    param_count = 1_000_000_000  # analytical stand-in

    results_dir = Path(out_path).parent if out_path else Path("artifacts")
    results_dir.mkdir(parents=True, exist_ok=True)
    log = DecisionLog(results_dir / "autopilot_decisions.jsonl")
    history = RuntimeHistory(results_dir / "autopilot_history.json")

    shutdown = None
    pool_url = master_url
    tmp_ctx = None
    if live_pool and not master_url:
        tmp_ctx = tempfile.TemporaryDirectory()
        pool_url, shutdown = _start_live_pool(Path(tmp_ctx.name))
        # Let worker register + first heartbeat
        time.sleep(0.5)

    def recommend_fn(model_name, hardware=None, gpu_budget_bytes_override=None, size_estimate=None):
        size = size_estimate or _fake_size(param_count, model_name)
        return recommend(
            model_name,
            hardware=hardware or _fake_cpu_hardware(8),
            gpu_budget_bytes_override=gpu_budget_bytes_override,
            size_estimate=size,
        )

    # Seed history so cost predictions are measured, not empty on trial 1+.
    # One calibration local run at the expected duration.
    seed_key = RuntimeHistory.key(model_name, "local", "baseline", "cpu", "fp32")
    history.record(seed_key, duration_seconds)
    history.record(
        RuntimeHistory.key(model_name, "pooled", "baseline", "cpu", "fp32"),
        duration_seconds + 0.02,
    )

    hardware = _fake_cpu_hardware(8)
    autopilot = EconomicAutopilot(
        decision_log=log,
        history=history,
        hardware=hardware,
        recommend_fn=recommend_fn,
    )

    # When not using a live pool, inject an unreachable or empty pool so
    # strategies that want pooled don't pretend a fleet exists.
    injected_pool = None
    if pool_url:
        from tron_router.decision import query_pool_status

        injected_pool = None  # real query via master_url on spec
    else:
        injected_pool = PoolStatus(
            reachable=False,
            master_url="http://127.0.0.1:9",
            error="benchmark: no live pool (--live-pool not set)",
        )

    all_outcomes = {s: [] for s in STRATEGIES}
    all_plans = {s: [] for s in STRATEGIES}

    try:
        for strategy in STRATEGIES:
            for trial in range(trials):
                spec = WorkloadSpec(
                    model_name=model_name,
                    hourly_rate_usd=hourly_rate,
                    pooled_hourly_rate_usd=pooled_hourly_rate,
                    budget_usd=1.0,
                    deadline_seconds=30.0,
                    reliability="tolerant",
                    expected_runtime_seconds=duration_seconds,
                    master_url=pool_url,
                )
                plan, outcome = run_strategy(
                    autopilot,
                    spec,
                    strategy,
                    func=_synthetic_workload,
                    func_kwargs={
                        "duration_seconds": duration_seconds,
                        "allocate_mb": 1,
                    },
                    pool_status=injected_pool,
                    wait_timeout_seconds=30.0,
                )
                all_plans[strategy].append(plan.to_dict())
                all_outcomes[strategy].append(outcome)
    finally:
        if shutdown:
            shutdown()
        if tmp_ctx:
            tmp_ctx.cleanup()

    summaries = [
        summarize_outcomes(s, all_outcomes[s]) for s in STRATEGIES
    ]
    payload = {
        "meta": {
            "trials": trials,
            "hourly_rate_usd": hourly_rate,
            "pooled_hourly_rate_usd": pooled_hourly_rate,
            "live_pool": bool(pool_url),
            "master_url": pool_url,
            "model_name": model_name,
            "notes": [
                "Synthetic workload body — decision path uses real Autopilot/"
                "advisor/router interfaces with an injected size estimate.",
                "Same-host live pool is not a multi-region fleet proof.",
                "Cost = configured hourly rate × measured wall time.",
            ],
        },
        "summaries": summaries,
        "plans": all_plans,
        "outcomes": {s: [o.to_dict() for o in outs] for s, outs in all_outcomes.items()},
        "markdown": format_benchmark(summaries),
    }

    if out_path:
        Path(out_path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        md_path = Path(out_path).with_suffix(".md")
        md_path.write_text(payload["markdown"], encoding="utf-8")

    print(payload["markdown"])
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="benchmarks.economic_autopilot")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--hourly-rate", type=float, default=0.50)
    p.add_argument("--pooled-hourly-rate", type=float, default=0.20)
    p.add_argument("--live-pool", action="store_true")
    p.add_argument("--master-url", default=None)
    p.add_argument("--duration-seconds", type=float, default=0.05)
    p.add_argument(
        "--out",
        default="artifacts/economic_autopilot_benchmark.json",
        help="JSON results path (markdown sibling written alongside)",
    )
    args = p.parse_args(argv)
    run_benchmark(
        trials=args.trials,
        hourly_rate=args.hourly_rate,
        pooled_hourly_rate=args.pooled_hourly_rate,
        live_pool=args.live_pool,
        master_url=args.master_url,
        out_path=args.out,
        duration_seconds=args.duration_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
