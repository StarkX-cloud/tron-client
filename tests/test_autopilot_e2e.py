"""End-to-end Autopilot paths: local + live pooled + failure recovery evidence."""

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

from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.execute import execute_plan
from tron_autopilot.store import DecisionLog, RuntimeHistory
from tron_autopilot.types import WorkloadSpec


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_master_with_watchdog(tmp_path, monkeypatch):
    monkeypatch.setenv("TRON_SPINE_DIR", str(tmp_path / "spine"))
    import queue_server

    importlib.reload(queue_server)
    queue_server.HEARTBEAT_TIMEOUT_SECONDS = 1.0

    port = _free_port()
    config = uvicorn.Config(
        queue_server.app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 10
    while not getattr(server, "started", False) and time.time() < deadline:
        time.sleep(0.05)
    assert server.started

    threading.Thread(target=queue_server._watchdog_loop, daemon=True).start()
    threading.Thread(target=queue_server._match_loop, daemon=True).start()

    url = f"http://127.0.0.1:{port}"
    yield url, queue_server
    server.should_exit = True
    t.join(timeout=5)


def _rec():
    return Recommendation(
        model_name="fake/model",
        estimated_param_count=70_000_000,
        param_count_is_estimate=True,
        device="cpu",
        dtype="fp32",
        policy="baseline",
        memory_budget_bytes=None,
        confidence="medium",
        rationale=["e2e test"],
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


def _make_autopilot(tmp_path, master_url, pool_status):
    rec = _rec()

    def recommend_fn(*a, **k):
        return rec

    def decide_fn(job, hardware=None, pool_status=None, master_url=None):
        status = pool_status
        return PlacementDecision(
            target="pooled" if status and status.reachable else "local",
            confidence="medium",
            rationale=["e2e"],
            local_recommendation=rec.to_dict(),
            pool_status=status.to_dict() if status else None,
            required_memory_gb=0.14,
        )

    return EconomicAutopilot(
        decision_log=DecisionLog(tmp_path / "decisions.jsonl"),
        history=RuntimeHistory(tmp_path / "history.json"),
        hardware=_hw(),
        recommend_fn=recommend_fn,
        decide_placement_fn=decide_fn,
        query_pool_status_fn=lambda master_url=None: pool_status,
    )


def _register_worker(master_url, tmp_path, monkeypatch, name):
    monkeypatch.setenv("TRON_MASTER_URL", master_url)
    monkeypatch.setenv("TRON_WORKER_NAME", name)
    monkeypatch.setenv("TRON_AUTH_TOKEN_FILE", str(tmp_path / f"{name}_auth.json"))
    import worker

    importlib.reload(worker)
    worker._MP_CONTEXT = None
    return worker, worker.register_worker()


def _run_one_job(worker_mod, auth_token):
    job = None
    for _ in range(50):
        job = worker_mod.fetch_next_job(auth_token)
        if job:
            break
        time.sleep(0.05)
    assert job is not None, "worker never received a job"
    job_id = job["id"]
    t0 = time.time()
    result = worker_mod.execute_job(job)
    worker_mod.report_complete(auth_token, job_id, result, time.time() - t0)
    return result


def _multiply(a, b):
    return a * b


def test_autopilot_pooled_e2e_executes_real_callable(
    live_master_with_watchdog, tmp_path, monkeypatch
):
    master_url, qs = live_master_with_watchdog
    pool = PoolStatus(
        reachable=True,
        master_url=master_url,
        alive_worker_count=1,
        total_reported_memory_gb=16.0,
        checked_worker_count=1,
        measured_links={"w1": {"latency_ms": 5.0, "bandwidth_mbps": 200.0}},
    )
    ap = _make_autopilot(tmp_path, master_url, pool)
    ap.history.record(
        RuntimeHistory.key("fake/model", "pooled", "baseline", "cpu", "fp32"), 0.2
    )
    # Prefer pool via cheaper rate
    spec = WorkloadSpec(
        model_name="fake/model",
        hourly_rate_usd=2.0,
        pooled_hourly_rate_usd=0.2,
        expected_runtime_seconds=0.2,
        master_url=master_url,
        reliability="tolerant",
    )
    plan = ap.plan(spec, pool_status=pool, strategy="autopilot")
    assert plan.chosen is not None
    assert plan.chosen.target == "pooled"

    worker_mod, auth = _register_worker(master_url, tmp_path, monkeypatch, "ap-w1")

    outcome_box = {}

    def waiter():
        outcome_box["o"] = execute_plan(
            plan,
            func=_multiply,
            func_args=(6, 7),
            wait_timeout_seconds=15.0,
            strategy="autopilot",
        )

    wt = threading.Thread(target=waiter, daemon=True)
    wt.start()
    time.sleep(0.1)
    result = _run_one_job(worker_mod, auth)
    assert result["status"] == "ok"
    assert result["result"] == 42
    wt.join(timeout=15)
    outcome = outcome_box["o"]
    ap.record_feedback(plan, outcome)
    assert outcome.success is True
    assert outcome.result == 42
    assert outcome.job_id is not None
    assert outcome.actual_cost_usd is not None
    assert outcome.recovery["mechanism"] == "queue_server_watchdog_requeue"


def test_autopilot_recovery_after_dead_worker(
    live_master_with_watchdog, tmp_path, monkeypatch
):
    """Evidence that Autopilot pooled execution still completes via the
    existing watchdog when the first worker dies mid-job — Autopilot does
    not implement a second retry loop."""
    master_url, qs = live_master_with_watchdog
    pool = PoolStatus(
        reachable=True,
        master_url=master_url,
        alive_worker_count=2,
        total_reported_memory_gb=32.0,
        checked_worker_count=2,
        measured_links={
            "dead": {"latency_ms": 5.0, "bandwidth_mbps": 100.0},
            "alive": {"latency_ms": 8.0, "bandwidth_mbps": 100.0},
        },
    )
    ap = _make_autopilot(tmp_path, master_url, pool)
    ap.history.record(
        RuntimeHistory.key("fake/model", "pooled", "baseline", "cpu", "fp32"), 0.5
    )
    spec = WorkloadSpec(
        model_name="fake/model",
        hourly_rate_usd=1.0,
        pooled_hourly_rate_usd=0.2,
        expected_runtime_seconds=0.5,
        master_url=master_url,
    )
    plan = ap.plan(spec, pool_status=pool, strategy="autopilot")
    assert plan.chosen.target == "pooled"

    # Register only the doomed worker first so the match loop cannot hand
    # the job to a live rescuer before we observe the mid-job death
    # (same ordering as tests/test_pool_reliability.py).
    dead, dead_auth = _register_worker(master_url, tmp_path, monkeypatch, "dead")

    outcome_box = {}

    def waiter():
        outcome_box["o"] = execute_plan(
            plan,
            func=_multiply,
            func_args=(3, 11),
            wait_timeout_seconds=20.0,
            strategy="autopilot",
        )

    threading.Thread(target=waiter, daemon=True).start()
    time.sleep(0.1)

    # Dead worker claims the job and then vanishes (no complete, no heartbeat).
    claimed = None
    for _ in range(100):
        claimed = dead.fetch_next_job(dead_auth)
        if claimed:
            break
        time.sleep(0.05)
    assert claimed is not None, "doomed worker never received the Autopilot pooled job"
    job_id = claimed["id"]

    # Wait for watchdog to requeue (heartbeat timeout 1s + poll).
    deadline = time.time() + 15
    back_in_queue = False
    while time.time() < deadline:
        with qs.lock:
            back_in_queue = any(j["id"] == job_id for j in qs.job_queue)
        if back_in_queue:
            break
        time.sleep(0.25)
    assert back_in_queue, "watchdog never requeued the dead worker's job"

    alive, alive_auth = _register_worker(master_url, tmp_path, monkeypatch, "alive")
    rescued = None
    deadline = time.time() + 10
    while time.time() < deadline:
        rescued = alive.fetch_next_job(alive_auth)
        if rescued:
            break
        time.sleep(0.05)
    assert rescued is not None and rescued["id"] == job_id
    # requeue_count lives on the job object returned to the worker, not on
    # job_store after completion (same as queue_server._watchdog_loop).
    assert rescued.get("requeue_count", 0) >= 1
    t0 = time.time()
    result = alive.execute_job(rescued)
    alive.report_complete(alive_auth, job_id, result, time.time() - t0)
    assert result["status"] == "ok"
    assert result["result"] == 33

    deadline = time.time() + 10
    while "o" not in outcome_box and time.time() < deadline:
        time.sleep(0.05)
    outcome = outcome_box["o"]
    assert outcome.success is True
    assert outcome.result == 33
    assert outcome.recovery["mechanism"] == "queue_server_watchdog_requeue"
    with qs.lock:
        stored = dict(qs.job_store.get(job_id) or {})
    assert stored.get("status") == "completed"
    # Spine event log must record the requeue (authoritative recovery evidence).
    events = qs.event_log.events_for(job_id)
    types = [e.type for e in events]
    assert "requeued" in types


def test_benchmark_strategies_run_locally(tmp_path):
    """fixed/cheapest/autopilot all produce outcomes on a synthetic local path."""
    from benchmarks.economic_autopilot import run_benchmark

    payload = run_benchmark(
        trials=1,
        hourly_rate=0.5,
        pooled_hourly_rate=0.2,
        live_pool=False,
        out_path=str(tmp_path / "bench.json"),
        duration_seconds=0.02,
    )
    assert set(payload["outcomes"]) == {"fixed", "cheapest", "autopilot"}
    for s in ("fixed", "cheapest", "autopilot"):
        assert payload["outcomes"][s][0]["success"] is True
    assert (tmp_path / "bench.md").exists()
