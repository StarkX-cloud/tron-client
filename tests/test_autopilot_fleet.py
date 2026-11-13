"""Fleet capacity + efficiency_runtime advertising."""

from __future__ import annotations

from tron_router.decision import PoolStatus

from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.fleet import FleetCapacity, WorkerCapacity, query_fleet_capacity
from tron_autopilot.store import DecisionLog, RuntimeHistory
from tron_autopilot.types import WorkloadSpec
from tron_efficiency.advisor.recommend import Recommendation
from tron_efficiency.hardware.detect import HardwareInfo
from tron_router.decision import PlacementDecision


def test_query_fleet_capacity_from_injected_workers():
    workers = {
        "a": {
            "memory_gb": 16,
            "free_memory_gb": 10,
            "gpu": False,
            "efficiency_runtime": True,
            "last_heartbeat": 1e18,
        },
        "b": {
            "memory_gb": 8,
            "gpu": True,
            "efficiency_runtime": False,
            "capabilities": {},
            "last_heartbeat": 1e18,
        },
    }
    status = PoolStatus(
        reachable=True,
        master_url="http://x",
        alive_worker_count=2,
        measured_links={"a": {"latency_ms": 5.0, "bandwidth_mbps": 100.0}},
    )
    fleet = query_fleet_capacity(
        master_url="http://x", pool_status=status, workers_payload=workers
    )
    assert fleet.reachable
    assert fleet.alive_count == 2
    assert fleet.efficiency_worker_count == 1
    assert fleet.gpu_worker_count == 1
    assert fleet.can_run_efficiency_pooled()
    assert fleet.total_memory_gb == 24.0
    d = fleet.to_dict()
    assert "dedicated provisioner" in d["limits_honestly"][1]


def test_requires_efficiency_runtime_blocks_pool_without_capable_workers(tmp_path):
    rec = Recommendation(
        model_name="m",
        estimated_param_count=1_000_000,
        param_count_is_estimate=True,
        device="cpu",
        dtype="fp32",
        policy="baseline",
        memory_budget_bytes=None,
        confidence="medium",
        rationale=[],
        warnings=[],
    )

    def recommend_fn(*a, **k):
        return rec

    pool = PoolStatus(
        reachable=True,
        master_url="http://x",
        alive_worker_count=2,
        total_reported_memory_gb=32,
        efficiency_worker_count=0,
        measured_links={"w": {"latency_ms": 1.0}},
    )

    def decide_fn(job, hardware=None, pool_status=None, master_url=None):
        return PlacementDecision(
            target="pooled",
            confidence="low",
            rationale=["test"],
            pool_status=pool.to_dict(),
            required_memory_gb=0.1,
        )

    ap = EconomicAutopilot(
        decision_log=DecisionLog(tmp_path / "d.jsonl"),
        history=RuntimeHistory(tmp_path / "h.json"),
        hardware=HardwareInfo(
            platform="t",
            python_version="3",
            cpu_count=2,
            cpu_total_memory_bytes=8 * 1024**3,
            cuda_available=False,
            torch_version=None,
            gpus=[],
            warnings=[],
        ),
        recommend_fn=recommend_fn,
        decide_placement_fn=decide_fn,
        query_pool_status_fn=lambda master_url=None: pool,
    )
    ap.history.record(
        RuntimeHistory.key("m", "local", "baseline", "cpu", "fp32"), 1.0
    )
    ap.history.record(
        RuntimeHistory.key("m", "pooled", "baseline", "cpu", "fp32"), 1.0
    )
    spec = WorkloadSpec(
        model_name="m",
        hourly_rate_usd=1.0,
        pooled_hourly_rate_usd=0.1,
        expected_runtime_seconds=1.0,
        requires_efficiency_runtime=True,
    )
    plan = ap.plan(spec, strategy="autopilot")
    pooled = next(c for c in plan.candidates if c.target == "pooled")
    assert pooled.feasible is False
    assert plan.chosen is not None
    assert plan.chosen.target == "local"
