"""Autopilot distributed candidate feasibility (no live pool)."""

from tron_efficiency.advisor.recommend import Recommendation
from tron_efficiency.hardware.detect import HardwareInfo
from tron_router.decision import PlacementDecision, PoolStatus

from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.store import DecisionLog, RuntimeHistory
from tron_autopilot.types import WorkloadSpec


def _rec(name="m"):
    return Recommendation(
        model_name=name,
        estimated_param_count=1000,
        param_count_is_estimate=True,
        device="cpu",
        dtype="fp32",
        policy="baseline",
        memory_budget_bytes=None,
        confidence="medium",
        rationale=["t"],
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


def _ap(tmp_path, pool):
    rec = _rec()

    def recommend_fn(*a, **k):
        return rec

    def decide_fn(job, hardware=None, pool_status=None, master_url=None):
        return PlacementDecision(
            target="pooled",
            confidence="medium",
            rationale=["t"],
            required_memory_gb=0.25,
        )

    return EconomicAutopilot(
        decision_log=DecisionLog(tmp_path / "d.jsonl"),
        history=RuntimeHistory(tmp_path / "h.json"),
        hardware=_hw(),
        recommend_fn=recommend_fn,
        decide_placement_fn=decide_fn,
        query_pool_status_fn=lambda master_url=None: pool,
    )


def test_distributed_refused_without_parallelizable(tmp_path):
    pool = PoolStatus(
        reachable=True,
        master_url="http://127.0.0.1:9",
        alive_worker_count=2,
        total_reported_memory_gb=8.0,
        checked_worker_count=2,
    )
    ap = _ap(tmp_path, pool)
    plan = ap.plan(
        WorkloadSpec(
            model_name="m",
            hourly_rate_usd=0.5,
            parallelizable=False,
            partition_items=list(range(10)),
            master_url=pool.master_url,
        ),
        pool_status=pool,
    )
    dist = next(c for c in plan.candidates if c.target == "distributed")
    assert not dist.feasible
    assert any("parallelizable" in r for r in dist.rejection_reasons)


def test_distributed_refused_when_comm_dominates(tmp_path):
    pool = PoolStatus(
        reachable=True,
        master_url="http://127.0.0.1:9",
        alive_worker_count=4,
        total_reported_memory_gb=16.0,
        checked_worker_count=4,
        measured_links={
            "a": {"latency_ms": 800.0, "bandwidth_mbps": 0.5},
            "b": {"latency_ms": 900.0, "bandwidth_mbps": 0.5},
        },
    )
    ap = _ap(tmp_path, pool)
    # Tiny serial → communication overhead wins
    plan = ap.plan(
        WorkloadSpec(
            model_name="m",
            hourly_rate_usd=0.5,
            parallelizable=True,
            partition_items=list(range(20)),
            expected_runtime_seconds=0.05,
            bytes_per_item=1_000_000.0,
            master_url=pool.master_url,
        ),
        pool_status=pool,
    )
    dist = next(c for c in plan.candidates if c.target == "distributed")
    assert not dist.feasible
    assert any("communication overhead" in r for r in dist.rejection_reasons)


def test_distributed_feasible_when_speedup_clear(tmp_path):
    pool = PoolStatus(
        reachable=True,
        master_url="http://127.0.0.1:9",
        alive_worker_count=4,
        total_reported_memory_gb=16.0,
        checked_worker_count=4,
        measured_links={
            "a": {"latency_ms": 5.0, "bandwidth_mbps": 200.0},
            "b": {"latency_ms": 6.0, "bandwidth_mbps": 200.0},
        },
    )
    ap = _ap(tmp_path, pool)
    ap.history.record(
        RuntimeHistory.key("m", "local", "baseline", "cpu", "fp32"), 40.0
    )
    plan = ap.plan(
        WorkloadSpec(
            model_name="m",
            hourly_rate_usd=0.5,
            parallelizable=True,
            partition_items=list(range(1000)),
            expected_runtime_seconds=40.0,
            bytes_per_item=8.0,
            master_url=pool.master_url,
        ),
        pool_status=pool,
    )
    dist = next(c for c in plan.candidates if c.target == "distributed")
    assert dist.feasible, dist.rejection_reasons
    assert dist.predicted_runtime_seconds is not None
    assert dist.predicted_runtime_seconds < 40.0
