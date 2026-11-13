"""Unit tests for Economic Autopilot decision + feedback (no live master)."""

from __future__ import annotations

from tron_efficiency.advisor.recommend import Recommendation
from tron_efficiency.hardware.detect import HardwareInfo
from tron_router.decision import PoolStatus

from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.execute import execute_plan
from tron_autopilot.store import DecisionLog, RuntimeHistory
from tron_autopilot.types import WorkloadSpec


def _hw():
    return HardwareInfo(
        platform="test",
        python_version="3.11",
        cpu_count=4,
        cpu_total_memory_bytes=8 * 1024**3,
        cuda_available=False,
        torch_version=None,
        gpus=[],
        warnings=[],
    )


def _rec(policy="baseline", device="cpu", dtype="fp32", params=70_000_000):
    return Recommendation(
        model_name="fake/model",
        estimated_param_count=params,
        param_count_is_estimate=True,
        device=device,
        dtype=dtype,
        policy=policy,
        memory_budget_bytes=100 * 1024 * 1024 if policy == "offload" else None,
        confidence="high" if policy == "baseline" else "medium",
        rationale=["test recommendation"],
        warnings=[],
    )


def _empty_pool():
    return PoolStatus(reachable=False, master_url="http://127.0.0.1:9", error="no pool")


def _alive_pool(workers=2, mem_gb=32.0, rate_links=True):
    links = None
    if rate_links:
        links = {
            f"w{i}": {"latency_ms": 10.0 * (i + 1), "bandwidth_mbps": 100.0}
            for i in range(workers)
        }
    return PoolStatus(
        reachable=True,
        master_url="http://127.0.0.1:9000",
        alive_worker_count=workers,
        total_reported_memory_gb=mem_gb,
        checked_worker_count=workers,
        measured_links=links,
    )


def _autopilot(tmp_path, recommend_return=None, pool=None):
    rec = recommend_return or _rec()

    def recommend_fn(*a, **k):
        return rec

    def decide_fn(job, hardware=None, pool_status=None, master_url=None):
        from tron_router.decision import PlacementDecision

        status = pool_status or pool or _empty_pool()
        target = "local" if rec.policy == "baseline" else (
            "pooled" if status.reachable and status.alive_worker_count else "dedicated"
        )
        return PlacementDecision(
            target=target,
            confidence="medium",
            rationale=["test placement"],
            warnings=[],
            local_recommendation=rec.to_dict(),
            pool_status=status.to_dict(),
            required_memory_gb=0.14,
        )

    return EconomicAutopilot(
        decision_log=DecisionLog(tmp_path / "decisions.jsonl"),
        history=RuntimeHistory(tmp_path / "history.json"),
        hardware=_hw(),
        recommend_fn=recommend_fn,
        decide_placement_fn=decide_fn,
        query_pool_status_fn=lambda master_url=None: pool or _empty_pool(),
    )


def test_autopilot_prefers_cheaper_feasible_pool_when_rates_favor_it(tmp_path):
    ap = _autopilot(tmp_path, recommend_return=_rec("baseline"), pool=_alive_pool())
    # Seed history so predicted costs exist for both targets.
    ap.history.record(
        RuntimeHistory.key("fake/model", "local", "baseline", "cpu", "fp32"), 1.0
    )
    ap.history.record(
        RuntimeHistory.key("fake/model", "pooled", "baseline", "cpu", "fp32"), 1.0
    )
    spec = WorkloadSpec(
        model_name="fake/model",
        hourly_rate_usd=1.0,
        pooled_hourly_rate_usd=0.1,
        expected_runtime_seconds=1.0,
        budget_usd=1.0,
        deadline_seconds=10.0,
    )
    plan = ap.plan(spec, strategy="autopilot")
    assert plan.chosen is not None
    assert plan.chosen.target == "pooled"
    assert plan.chosen.predicted_cost_usd is not None
    assert plan.chosen.predicted_cost_usd < 0.001  # 0.1$/hr * 1s


def test_fixed_strategy_forces_local_baseline(tmp_path):
    ap = _autopilot(tmp_path, recommend_return=_rec("offload"), pool=_alive_pool())
    ap.history.record(
        RuntimeHistory.key("fake/model", "local", "baseline", "cpu", "fp32"), 1.0
    )
    spec = WorkloadSpec(
        model_name="fake/model",
        hourly_rate_usd=1.0,
        pooled_hourly_rate_usd=0.01,
        expected_runtime_seconds=1.0,
    )
    plan = ap.plan(spec, strategy="fixed")
    assert plan.chosen is not None
    assert plan.chosen.target == "local"
    assert plan.chosen.policy == "baseline"


def test_strict_reliability_excludes_pool(tmp_path):
    ap = _autopilot(tmp_path, pool=_alive_pool())
    ap.history.record(
        RuntimeHistory.key("fake/model", "local", "baseline", "cpu", "fp32"), 1.0
    )
    ap.history.record(
        RuntimeHistory.key("fake/model", "pooled", "baseline", "cpu", "fp32"), 1.0
    )
    spec = WorkloadSpec(
        model_name="fake/model",
        hourly_rate_usd=1.0,
        pooled_hourly_rate_usd=0.01,
        reliability="strict",
        expected_runtime_seconds=1.0,
    )
    plan = ap.plan(spec, strategy="autopilot")
    assert plan.chosen is not None
    assert plan.chosen.target == "local"
    pooled = next(c for c in plan.candidates if c.target == "pooled")
    assert pooled.feasible is False


def test_budget_makes_expensive_option_infeasible(tmp_path):
    ap = _autopilot(tmp_path, pool=_empty_pool())
    ap.history.record(
        RuntimeHistory.key("fake/model", "local", "baseline", "cpu", "fp32"), 100.0
    )
    spec = WorkloadSpec(
        model_name="fake/model",
        hourly_rate_usd=36.0,  # $1 per 100s
        expected_runtime_seconds=100.0,
        budget_usd=0.5,
    )
    plan = ap.plan(spec, strategy="autopilot")
    local = next(c for c in plan.candidates if c.option_id == "local-baseline")
    assert local.feasible is False
    assert plan.chosen is None


def test_local_execute_records_feedback_and_history(tmp_path):
    ap = _autopilot(tmp_path, pool=_empty_pool())
    spec = WorkloadSpec(
        model_name="fake/model",
        hourly_rate_usd=3.6,
        expected_runtime_seconds=0.05,
    )
    plan = ap.plan(spec, strategy="autopilot")
    assert plan.chosen is not None

    def work():
        return 42

    outcome = execute_plan(plan, func=work, strategy="autopilot")
    ap.record_feedback(plan, outcome)
    assert outcome.success is True
    assert outcome.result == 42
    assert outcome.actual_cost_usd is not None
    fb = outcome.feedback_delta()
    assert "runtime_seconds_delta" in fb

    key = RuntimeHistory.key("fake/model", "local", "baseline", "cpu", "fp32")
    assert ap.history.mean_runtime(key) is not None

    rows = ap.decision_log.read_all()
    kinds = [r["kind"] for r in rows]
    assert "plan" in kinds and "outcome" in kinds


def test_plan_includes_safety_contract(tmp_path):
    ap = _autopilot(tmp_path)
    spec = WorkloadSpec(model_name="fake/model", hourly_rate_usd=1.0)
    plan = ap.plan(spec)
    assert plan.safety["pooled_failure_recovery"] == "watchdog_requeue"
    assert plan.safety["distributed_vram"] == "not claimed — not demonstrated"
    assert "checkpoint_recovery" in plan.safety
