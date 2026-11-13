"""Autopilot ↔ tron_efficiency bridge (mocked run_workload for default CI)."""

from __future__ import annotations

import os

import pytest

from tron_efficiency.advisor.recommend import Recommendation
from tron_efficiency.hardware.detect import HardwareInfo
from tron_efficiency.workloads.runner import RunOutcome
from tron_router.decision import PlacementDecision, PoolStatus

from tron_autopilot.efficiency_exec import policy_from_candidate, run_efficiency_workload
from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.execute import execute_plan
from tron_autopilot.store import DecisionLog, RuntimeHistory
from tron_autopilot.types import CandidateOption, WorkloadPlan, WorkloadSpec


def _hw():
    return HardwareInfo(
        platform="test",
        python_version="3.11",
        cpu_count=4,
        cpu_total_memory_bytes=8 * 1024**3,
        cuda_available=False,
        torch_version="2.x",
        gpus=[],
        warnings=[],
    )


def _rec(policy="baseline"):
    return Recommendation(
        model_name="EleutherAI/pythia-70m",
        estimated_param_count=70_000_000,
        param_count_is_estimate=True,
        device="cpu",
        dtype="fp32",
        policy=policy,
        memory_budget_bytes=100 * 1024 * 1024 if policy == "offload" else None,
        confidence="high",
        rationale=["test"],
        warnings=[],
    )


def test_policy_from_candidate_baseline_and_offload():
    base = CandidateOption(
        option_id="local-baseline",
        target="local",
        policy="baseline",
        device="cpu",
        dtype="fp32",
        memory_budget_bytes=None,
        required_memory_gb=0.3,
        hourly_rate_usd=1.0,
        predicted_runtime_seconds=1.0,
        predicted_cost_usd=0.001,
        predicted_peak_memory_bytes=1000,
        feasible=True,
    )
    assert policy_from_candidate(base).name == "baseline"

    off = CandidateOption(
        **{**base.to_dict(), "policy": "offload", "memory_budget_bytes": 50_000_000}
    )
    assert policy_from_candidate(off).name == "offload"


def test_efficiency_exec_records_metrics_from_run_workload(tmp_path, monkeypatch):
    fake = RunOutcome(
        success=True,
        policy_name="baseline",
        load_time_seconds=0.5,
        time_to_first_token_seconds=0.1,
        end_to_end_latency_seconds=1.5,
        tokens_generated=16,
        tokens_per_second=10.0,
        peak_cpu_memory_bytes=123_456_789,
        peak_gpu_memory_allocated_bytes=None,
        generated_text="hello",
        dtype="fp32",
        warnings=[],
    )

    import tron_autopilot.efficiency_exec as ee

    monkeypatch.setattr(ee, "run_workload", lambda config, policy: fake)

    chosen = CandidateOption(
        option_id="local-baseline",
        target="local",
        policy="baseline",
        device="cpu",
        dtype="fp32",
        memory_budget_bytes=None,
        required_memory_gb=0.3,
        hourly_rate_usd=3.6,
        predicted_runtime_seconds=2.0,
        predicted_cost_usd=0.002,
        predicted_peak_memory_bytes=100_000_000,
        feasible=True,
    )
    plan = WorkloadPlan(
        plan_id="plan_test",
        spec=WorkloadSpec(model_name="EleutherAI/pythia-70m", hourly_rate_usd=3.6),
        chosen=chosen,
        candidates=[chosen],
        placement_decision=None,
        local_recommendation=_rec().to_dict(),
        pool_status=None,
        safety={},
    )
    outcome = run_efficiency_workload(plan, strategy="autopilot")
    assert outcome.success is True
    assert outcome.actual_runtime_seconds == pytest.approx(2.0)  # load + e2e
    assert outcome.actual_peak_memory_bytes == 123_456_789
    assert outcome.metrics["tokens_per_second"] == 10.0
    assert outcome.metrics["runtime_basis"] == "load+e2e"
    assert outcome.actual_cost_usd == pytest.approx(3.6 / 3600.0 * 2.0)


def test_execute_plan_efficiency_flag_uses_bridge(tmp_path, monkeypatch):
    fake = RunOutcome(
        success=True,
        policy_name="baseline",
        load_time_seconds=0.2,
        end_to_end_latency_seconds=0.3,
        tokens_per_second=5.0,
        peak_cpu_memory_bytes=1000,
        generated_text="x",
    )
    import tron_autopilot.efficiency_exec as ee

    monkeypatch.setattr(ee, "run_workload", lambda config, policy: fake)

    rec = _rec()

    def recommend_fn(*a, **k):
        return rec

    def decide_fn(job, hardware=None, pool_status=None, master_url=None):
        return PlacementDecision(
            target="local",
            confidence="high",
            rationale=["local"],
            local_recommendation=rec.to_dict(),
        )

    ap = EconomicAutopilot(
        decision_log=DecisionLog(tmp_path / "d.jsonl"),
        history=RuntimeHistory(tmp_path / "h.json"),
        hardware=_hw(),
        recommend_fn=recommend_fn,
        decide_placement_fn=decide_fn,
        query_pool_status_fn=lambda master_url=None: PoolStatus(
            reachable=False, master_url="x", error="none"
        ),
    )
    spec = WorkloadSpec(model_name="EleutherAI/pythia-70m", hourly_rate_usd=1.0)
    plan = ap.plan(spec, strategy="autopilot")
    assert plan.chosen is not None
    assert plan.chosen.target == "local"
    outcome = execute_plan(plan, efficiency=True, strategy="autopilot")
    ap.record_feedback(plan, outcome)
    assert outcome.success
    assert outcome.metrics["tokens_per_second"] == 5.0
    key = RuntimeHistory.key(
        "EleutherAI/pythia-70m", "local", plan.chosen.policy, plan.chosen.device, plan.chosen.dtype
    )
    assert ap.history.mean_runtime(key) == pytest.approx(0.5)


@pytest.mark.skipif(
    os.environ.get("TRON_AUTOPILOT_RUN_MODEL_TESTS") != "1",
    reason="real model download; set TRON_AUTOPILOT_RUN_MODEL_TESTS=1",
)
def test_real_pythia_efficiency_through_autopilot(tmp_path):
    """Optional live path — same gate style as tron_efficiency real tests."""
    from tron_efficiency.hardware.detect import detect_hardware

    ap = EconomicAutopilot(
        decision_log=DecisionLog(tmp_path / "d.jsonl"),
        history=RuntimeHistory(tmp_path / "h.json"),
        hardware=detect_hardware(),
        query_pool_status_fn=lambda master_url=None: PoolStatus(
            reachable=False, master_url="x", error="none"
        ),
    )
    spec = WorkloadSpec(
        model_name="EleutherAI/pythia-70m",
        hourly_rate_usd=0.50,
        max_new_tokens=8,
        reliability="tolerant",
    )
    plan = ap.plan(spec, strategy="autopilot")
    assert plan.chosen is not None
    outcome = execute_plan(plan, efficiency=True, strategy="autopilot")
    ap.record_feedback(plan, outcome)
    assert outcome.success, outcome.error
    assert outcome.metrics.get("tokens_per_second") is not None
    assert outcome.actual_peak_memory_bytes is not None
