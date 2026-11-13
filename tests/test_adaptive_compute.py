"""Unit + light integration tests for Adaptive Compute Intelligence."""

from __future__ import annotations

import time

from tron_efficiency.advisor.recommend import Recommendation
from tron_efficiency.hardware.detect import HardwareInfo

from tron_adaptive.batching import choose_batch_size, group_items
from tron_adaptive.loop import AdaptiveIntelligence, close_feedback_loop
from tron_adaptive.metrics import aggregate_run_metrics
from tron_adaptive.optimizer import AdaptiveOptimizer
from tron_adaptive.performance_store import PerformanceSample, PerformanceStore
from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.store import DecisionLog, RuntimeHistory
from tron_autopilot.types import ExecutionOutcome, WorkloadPlan, WorkloadSpec


def _hw(cuda: bool = False) -> HardwareInfo:
    return HardwareInfo(
        platform="test",
        python_version="3.11",
        cpu_count=2,
        cpu_total_memory_bytes=8 * 1024**3,
        cuda_available=cuda,
        torch_version=None,
        gpus=[],
        warnings=[],
    )


def test_batching_defaults_without_history():
    plan = choose_batch_size(10, measured_peak_memory_bytes=None, memory_budget_bytes=None)
    assert plan.batch_size == 1
    assert group_items(list(range(5)), 2) == [[0, 1], [2, 3], [4]]


def test_batching_uses_history():
    plan = choose_batch_size(
        20,
        measured_peak_memory_bytes=100,
        memory_budget_bytes=1000,
        history_best_batch=4,
        max_batch_size=8,
    )
    assert plan.batch_size == 4


def test_unoptimized_profile_is_naive():
    opt = AdaptiveOptimizer()
    spec = WorkloadSpec(model_name="m", hourly_rate_usd=0.5)
    p = opt.unoptimized_profile(spec)
    assert p.mode == "unoptimized"
    assert p.policy_override == "baseline"
    assert p.dtype_override == "fp32"
    assert p.batch_size == 1
    assert p.enable_distributed is False


def test_optimized_uses_advisor_and_history(tmp_path):
    store = PerformanceStore(tmp_path / "p.json")
    store.record(
        PerformanceSample(
            key=PerformanceStore.key(
                "m", target="local", policy="offload", device="cpu", dtype="fp16", batch_size=2
            ),
            runtime_seconds=1.0,
            success=True,
            tokens_per_second=20.0,
            batch_size=2,
            policy="offload",
            device="cpu",
            dtype="fp16",
            target="local",
        )
    )
    opt = AdaptiveOptimizer(store)
    rec = Recommendation(
        model_name="m",
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
    profile = opt.optimized_profile(
        WorkloadSpec(model_name="m", hourly_rate_usd=0.5),
        hardware=_hw(),
        recommendation=rec,
    )
    assert profile.mode == "optimized"
    # History should override advisor when a successful sample exists.
    assert profile.policy_override == "offload"
    assert profile.dtype_override == "fp16"
    assert profile.batch_size == 2


def test_feedback_loop_records_predicted_vs_actual(tmp_path):
    store = PerformanceStore(tmp_path / "p.json")
    spec = WorkloadSpec(model_name="m", hourly_rate_usd=0.5, batch_size=1)
    plan = WorkloadPlan(
        plan_id="plan_x",
        spec=spec,
        chosen=None,
        candidates=[],
        placement_decision=None,
        local_recommendation=None,
        pool_status=None,
        safety={},
    )
    # Minimal chosen via a tiny CandidateOption-like path: use engine types
    from tron_autopilot.types import CandidateOption

    plan.chosen = CandidateOption(
        option_id="local-baseline",
        target="local",
        policy="baseline",
        device="cpu",
        dtype="fp32",
        memory_budget_bytes=None,
        required_memory_gb=0.1,
        hourly_rate_usd=0.5,
        predicted_runtime_seconds=1.0,
        predicted_cost_usd=0.0001,
        predicted_peak_memory_bytes=None,
        feasible=True,
    )
    outcome = ExecutionOutcome(
        plan_id="plan_x",
        success=True,
        target="local",
        strategy="autopilot",
        actual_runtime_seconds=1.2,
        actual_cost_usd=0.0002,
        actual_peak_memory_bytes=1000,
        predicted_runtime_seconds=1.0,
        predicted_cost_usd=0.0001,
        predicted_peak_memory_bytes=None,
        metrics={"tokens_per_second": 5.0},
    )
    sample = close_feedback_loop(store, plan, outcome)
    assert sample.success
    assert abs(sample.extra["runtime_delta"] - 0.2) < 1e-9
    assert abs(store.mean_runtime(sample.key) - 1.2) < 1e-9


def test_aggregate_metrics_successes_per_hour():
    outs = [
        ExecutionOutcome(
            plan_id="a",
            success=True,
            target="local",
            strategy="fixed",
            actual_runtime_seconds=1.0,
            actual_cost_usd=0.01,
            actual_peak_memory_bytes=100,
            predicted_runtime_seconds=1.0,
            predicted_cost_usd=0.01,
            predicted_peak_memory_bytes=None,
            metrics={},
        ),
        ExecutionOutcome(
            plan_id="b",
            success=True,
            target="local",
            strategy="fixed",
            actual_runtime_seconds=1.0,
            actual_cost_usd=0.01,
            actual_peak_memory_bytes=100,
            predicted_runtime_seconds=1.0,
            predicted_cost_usd=0.01,
            predicted_peak_memory_bytes=None,
            metrics={},
        ),
    ]
    agg = aggregate_run_metrics("unoptimized", outs, wall_seconds=2.0)
    assert agg["successes"] == 2
    # 2 successes in 2 seconds → 3600/hour
    assert abs(agg["successes_per_hour"] - 3600.0) < 1e-6
    assert agg["gpu_utilization_percent"] is None
    assert any("unified VRAM" in n for n in agg["notes"])


def test_adaptive_intelligence_synthetic_run(tmp_path):
    ap = EconomicAutopilot(
        decision_log=DecisionLog(tmp_path / "d.jsonl"),
        history=RuntimeHistory(tmp_path / "h.json"),
        hardware=_hw(),
        recommend_fn=lambda *a, **k: Recommendation(
            model_name="synthetic/adaptive",
            estimated_param_count=1000,
            param_count_is_estimate=True,
            device="cpu",
            dtype="fp32",
            policy="baseline",
            memory_budget_bytes=None,
            confidence="medium",
            rationale=["t"],
            warnings=[],
        ),
    )
    brain = AdaptiveIntelligence(
        autopilot=ap,
        store=PerformanceStore(tmp_path / "p.json"),
        hardware=_hw(),
    )
    spec = WorkloadSpec(
        model_name="synthetic/adaptive",
        hourly_rate_usd=0.5,
        expected_runtime_seconds=0.1,
    )
    profile, plan, outcome, sample = brain.run(
        spec,
        mode="unoptimized",
        efficiency=False,
        func=lambda: {"ok": True},
    )
    assert profile.mode == "unoptimized"
    assert outcome.success
    assert sample.success

    # Optimized second pass should still succeed and record learning sample.
    profile2, plan2, outcome2, sample2 = brain.run(
        spec,
        mode="optimized",
        efficiency=False,
        func=lambda: {"ok": True},
    )
    assert profile2.mode == "optimized"
    assert outcome2.success
    assert len(brain.store.samples_for()) >= 2
