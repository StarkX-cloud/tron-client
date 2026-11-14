"""Predicted → actual → updated optimization decisions."""

from __future__ import annotations

from typing import Any, Callable

from tron_efficiency.advisor.recommend import recommend
from tron_efficiency.hardware.detect import HardwareInfo, detect_hardware

from tron_adaptive.metrics import sample_gpu_utilization
from tron_adaptive.optimizer import AdaptiveOptimizer, OptimizationProfile
from tron_adaptive.performance_store import PerformanceSample, PerformanceStore
from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.execute import execute_plan
from tron_autopilot.types import ExecutionOutcome, WorkloadPlan, WorkloadSpec


def close_feedback_loop(
    store: PerformanceStore,
    plan: WorkloadPlan,
    outcome: ExecutionOutcome,
) -> PerformanceSample:
    """Record measured outcome into the adaptive performance store."""
    chosen = plan.chosen
    batch = int(getattr(plan.spec, "batch_size", 1) or 1)
    key = PerformanceStore.key(
        plan.spec.model_name,
        target=outcome.target,
        policy=chosen.policy if chosen else None,
        device=chosen.device if chosen else None,
        dtype=chosen.dtype if chosen else None,
        batch_size=batch,
    )
    metrics = outcome.metrics or {}
    recovered = False
    rec = outcome.recovery or {}
    if rec.get("requeue_count_observed"):
        recovered = float(rec["requeue_count_observed"]) > 0 and outcome.success
    if rec.get("shard_requeues"):
        recovered = any(int(x) > 0 for x in rec["shard_requeues"]) and outcome.success

    sample = PerformanceSample(
        key=key,
        runtime_seconds=float(outcome.actual_runtime_seconds or 0.0),
        success=bool(outcome.success),
        cost_usd=outcome.actual_cost_usd,
        tokens_per_second=metrics.get("tokens_per_second"),
        peak_memory_bytes=outcome.actual_peak_memory_bytes,
        peak_gpu_memory_bytes=metrics.get("peak_gpu_memory_allocated_bytes"),
        gpu_utilization_percent=metrics.get("gpu_utilization_percent"),
        batch_size=batch,
        policy=chosen.policy if chosen else None,
        device=chosen.device if chosen else None,
        dtype=chosen.dtype if chosen else None,
        target=outcome.target,
        recovered=recovered,
        extra={
            "plan_id": plan.plan_id,
            "predicted_runtime_seconds": outcome.predicted_runtime_seconds,
            "predicted_cost_usd": outcome.predicted_cost_usd,
            "runtime_delta": (outcome.feedback_delta() or {}).get(
                "runtime_seconds_delta"
            ),
            "adaptive_mode": getattr(plan.spec, "adaptive_mode", None),
        },
    )
    store.record(sample)
    return sample


class AdaptiveIntelligence:
    """Plan + execute with optimization profile + feedback learning."""

    def __init__(
        self,
        autopilot: EconomicAutopilot | None = None,
        store: PerformanceStore | None = None,
        hardware: HardwareInfo | None = None,
        recommend_fn: Callable | None = None,
    ):
        self.autopilot = autopilot or EconomicAutopilot(hardware=hardware)
        self.store = store or PerformanceStore()
        self.optimizer = AdaptiveOptimizer(self.store)
        self.hardware = hardware
        self._recommend = recommend_fn or recommend

    def profile_for(
        self, spec: WorkloadSpec, *, mode: str = "optimized"
    ) -> OptimizationProfile:
        hardware = self.hardware or detect_hardware()
        if mode == "unoptimized":
            profile = self.optimizer.unoptimized_profile(spec)
            # If CUDA is real, unoptimized still may run on GPU with baseline/fp32
            # but keep batch=1 — only upgrade device when CUDA present.
            if hardware.cuda_available:
                profile.device_override = "cuda"
                profile.rationale.append(
                    "unoptimized on CUDA host: still baseline/fp32/batch=1 on GPU"
                )
            return profile
        rec = None
        try:
            rec = self._recommend(
                spec.model_name,
                hardware=hardware,
                gpu_budget_bytes_override=spec.gpu_budget_bytes_override,
            )
        except Exception:  # noqa: BLE001
            rec = None
        return self.optimizer.optimized_profile(
            spec, hardware=hardware, recommendation=rec
        )

    def run(
        self,
        spec: WorkloadSpec,
        *,
        mode: str = "optimized",
        strategy: str = "autopilot",
        efficiency: bool = True,
        func: Callable[..., Any] | None = None,
        func_args: tuple = (),
        func_kwargs: dict | None = None,
        pool_status=None,
        wait_timeout_seconds: float = 300.0,
    ) -> tuple[OptimizationProfile, WorkloadPlan, ExecutionOutcome, PerformanceSample]:
        profile = self.profile_for(spec, mode=mode)
        applied = self.optimizer.apply_to_spec(spec, profile)
        # fixed strategy for unoptimized so advisor/pool cannot "help"
        use_strategy = "fixed" if mode == "unoptimized" else strategy
        plan = self.autopilot.plan(
            applied, pool_status=pool_status, strategy=use_strategy
        )
        # Stamp adaptive rationale onto plan warnings for auditability.
        plan.warnings = list(plan.warnings) + [
            f"adaptive_mode={profile.mode}",
            *[f"adaptive: {r}" for r in profile.rationale[:6]],
        ]
        outcome = execute_plan(
            plan,
            func=func,
            func_args=func_args,
            func_kwargs=func_kwargs or {},
            wait_timeout_seconds=wait_timeout_seconds,
            strategy=use_strategy,
            efficiency=efficiency and func is None,
        )
        # Attach a late GPU util sample if missing and CUDA was used.
        if (
            outcome.metrics is not None
            and outcome.metrics.get("gpu_utilization_percent") is None
            and (plan.chosen and plan.chosen.device == "cuda")
        ):
            outcome.metrics["gpu_utilization_percent"] = sample_gpu_utilization()

        self.autopilot.record_feedback(plan, outcome)
        sample = close_feedback_loop(self.store, plan, outcome)
        return profile, plan, outcome, sample
