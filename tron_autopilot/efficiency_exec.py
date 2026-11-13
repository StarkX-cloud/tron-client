"""Run a plan's chosen config through ``tron_efficiency`` policies.

This is the Autopilot ↔ efficiency bridge from the next-milestone write-up:
advisor-chosen ``baseline`` / ``offload`` becomes a real ``MemoryPolicy``,
``run_workload`` measures load/TTFT/e2e/tok/s/peak memory, and Autopilot
records those as the execution outcome (no second measurement stack).

Does not invent distributed VRAM. On CPU-only machines the CUDA branch is
simply unused — same honesty contract as ``tron_efficiency`` itself.
"""

from __future__ import annotations

from tron_efficiency.memory.policies import BaselinePolicy, OffloadPolicy
from tron_efficiency.workloads.config import WorkloadConfig
from tron_efficiency.workloads.runner import RunOutcome, run_workload

from tron_autopilot.store import cost_from_runtime, now
from tron_autopilot.types import CandidateOption, ExecutionOutcome, WorkloadPlan


def policy_from_candidate(chosen: CandidateOption):
    """Instantiate the real efficiency policy the plan selected."""
    if chosen.policy == "offload":
        budget = chosen.memory_budget_bytes
        if budget is None or budget <= 0:
            raise ValueError(
                "offload candidate is missing memory_budget_bytes — "
                "advisor must supply a nominal budget for OffloadPolicy"
            )
        return OffloadPolicy(memory_budget_bytes=int(budget))
    # baseline (or unknown → safest baseline)
    return BaselinePolicy()


def workload_config_from_plan(plan: WorkloadPlan) -> WorkloadConfig:
    chosen = plan.chosen
    if chosen is None:
        raise ValueError("plan has no chosen candidate")
    spec = plan.spec
    device = chosen.device or "cpu"
    # Never claim CUDA when the candidate said cpu; never silently upgrade.
    return WorkloadConfig(
        model_name=spec.model_name,
        prompt=spec.prompt,
        max_new_tokens=int(spec.max_new_tokens),
        batch_size=1,
        device=device,
        dtype=chosen.dtype,
    )


def run_efficiency_workload(plan: WorkloadPlan, strategy: str = "autopilot") -> ExecutionOutcome:
    """Execute the plan locally via Baseline/Offload + run_workload."""
    if plan.chosen is None:
        return ExecutionOutcome(
            plan_id=plan.plan_id,
            success=False,
            target="rejected",
            strategy=strategy,
            actual_runtime_seconds=0.0,
            actual_cost_usd=None,
            actual_peak_memory_bytes=None,
            predicted_runtime_seconds=None,
            predicted_cost_usd=None,
            predicted_peak_memory_bytes=None,
            error="no feasible plan — efficiency workload not started",
            decision_rationale=list(plan.rationale),
            warnings=list(plan.warnings),
            timestamp=now(),
        )

    chosen = plan.chosen
    if chosen.target != "local":
        return ExecutionOutcome(
            plan_id=plan.plan_id,
            success=False,
            target=chosen.target,
            strategy=strategy,
            actual_runtime_seconds=0.0,
            actual_cost_usd=None,
            actual_peak_memory_bytes=None,
            predicted_runtime_seconds=chosen.predicted_runtime_seconds,
            predicted_cost_usd=chosen.predicted_cost_usd,
            predicted_peak_memory_bytes=chosen.predicted_peak_memory_bytes,
            error=(
                "run_efficiency_workload is local-only; use execute_plan("
                "efficiency=True) which routes pooled plans through "
                "pooled_efficiency.run_efficiency_job on efficiency_runtime workers"
            ),
            decision_rationale=list(plan.rationale) + list(chosen.rationale),
            warnings=list(plan.warnings),
            timestamp=now(),
        )

    try:
        policy = policy_from_candidate(chosen)
        config = workload_config_from_plan(plan)
    except Exception as exc:  # noqa: BLE001
        return ExecutionOutcome(
            plan_id=plan.plan_id,
            success=False,
            target="local",
            strategy=strategy,
            actual_runtime_seconds=0.0,
            actual_cost_usd=None,
            actual_peak_memory_bytes=None,
            predicted_runtime_seconds=chosen.predicted_runtime_seconds,
            predicted_cost_usd=chosen.predicted_cost_usd,
            predicted_peak_memory_bytes=chosen.predicted_peak_memory_bytes,
            error=f"{type(exc).__name__}: {exc}",
            decision_rationale=list(plan.rationale),
            warnings=list(plan.warnings),
            timestamp=now(),
        )

    import time

    wall_start = time.perf_counter()
    outcome: RunOutcome = run_workload(config, policy)
    wall = time.perf_counter() - wall_start

    # Prefer measured model timings when present; fall back to wall clock.
    measured = None
    if outcome.load_time_seconds is not None and outcome.end_to_end_latency_seconds is not None:
        measured = outcome.load_time_seconds + outcome.end_to_end_latency_seconds
    runtime = measured if measured is not None else wall

    peak = outcome.peak_gpu_memory_allocated_bytes
    if peak is None:
        peak = outcome.peak_cpu_memory_bytes

    actual_cost = (
        cost_from_runtime(chosen.hourly_rate_usd, runtime) if outcome.success else None
    )
    if not outcome.success:
        # Still account wall time spent failing when we have a rate — labeled
        # via success=False so cost/success metrics exclude it.
        actual_cost = cost_from_runtime(chosen.hourly_rate_usd, runtime)

    metrics = {
        "workload": "tron_efficiency.run_workload",
        "policy_name": outcome.policy_name,
        "device": config.device,
        "dtype": config.dtype,
        "load_time_seconds": outcome.load_time_seconds,
        "time_to_first_token_seconds": outcome.time_to_first_token_seconds,
        "end_to_end_latency_seconds": outcome.end_to_end_latency_seconds,
        "tokens_generated": outcome.tokens_generated,
        "tokens_per_second": outcome.tokens_per_second,
        "peak_cpu_memory_bytes": outcome.peak_cpu_memory_bytes,
        "peak_gpu_memory_allocated_bytes": outcome.peak_gpu_memory_allocated_bytes,
        "peak_gpu_memory_reserved_bytes": outcome.peak_gpu_memory_reserved_bytes,
        "wall_clock_seconds": wall,
        "runtime_basis": (
            "load+e2e" if measured is not None else "wall_clock_fallback"
        ),
        "device_map": outcome.device_map,
        "cuda_claimed": False,  # this module never claims CUDA without config.device
    }

    return ExecutionOutcome(
        plan_id=plan.plan_id,
        success=outcome.success,
        target="local",
        strategy=strategy,
        actual_runtime_seconds=runtime,
        actual_cost_usd=actual_cost,
        actual_peak_memory_bytes=peak,
        predicted_runtime_seconds=chosen.predicted_runtime_seconds,
        predicted_cost_usd=chosen.predicted_cost_usd,
        predicted_peak_memory_bytes=chosen.predicted_peak_memory_bytes,
        result={
            "generated_text": outcome.generated_text,
            "metrics": metrics,
        },
        error=outcome.error,
        recovery={"mechanism": "none_local_efficiency"},
        decision_rationale=list(plan.rationale) + list(chosen.rationale),
        warnings=list(plan.warnings) + list(outcome.warnings),
        timestamp=now(),
        metrics=metrics,
    )
