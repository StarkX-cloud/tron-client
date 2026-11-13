"""Execute a WorkloadPlan using existing local policies or pool handoff.

Local path runs the caller's callable in-process (or optionally a real
``tron_efficiency`` inference workload). Pooled path uses
``tron_router.handoff.submit_to_pool`` — the same ``/submit`` mechanism as
``@tron.remote``. Dedicated plans are refused (flag only).

Failure recovery for pooled jobs is the existing queue_server watchdog;
this module only waits on the ``MagicFuture`` and records how long
recovery took when the job status shows requeues.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from tron_router.decision import JobRequest, PlacementDecision
from tron_router.handoff import HandoffError, submit_to_pool

from tron_autopilot.store import cost_from_runtime, now
from tron_autopilot.types import ExecutionOutcome, WorkloadPlan

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def _peak_rss_bytes() -> int | None:
    if psutil is None:
        return None
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        return None


def execute_plan(
    plan: WorkloadPlan,
    func: Callable[..., Any] | None = None,
    func_args: tuple = (),
    func_kwargs: dict | None = None,
    wait_timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 0.1,
    strategy: str = "autopilot",
    efficiency: bool = False,
) -> ExecutionOutcome:
    """Run ``plan.chosen``.

    Pass ``func`` for an arbitrary callable (local or pooled). Set
    ``efficiency=True`` (and omit ``func``) to run the advisor-chosen
    Baseline/Offload path through ``tron_efficiency.workloads.runner``.
    """
    if efficiency and func is None:
        from tron_autopilot.efficiency_exec import run_efficiency_workload

        return run_efficiency_workload(plan, strategy=strategy)

    if plan.chosen is None:
        return ExecutionOutcome(
            plan_id=plan.plan_id,
            success=False,
            target="rejected",
            strategy=strategy,
            actual_runtime_seconds=0.0,
            actual_cost_usd=None,
            actual_peak_memory_bytes=_peak_rss_bytes(),
            predicted_runtime_seconds=None,
            predicted_cost_usd=None,
            predicted_peak_memory_bytes=None,
            error="no feasible plan — nothing executed",
            decision_rationale=list(plan.rationale),
            warnings=list(plan.warnings),
            timestamp=now(),
        )

    chosen = plan.chosen
    if chosen.target == "dedicated":
        return ExecutionOutcome(
            plan_id=plan.plan_id,
            success=False,
            target="dedicated",
            strategy=strategy,
            actual_runtime_seconds=0.0,
            actual_cost_usd=None,
            actual_peak_memory_bytes=None,
            predicted_runtime_seconds=chosen.predicted_runtime_seconds,
            predicted_cost_usd=chosen.predicted_cost_usd,
            predicted_peak_memory_bytes=chosen.predicted_peak_memory_bytes,
            error=(
                "dedicated is a flag only — Autopilot does not provision "
                "infrastructure or call a GPU marketplace"
            ),
            decision_rationale=list(plan.rationale) + list(chosen.rationale),
            warnings=list(plan.warnings),
            timestamp=now(),
        )

    if func is None:
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
                "execute_plan requires a callable `func`, or efficiency=True "
                "to run the tron_efficiency Baseline/Offload path"
            ),
            decision_rationale=list(plan.rationale),
            warnings=list(plan.warnings),
            timestamp=now(),
        )

    if chosen.target == "local":
        return _execute_local(
            plan, func, func_args, func_kwargs or {}, strategy=strategy
        )

    if chosen.target == "pooled":
        return _execute_pooled(
            plan,
            func,
            func_args,
            func_kwargs or {},
            wait_timeout_seconds=wait_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            strategy=strategy,
        )

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
        error=f"unknown target {chosen.target!r}",
        timestamp=now(),
    )


def _execute_local(plan, func, func_args, func_kwargs, strategy: str) -> ExecutionOutcome:
    chosen = plan.chosen
    mem_before = _peak_rss_bytes()
    start = time.monotonic()
    error = None
    result = None
    success = False
    try:
        result = func(*func_args, **func_kwargs)
        success = True
    except Exception as exc:  # noqa: BLE001 - recorded as failed outcome
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - start
    mem_after = _peak_rss_bytes()
    peak = None
    if mem_before is not None and mem_after is not None:
        peak = max(mem_before, mem_after)
    elif mem_after is not None:
        peak = mem_after

    actual_cost = cost_from_runtime(chosen.hourly_rate_usd, elapsed)
    return ExecutionOutcome(
        plan_id=plan.plan_id,
        success=success,
        target="local",
        strategy=strategy,
        actual_runtime_seconds=elapsed,
        actual_cost_usd=actual_cost,
        actual_peak_memory_bytes=peak,
        predicted_runtime_seconds=chosen.predicted_runtime_seconds,
        predicted_cost_usd=chosen.predicted_cost_usd,
        predicted_peak_memory_bytes=chosen.predicted_peak_memory_bytes,
        result=result,
        error=error,
        recovery={"mechanism": "none_local_inprocess"},
        decision_rationale=list(plan.rationale) + list(chosen.rationale),
        warnings=list(plan.warnings),
        timestamp=now(),
    )


def _execute_pooled(
    plan,
    func,
    func_args,
    func_kwargs,
    wait_timeout_seconds: float,
    poll_interval_seconds: float,
    strategy: str,
) -> ExecutionOutcome:
    chosen = plan.chosen
    spec = plan.spec
    job = JobRequest(
        model_name=spec.model_name,
        tolerant=spec.reliability != "strict",
        requires_multi_node=spec.requires_multi_node,
        gpu_budget_bytes_override=spec.gpu_budget_bytes_override,
    )
    # Build a PlacementDecision the handoff API expects — do not re-decide.
    decision = PlacementDecision(
        target="pooled",
        confidence="medium",
        rationale=list(chosen.rationale),
        warnings=list(plan.warnings),
        local_recommendation=plan.local_recommendation,
        pool_status=plan.pool_status,
        required_memory_gb=chosen.required_memory_gb,
    )

    start = time.monotonic()
    try:
        future = submit_to_pool(
            job,
            decision,
            func,
            func_args=func_args,
            func_kwargs=func_kwargs,
            master_url=spec.master_url,
            required_memory_gb=chosen.required_memory_gb,
        )
    except HandoffError as exc:
        elapsed = time.monotonic() - start
        return ExecutionOutcome(
            plan_id=plan.plan_id,
            success=False,
            target="pooled",
            strategy=strategy,
            actual_runtime_seconds=elapsed,
            actual_cost_usd=cost_from_runtime(chosen.hourly_rate_usd, elapsed),
            actual_peak_memory_bytes=None,
            predicted_runtime_seconds=chosen.predicted_runtime_seconds,
            predicted_cost_usd=chosen.predicted_cost_usd,
            predicted_peak_memory_bytes=chosen.predicted_peak_memory_bytes,
            error=str(exc),
            decision_rationale=list(plan.rationale),
            warnings=list(plan.warnings),
            timestamp=now(),
        )

    # A separate worker process/thread must claim the job (tests and the
    # benchmark drive that). We wait on MagicFuture.get — the same path
    # @tron.remote / handoff tests use — and sample status for requeue evidence.
    result = None
    error = None
    success = False
    requeue_seen = 0
    last_status: dict = {}
    try:
        # Sample once before blocking so a fast completion still leaves a status snapshot.
        last_status = future.status() if hasattr(future, "status") else {}
        if isinstance(last_status, dict):
            requeue_seen = int(last_status.get("requeue_count") or 0)
        result = future.get(timeout=wait_timeout_seconds, poll=poll_interval_seconds)
        success = True
        last_status = future.status() if hasattr(future, "status") else last_status
        if isinstance(last_status, dict):
            requeue_seen = max(requeue_seen, int(last_status.get("requeue_count") or 0))
    except Exception as exc:  # noqa: BLE001 - TimeoutError / RuntimeError from MagicFuture
        error = f"{type(exc).__name__}: {exc}"
        try:
            last_status = future.status() if hasattr(future, "status") else {}
            if isinstance(last_status, dict):
                requeue_seen = max(requeue_seen, int(last_status.get("requeue_count") or 0))
        except Exception:  # noqa: BLE001
            pass

    elapsed = time.monotonic() - start
    recovery = {
        "mechanism": "queue_server_watchdog_requeue",
        "requeue_count_observed": requeue_seen,
        "max_job_requeue_attempts": plan.safety.get("max_job_requeue_attempts"),
        "last_status": {
            k: last_status.get(k)
            for k in ("status", "error", "worker", "requeue_count", "result")
            if isinstance(last_status, dict) and k in last_status
        },
        "note": (
            "Recovery is performed by queue_server's existing watchdog + spine "
            "event log, not by Autopilot inventing a second retry loop."
        ),
    }

    return ExecutionOutcome(
        plan_id=plan.plan_id,
        success=success,
        target="pooled",
        strategy=strategy,
        actual_runtime_seconds=elapsed,
        actual_cost_usd=cost_from_runtime(chosen.hourly_rate_usd, elapsed),
        actual_peak_memory_bytes=None,  # submitter process ≠ worker RSS
        predicted_runtime_seconds=chosen.predicted_runtime_seconds,
        predicted_cost_usd=chosen.predicted_cost_usd,
        predicted_peak_memory_bytes=chosen.predicted_peak_memory_bytes,
        result=result,
        error=error,
        recovery=recovery,
        decision_rationale=list(plan.rationale) + list(chosen.rationale),
        warnings=list(plan.warnings),
        job_id=getattr(future, "job_id", None),
        timestamp=now(),
    )
