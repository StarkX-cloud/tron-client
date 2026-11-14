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


def _record_costtrace(plan: WorkloadPlan, outcome: ExecutionOutcome) -> None:
    task = getattr(plan.spec, "costtrace_task", None)
    if not task:
        return
    try:
        import time as _time

        from tron_costtrace.schema import CallRecord
        from tron_costtrace.tracer import default_store

        device = None
        if plan.chosen is not None:
            device = plan.chosen.device
        default_store().append(
            CallRecord(
                task=str(task),
                model=f"tron-compute/{device or outcome.target}",
                input_tokens=None,
                output_tokens=None,
                cost_usd=outcome.actual_cost_usd,
                latency_seconds=float(outcome.actual_runtime_seconds or 0.0),
                retry_count=int(
                    (outcome.recovery or {}).get("requeue_count_observed") or 0
                ),
                success=bool(outcome.success),
                error=outcome.error,
                timestamp=_time.time(),
            )
        )
    except Exception:  # noqa: BLE001 - ledger must never break execution
        pass


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
    outcome = _execute_plan_body(
        plan,
        func=func,
        func_args=func_args,
        func_kwargs=func_kwargs,
        wait_timeout_seconds=wait_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        strategy=strategy,
        efficiency=efficiency,
    )
    _record_costtrace(plan, outcome)
    return outcome


def _execute_plan_body(
    plan: WorkloadPlan,
    func: Callable[..., Any] | None = None,
    func_args: tuple = (),
    func_kwargs: dict | None = None,
    wait_timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 0.1,
    strategy: str = "autopilot",
    efficiency: bool = False,
) -> ExecutionOutcome:
    # Distributed map does not use efficiency policies or a caller func.
    if plan.chosen is not None and plan.chosen.target == "distributed":
        return _execute_distributed(
            plan,
            wait_timeout_seconds=wait_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            strategy=strategy,
        )

    if efficiency and func is None:
        if plan.chosen is not None and plan.chosen.target == "pooled":
            from tron_autopilot.pooled_efficiency import run_efficiency_job

            # Submit the real-model job to a worker that advertised
            # efficiency_runtime. Watchdog recovery stays on queue_server.
            outcome = _execute_plan_body(
                plan,
                func=run_efficiency_job,
                func_kwargs={
                    "model_name": plan.spec.model_name,
                    "policy": plan.chosen.policy or "baseline",
                    "device": plan.chosen.device or "cpu",
                    "dtype": plan.chosen.dtype,
                    "memory_budget_bytes": plan.chosen.memory_budget_bytes,
                    "max_new_tokens": plan.spec.max_new_tokens,
                    "prompt": plan.spec.prompt,
                    "hourly_rate_usd": plan.chosen.hourly_rate_usd,
                },
                wait_timeout_seconds=wait_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                strategy=strategy,
                efficiency=False,
            )
            # Normalize worker dict result into metrics when present.
            if isinstance(outcome.result, dict) and outcome.result.get("metrics"):
                outcome.metrics = dict(outcome.result.get("metrics") or {})
                if outcome.result.get("actual_runtime_seconds") is not None:
                    outcome.actual_runtime_seconds = float(
                        outcome.result["actual_runtime_seconds"]
                    )
                if outcome.result.get("actual_peak_memory_bytes") is not None:
                    outcome.actual_peak_memory_bytes = outcome.result[
                        "actual_peak_memory_bytes"
                    ]
                if outcome.result.get("actual_cost_usd") is not None:
                    outcome.actual_cost_usd = outcome.result["actual_cost_usd"]
                if outcome.result.get("success") is False:
                    outcome.success = False
                    outcome.error = outcome.result.get("error") or outcome.error
            return outcome

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
        return _execute_dedicated(
            plan,
            func=func,
            func_args=func_args,
            func_kwargs=func_kwargs or {},
            wait_timeout_seconds=wait_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            strategy=strategy,
            efficiency=efficiency,
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
                "to run the tron_efficiency Baseline/Offload path "
                "(distributed map needs partition_items, not func)"
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


def _execute_dedicated(
    plan,
    func,
    func_args,
    func_kwargs,
    wait_timeout_seconds: float,
    poll_interval_seconds: float,
    strategy: str,
    efficiency: bool,
) -> ExecutionOutcome:
    """Opt-in provisioner path. Without TRON_DEDICATED_PROVISIONER, refuse honestly."""
    chosen = plan.chosen
    try:
        from tron_autopilot.provisioners import get_provisioner

        provisioner = get_provisioner()
    except Exception as exc:  # noqa: BLE001
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
            error=f"dedicated provisioner error: {exc}",
            decision_rationale=list(plan.rationale) + list(chosen.rationale),
            warnings=list(plan.warnings),
            timestamp=now(),
        )

    if provisioner is None:
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
                "dedicated requires TRON_DEDICATED_PROVISIONER "
                "(e.g. linode) — unset means refuse, not fake a marketplace"
            ),
            decision_rationale=list(plan.rationale) + list(chosen.rationale),
            warnings=list(plan.warnings),
            timestamp=now(),
        )

    master_url = (plan.spec.master_url or "http://localhost:9000").rstrip("/")
    worker = None
    try:
        worker = provisioner.provision(master_url=master_url)
        # Re-route as pooled onto the freshly attached worker.
        from dataclasses import replace

        from tron_autopilot.types import CandidateOption

        pooled = CandidateOption(
            option_id="dedicated-provisioned-pool",
            target="pooled",
            policy=chosen.policy,
            device=chosen.device,
            dtype=chosen.dtype,
            memory_budget_bytes=chosen.memory_budget_bytes,
            required_memory_gb=chosen.required_memory_gb,
            hourly_rate_usd=getattr(worker, "hourly_rate_usd", chosen.hourly_rate_usd),
            predicted_runtime_seconds=chosen.predicted_runtime_seconds,
            predicted_cost_usd=chosen.predicted_cost_usd,
            predicted_peak_memory_bytes=chosen.predicted_peak_memory_bytes,
            feasible=True,
            rejection_reasons=[],
            rationale=list(chosen.rationale)
            + [
                f"provisioned {getattr(worker, 'sku', 'node')} "
                f"as worker {getattr(worker, 'worker_name', '?')}"
            ],
            source_signals={
                "provisioner": "linode",
                "worker_name": getattr(worker, "worker_name", None),
                "sku": getattr(worker, "sku", None),
            },
        )
        pooled_plan = replace(plan, chosen=pooled)
        if efficiency and func is None:
            from tron_autopilot.pooled_efficiency import run_efficiency_job

            return _execute_pooled(
                pooled_plan,
                run_efficiency_job,
                (),
                {
                    "model_name": plan.spec.model_name,
                    "policy": pooled.policy or "baseline",
                    "device": pooled.device or "cpu",
                    "dtype": pooled.dtype,
                    "memory_budget_bytes": pooled.memory_budget_bytes,
                    "max_new_tokens": plan.spec.max_new_tokens,
                    "prompt": plan.spec.prompt,
                    "hourly_rate_usd": pooled.hourly_rate_usd,
                },
                wait_timeout_seconds=wait_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                strategy=strategy,
            )
        if func is None:
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
                error="dedicated provisioned but no func/efficiency payload to run",
                timestamp=now(),
            )
        return _execute_pooled(
            pooled_plan,
            func,
            func_args,
            func_kwargs,
            wait_timeout_seconds=wait_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            strategy=strategy,
        )
    except Exception as exc:  # noqa: BLE001
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
            error=f"{type(exc).__name__}: {exc}",
            decision_rationale=list(plan.rationale) + list(chosen.rationale),
            warnings=list(plan.warnings),
            timestamp=now(),
        )
    finally:
        if worker is not None:
            try:
                worker.destroy()
            except Exception:  # noqa: BLE001
                pass


def _execute_distributed(
    plan: WorkloadPlan,
    wait_timeout_seconds: float,
    poll_interval_seconds: float,
    strategy: str,
) -> ExecutionOutcome:
    """Independent-task map across pool workers via DistributedCoordinator."""
    from tron_autopilot.distributed.coordinator import DistributedCoordinator
    from tron_autopilot.distributed.ops import closed_form_sum_squares
    from tron_autopilot.distributed.partition import partition_items

    chosen = plan.chosen
    spec = plan.spec
    items = list(spec.partition_items or [])
    n_shards = int((chosen.source_signals or {}).get("n_shards") or 0)
    if n_shards < 1:
        n_shards = max(2, min(len(items), 4))
    shards = partition_items(
        items,
        n_shards,
        workload_id=f"{spec.model_name}:{spec.map_op}",
        min_shard_size=spec.min_shard_size,
    )
    master = (spec.master_url or "http://localhost:9000").rstrip("/")
    coord = DistributedCoordinator(
        master,
        wait_timeout_seconds=wait_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        required_memory_gb=chosen.required_memory_gb or 0.25,
    )
    run = coord.run(shards, op=spec.map_op or "sum_squares")
    actual_cost = cost_from_runtime(chosen.hourly_rate_usd, run.wall_seconds)

    # Correctness check for the reference sum_squares workload.
    # Approximate actual communication overhead as wall − max shard compute
    # when shard timings are unavailable; prefer predicted when no better signal.
    pred_comm = (chosen.source_signals or {}).get("parallel_estimate", {}).get(
        "predicted_comm_seconds"
    )
    actual_comm = pred_comm
    shard_walls = [
        float((s.result or {}).get("n") or 0)  # placeholder; n is count not time
        for s in run.shard_results
    ]
    _ = shard_walls  # kept for future per-shard timing fields
    # Honest: without per-shard timers, report predicted comm and label it.
    metrics = {
        "distributed": True,
        "n_shards": len(shards),
        "attempts_total": run.attempts_total,
        "duplicates_suppressed": run.duplicates_suppressed,
        "comm_predicted_seconds": pred_comm,
        "actual_comm_seconds": actual_comm,
        "comm_seconds_basis": "predicted_cost_model"
        if pred_comm is not None
        else None,
    }
    error = run.error
    success = run.success
    if success and (spec.map_op or "sum_squares") == "sum_squares":
        expected = closed_form_sum_squares(len(items))
        got = int(run.reduced["total"]) if run.reduced else None
        metrics["expected_total"] = expected
        metrics["got_total"] = got
        if got != expected:
            success = False
            error = f"correctness failure: got {got} expected {expected}"

    return ExecutionOutcome(
        plan_id=plan.plan_id,
        success=success,
        target="distributed",
        strategy=strategy,
        actual_runtime_seconds=run.wall_seconds,
        actual_cost_usd=actual_cost,
        actual_peak_memory_bytes=None,
        predicted_runtime_seconds=chosen.predicted_runtime_seconds,
        predicted_cost_usd=chosen.predicted_cost_usd,
        predicted_peak_memory_bytes=chosen.predicted_peak_memory_bytes,
        result=run.to_dict(),
        error=error,
        recovery={
            "mechanism": "per_shard_retry_plus_queue_watchdog",
            "shard_requeues": [
                s.requeue_count for s in run.shard_results
            ],
            "duplicates_suppressed": run.duplicates_suppressed,
        },
        decision_rationale=list(plan.rationale) + list(chosen.rationale),
        warnings=list(plan.warnings),
        timestamp=now(),
        metrics=metrics,
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
            requires_efficiency_runtime=bool(spec.requires_efficiency_runtime),
            preferred_location=getattr(spec, "preferred_region", None),
            preferred_location_strict=bool(
                getattr(spec, "preferred_location_strict", False)
            ),
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
