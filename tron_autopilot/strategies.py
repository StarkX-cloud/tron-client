"""Named execution strategies for the Economic Autopilot benchmark.

All three share the same ``EconomicAutopilot.plan`` + ``execute_plan`` path.
Only the ``strategy`` argument changes which candidate is selected — so the
comparison measures decision policy, not three different executors.
"""

from __future__ import annotations

from typing import Any, Callable

from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.execute import execute_plan
from tron_autopilot.types import ExecutionOutcome, WorkloadPlan, WorkloadSpec

STRATEGIES = ("fixed", "cheapest", "autopilot")


def run_strategy(
    autopilot: EconomicAutopilot,
    spec: WorkloadSpec,
    strategy: str,
    func: Callable[..., Any],
    func_args: tuple = (),
    func_kwargs: dict | None = None,
    pool_status=None,
    wait_timeout_seconds: float = 120.0,
) -> tuple[WorkloadPlan, ExecutionOutcome]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    plan = autopilot.plan(spec, pool_status=pool_status, strategy=strategy)
    outcome = execute_plan(
        plan,
        func=func,
        func_args=func_args,
        func_kwargs=func_kwargs,
        wait_timeout_seconds=wait_timeout_seconds,
        strategy=strategy,
    )
    autopilot.record_feedback(plan, outcome)
    return plan, outcome
