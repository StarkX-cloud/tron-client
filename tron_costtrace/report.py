"""Cost-per-successful-task report, built from raw CallRecords.

Follows the same reporting discipline as ``tron_efficiency.results.report``:
every number traces back to a real recorded call, missing or uncertain data
is stated explicitly rather than silently rounded to zero or omitted, and a
failed call's cost is shown as overhead rather than dropped from the total.

The headline number this module exists to produce is cost-per-successful-
task = (known cost of every call tagged with that task) / (number of calls
that succeeded) — not cost-per-call and not cost-per-token, because those
hide the real cost of retries and failures.
"""

from __future__ import annotations

from dataclasses import dataclass

from tron_costtrace.schema import CallRecord


@dataclass
class TaskSummary:
    task: str
    n_calls: int
    n_success: int
    n_failed: int
    n_unknown_cost: int
    total_known_cost_usd: float
    failed_cost_usd: float
    total_retries: int

    @property
    def cost_per_successful_task_usd(self) -> float | None:
        if self.n_success == 0:
            return None
        return self.total_known_cost_usd / self.n_success


def summarize_by_task(records: list[CallRecord]) -> list[TaskSummary]:
    by_task: dict[str, list[CallRecord]] = {}
    for r in records:
        by_task.setdefault(r.task, []).append(r)

    summaries = []
    for task, recs in by_task.items():
        n_success = sum(1 for r in recs if r.success)
        n_failed = sum(1 for r in recs if r.success is False)
        n_unknown_cost = sum(1 for r in recs if r.cost_usd is None)
        total_known_cost = sum(r.cost_usd for r in recs if r.cost_usd is not None)
        failed_cost = sum(r.cost_usd for r in recs if r.success is False and r.cost_usd is not None)
        total_retries = sum(r.retry_count for r in recs)
        summaries.append(
            TaskSummary(
                task=task,
                n_calls=len(recs),
                n_success=n_success,
                n_failed=n_failed,
                n_unknown_cost=n_unknown_cost,
                total_known_cost_usd=total_known_cost,
                failed_cost_usd=failed_cost,
                total_retries=total_retries,
            )
        )
    return sorted(summaries, key=lambda s: s.task)


def generate_task_report(records: list[CallRecord]) -> str:
    if not records:
        return "# Cost trace report\n\nNo recorded calls.\n"

    summaries = summarize_by_task(records)

    lines = ["# Cost trace report", ""]
    lines.append(f"{len(records)} recorded call(s) across {len(summaries)} task(s).")
    lines.append("")

    lines.append("## Cost per successful task")
    lines.append("")
    lines.append(
        "| Task | Calls | Succeeded | Failed | Cost/successful task | Known total cost | "
        "Failed-call cost (overhead) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for s in summaries:
        per_task = (
            "undefined (no successes)"
            if s.cost_per_successful_task_usd is None
            else f"${s.cost_per_successful_task_usd:.6f}"
        )
        lines.append(
            f"| `{s.task}` | {s.n_calls} | {s.n_success} | {s.n_failed} | {per_task} | "
            f"${s.total_known_cost_usd:.6f} | ${s.failed_cost_usd:.6f} |"
        )
    lines.append("")

    warnings = []
    for s in summaries:
        if s.n_unknown_cost > 0:
            warnings.append(
                f"`{s.task}`: {s.n_unknown_cost}/{s.n_calls} call(s) had unknown cost "
                "(model not in the pinned pricing table, or usage data missing) — "
                "the known-cost total for this task is a floor, not an exact figure."
            )
        if s.total_retries > 0:
            warnings.append(
                f"`{s.task}`: {s.total_retries} retry attempt(s) occurred inside traced "
                "calls. Only the final attempt's token usage is recorded, so the known "
                "cost for this task may undercount the true spend on retried calls."
            )
        if s.n_failed > 0:
            warnings.append(
                f"`{s.task}`: {s.n_failed} call(s) failed and contributed "
                f"${s.failed_cost_usd:.6f} in known cost with zero successful outcome — "
                "this is the hidden overhead a plain cost-per-call or cost-per-token "
                "number would not show."
            )

    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines) + "\n"
