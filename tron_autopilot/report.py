"""Human-readable reports for Autopilot plans and benchmark comparisons."""

from __future__ import annotations

from tron_autopilot.types import ExecutionOutcome, WorkloadPlan


def format_plan(plan: WorkloadPlan) -> str:
    lines = [
        f"# Autopilot plan `{plan.plan_id}`",
        "",
        f"- model: `{plan.spec.model_name}`",
        f"- budget_usd: {plan.spec.budget_usd}",
        f"- deadline_seconds: {plan.spec.deadline_seconds}",
        f"- reliability: {plan.spec.reliability}",
        f"- hourly_rate_usd (local): {plan.spec.hourly_rate_usd} "
        "(user-configured, not a live cloud price)",
        "",
        "## Chosen",
    ]
    if plan.chosen is None:
        lines.append("- **none** — no feasible candidate")
    else:
        c = plan.chosen
        lines.append(f"- option: `{c.option_id}` → target=`{c.target}` policy=`{c.policy}`")
        lines.append(f"- predicted_runtime_seconds: {c.predicted_runtime_seconds}")
        lines.append(f"- predicted_cost_usd: {c.predicted_cost_usd}")
        lines.append(f"- required_memory_gb: {c.required_memory_gb}")
    lines.append("")
    lines.append("## Candidates")
    for c in plan.candidates:
        flag = "feasible" if c.feasible else "infeasible"
        lines.append(
            f"- `{c.option_id}` [{flag}] target={c.target} "
            f"pred_cost={c.predicted_cost_usd} reasons={c.rejection_reasons or '-'}"
        )
    lines.append("")
    lines.append("## Why")
    for r in plan.rationale:
        lines.append(f"- {r}")
    if plan.warnings:
        lines.append("")
        lines.append("## Warnings")
        for w in plan.warnings:
            lines.append(f"- {w}")
    lines.append("")
    lines.append("## Safety contract")
    for k, v in plan.safety.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


def format_outcome(outcome: ExecutionOutcome) -> str:
    fb = outcome.feedback_delta()
    lines = [
        f"# Execution outcome for `{outcome.plan_id}`",
        "",
        f"- success: {outcome.success}",
        f"- strategy: {outcome.strategy}",
        f"- target: {outcome.target}",
        f"- actual_runtime_seconds: {outcome.actual_runtime_seconds:.4f}",
        f"- actual_cost_usd: {outcome.actual_cost_usd}",
        f"- actual_peak_memory_bytes: {outcome.actual_peak_memory_bytes}",
        f"- predicted_runtime_seconds: {outcome.predicted_runtime_seconds}",
        f"- predicted_cost_usd: {outcome.predicted_cost_usd}",
        f"- runtime_delta: {fb['runtime_seconds_delta']}",
        f"- cost_delta: {fb['cost_usd_delta']}",
    ]
    if outcome.error:
        lines.append(f"- error: {outcome.error}")
    if outcome.recovery:
        lines.append(f"- recovery: {outcome.recovery}")
    return "\n".join(lines) + "\n"


def format_benchmark(rows: list[dict]) -> str:
    """``rows`` are dicts with strategy + aggregate metrics."""
    lines = [
        "# Economic Autopilot benchmark",
        "",
        "| Strategy | Successes | Failures | Cost/success ($) | "
        "Avg completion (s) | Avg peak mem (B) | Avg tok/s | Avg recovery requeues |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['strategy']} | {r['successes']} | {r['failures']} | "
            f"{_fmt(r.get('cost_per_successful_usd'))} | "
            f"{_fmt(r.get('avg_completion_seconds'))} | "
            f"{_fmt(r.get('avg_peak_memory_bytes'), as_int=True)} | "
            f"{_fmt(r.get('avg_tokens_per_second'))} | "
            f"{_fmt(r.get('avg_requeue_count'))} |"
        )
    lines.append("")
    lines.append(
        "Cost uses the configured hourly rate × measured wall time "
        "(estimate only — not a billed cloud invoice)."
    )
    return "\n".join(lines) + "\n"


def _fmt(v, as_int: bool = False) -> str:
    if v is None:
        return "—"
    if as_int:
        return f"{int(v)}"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def summarize_outcomes(strategy: str, outcomes: list[ExecutionOutcome]) -> dict:
    successes = [o for o in outcomes if o.success]
    failures = [o for o in outcomes if not o.success]
    cost_per = None
    if successes:
        known = [o.actual_cost_usd for o in successes if o.actual_cost_usd is not None]
        if known:
            cost_per = sum(known) / len(successes)
    avg_rt = (
        sum(o.actual_runtime_seconds for o in outcomes) / len(outcomes) if outcomes else None
    )
    mem_vals = [o.actual_peak_memory_bytes for o in successes if o.actual_peak_memory_bytes]
    avg_mem = sum(mem_vals) / len(mem_vals) if mem_vals else None
    requeues = []
    for o in outcomes:
        rq = (o.recovery or {}).get("requeue_count_observed")
        if rq is not None:
            requeues.append(float(rq))
    avg_rq = sum(requeues) / len(requeues) if requeues else 0.0
    tok_vals = [
        o.metrics.get("tokens_per_second")
        for o in successes
        if o.metrics and o.metrics.get("tokens_per_second") is not None
    ]
    avg_tok = sum(tok_vals) / len(tok_vals) if tok_vals else None
    gpu_vals = [
        o.metrics.get("peak_gpu_memory_allocated_bytes")
        for o in successes
        if o.metrics and o.metrics.get("peak_gpu_memory_allocated_bytes") is not None
    ]
    avg_gpu = sum(gpu_vals) / len(gpu_vals) if gpu_vals else None
    return {
        "strategy": strategy,
        "successes": len(successes),
        "failures": len(failures),
        "failure_rate": (len(failures) / len(outcomes)) if outcomes else None,
        "cost_per_successful_usd": cost_per,
        "avg_completion_seconds": avg_rt,
        "avg_peak_memory_bytes": avg_mem,
        "avg_tokens_per_second": avg_tok,
        "avg_peak_gpu_memory_bytes": avg_gpu,
        "avg_requeue_count": avg_rq,
        "total_runs": len(outcomes),
    }
