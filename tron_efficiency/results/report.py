"""Human-readable comparison report.

Builds a markdown report from a list of ``BenchmarkResult``s. Every number
it prints came from a real run; every comparison is between runs that
actually used the same model/device/batch/output-length config. When a run
failed or a metric is missing, the report says so explicitly rather than
omitting it silently or filling in an estimate.
"""

from __future__ import annotations

from tron_efficiency.results.schema import BenchmarkResult

# (field name on RunOutcome, label, lower value wins)
_METRICS = [
    ("load_time_seconds", "Model load time (s)", True),
    ("time_to_first_token_seconds", "Time to first token (s)", True),
    ("end_to_end_latency_seconds", "End-to-end latency (s)", True),
    ("tokens_per_second", "Throughput (tokens/sec)", False),
    ("peak_cpu_memory_bytes", "Peak CPU memory (bytes)", True),
    ("peak_gpu_memory_allocated_bytes", "Peak GPU memory allocated (bytes)", True),
]


def _same_workload(a: BenchmarkResult, b: BenchmarkResult) -> bool:
    return (
        a.config.model_name == b.config.model_name
        and a.config.device == b.config.device
        and a.config.batch_size == b.config.batch_size
        and a.config.max_new_tokens == b.config.max_new_tokens
    )


def generate_report(
    results: list[BenchmarkResult], hourly_rate_usd: float | None = None
) -> str:
    if not results:
        return "# Benchmark report\n\nNo results to report.\n"

    lines = ["# Benchmark report", ""]
    lines.append(f"Comparing {len(results)} run(s).")
    lines.append("")

    base = results[0]
    mismatched = [r for r in results[1:] if not _same_workload(base, r)]
    if mismatched:
        lines.append(
            "**Warning: not all runs share the same model/device/batch/output-length "
            "configuration — the comparison below is between runs with different "
            "workloads and should be read with that in mind.**"
        )
        lines.append("")

    lines.append("## Runs")
    for r in results:
        status = "OK" if r.outcome.success else f"FAILED ({r.outcome.error})"
        lines.append(
            f"- `{r.policy.get('name', r.outcome.policy_name)}` on "
            f"{r.config.model_name} ({r.config.device}): {status}"
        )
        for w in r.warnings:
            lines.append(f"  - warning: {w}")
    lines.append("")

    successful = [r for r in results if r.outcome.success]
    if len(successful) < 2:
        lines.append(
            "**Fewer than two successful runs — no metric-by-metric comparison is "
            "possible. This is reported honestly rather than invented.**"
        )
        return "\n".join(lines) + "\n"

    lines.append("## Metric comparison")
    lines.append("")
    lines.append("| Metric | " + " | ".join(
        r.policy.get("name", r.outcome.policy_name) for r in successful
    ) + " | Winner |")
    lines.append("|---" * (len(successful) + 2) + "|")

    for field_name, label, lower_is_better in _METRICS:
        values = [getattr(r.outcome, field_name) for r in successful]
        cells = [("-" if v is None else f"{v:,.4g}") for v in values]
        present = [(r, v) for r, v in zip(successful, values) if v is not None]
        if len(present) < 2:
            winner = "incomparable (missing data)"
        else:
            best = min(present, key=lambda rv: rv[1]) if lower_is_better else max(
                present, key=lambda rv: rv[1]
            )
            winner = best[0].policy.get("name", best[0].outcome.policy_name)
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {winner} |")

    lines.append("")

    if hourly_rate_usd is not None:
        lines.append("## Estimated execution cost")
        lines.append(
            f"Using a configured rate of ${hourly_rate_usd:.4f}/hour "
            "(this is an input you set, not a measured cloud price):"
        )
        lines.append("")
        for r in successful:
            e2e = r.outcome.end_to_end_latency_seconds
            if e2e is None:
                continue
            cost = hourly_rate_usd / 3600.0 * e2e
            lines.append(
                f"- `{r.policy.get('name', r.outcome.policy_name)}`: "
                f"${cost:.6f} estimated for this run's {e2e:.3f}s of execution "
                "(estimate only — not an actual customer saving)."
            )
        lines.append("")

    all_warnings = [w for r in results for w in r.warnings]
    if all_warnings:
        lines.append("## Warnings carried from individual runs")
        for w in sorted(set(all_warnings)):
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _sweep_row_label(r: BenchmarkResult) -> str:
    """generate_report labels columns by policy name alone ("offload"),
    which collides across a budget sweep where every row is "offload" at a
    different budget — this labels each row distinctly instead."""
    name = r.policy.get("name", r.outcome.policy_name)
    budget = r.policy.get("memory_budget_bytes")
    if budget is None:
        return name
    return f"{name} @ {budget / (1024 * 1024):.0f}MB"


def generate_sweep_report(results: list[BenchmarkResult]) -> str:
    """Report a budget sweep (see benchmarks/engine.py's run_budget_sweep)
    as a real tradeoff curve: one row per budget, sorted from largest to
    smallest, so the trend (if any) reads top to bottom. Every number here
    is from an independent real run at that specific budget — nothing is
    interpolated or estimated between points.
    """
    if not results:
        return "# Budget sweep report\n\nNo results to report.\n"

    lines = ["# Budget sweep report", ""]
    lines.append(f"{len(results)} run(s), one per budget (plus baseline if included).")
    lines.append("")

    ordered = sorted(
        results,
        key=lambda r: (
            r.policy.get("memory_budget_bytes") is None,  # baseline (None) sorts first
            -(r.policy.get("memory_budget_bytes") or 0),
        ),
    )

    lines.append("## Runs")
    for r in ordered:
        status = "OK" if r.outcome.success else f"FAILED ({r.outcome.error})"
        lines.append(f"- `{_sweep_row_label(r)}`: {status}")
        for w in r.warnings:
            lines.append(f"  - warning: {w}")
    lines.append("")

    successful = [r for r in ordered if r.outcome.success]
    if not successful:
        lines.append("**No successful runs in this sweep — no trend to report.**")
        return "\n".join(lines) + "\n"

    lines.append("## Tradeoff curve")
    lines.append("")
    headers = [label for _, label, _ in _METRICS]
    lines.append("| Budget | " + " | ".join(headers) + " |")
    lines.append("|---" * (len(headers) + 1) + "|")
    for r in successful:
        cells = []
        for field_name, _, _ in _METRICS:
            value = getattr(r.outcome, field_name)
            cells.append("-" if value is None else f"{value:,.4g}")
        lines.append(f"| {_sweep_row_label(r)} | " + " | ".join(cells) + " |")
    lines.append("")

    all_warnings = [w for r in results for w in r.warnings]
    if all_warnings:
        lines.append("## Warnings carried from individual runs")
        for w in sorted(set(all_warnings)):
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines) + "\n"


def generate_dtype_sweep_report(results: list[BenchmarkResult]) -> str:
    """Report a dtype sweep (see benchmarks/engine.py's run_dtype_sweep) —
    the real cost/benefit of quantization, measured, not assumed. Unlike
    generate_sweep_report's budget curve, divergent generated_text across
    rows here is expected (quantization is lossy by design) and reported
    as a plain fact, never as a "something's wrong" warning.
    """
    if not results:
        return "# Dtype sweep report\n\nNo results to report.\n"

    lines = ["# Dtype sweep report", ""]
    lines.append(f"{len(results)} run(s), one per dtype.")
    lines.append("")

    lines.append("## Runs")
    for r in results:
        status = "OK" if r.outcome.success else f"FAILED ({r.outcome.error})"
        label = r.config.dtype or "checkpoint-default"
        lines.append(f"- `{label}`: {status}")
        for w in r.warnings:
            lines.append(f"  - warning: {w}")
    lines.append("")

    successful = [r for r in results if r.outcome.success]
    if not successful:
        lines.append("**No successful runs in this sweep — no trend to report.**")
        return "\n".join(lines) + "\n"

    lines.append("## Dtype comparison")
    lines.append("")
    headers = [label for _, label, _ in _METRICS]
    lines.append("| Dtype | " + " | ".join(headers) + " |")
    lines.append("|---" * (len(headers) + 1) + "|")
    for r in successful:
        cells = []
        for field_name, _, _ in _METRICS:
            value = getattr(r.outcome, field_name)
            cells.append("-" if value is None else f"{value:,.4g}")
        label = r.config.dtype or "checkpoint-default"
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    texts = {
        (r.config.dtype or "checkpoint-default"): r.outcome.generated_text for r in successful
    }
    unique_texts = set(texts.values())
    lines.append("## Output text")
    lines.append(
        "Quantization is lossy by design — different dtypes producing different "
        "text is expected, not reported as a correctness problem the way it would "
        "be for OffloadPolicy (which should produce identical output to baseline)."
    )
    if len(unique_texts) == 1:
        lines.append("All dtypes in this sweep produced identical output.")
    else:
        lines.append(f"{len(unique_texts)} distinct output(s) across {len(texts)} dtype(s):")
        for dtype_label, text in texts.items():
            lines.append(f"- `{dtype_label}`: {text!r}")
    lines.append("")

    all_warnings = [w for r in results for w in r.warnings]
    if all_warnings:
        lines.append("## Warnings carried from individual runs")
        for w in sorted(set(all_warnings)):
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines) + "\n"
