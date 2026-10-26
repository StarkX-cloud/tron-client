"""Orchestrates baseline vs. memory-constrained runs and checks correctness.

Timing and memory numbers alone aren't enough — if the memory-constrained
path produces different output than the baseline under greedy decoding on
identical input, that's a correctness regression, not just a performance
trade-off. ``run_suite`` checks for this and warns loudly rather than
reporting a "win" that's actually wrong output.
"""

from __future__ import annotations

from dataclasses import replace

from tron_efficiency.hardware.detect import HardwareInfo, detect_hardware
from tron_efficiency.memory.policies import (
    DEFAULT_GPU_ACTIVATION_HEADROOM_FRACTION,
    BaselinePolicy,
    MemoryPolicy,
    OffloadPolicy,
)
from tron_efficiency.results.schema import BenchmarkResult
from tron_efficiency.workloads.config import WorkloadConfig
from tron_efficiency.workloads.runner import run_workload


def run_suite(
    config: WorkloadConfig,
    policies: list[MemoryPolicy],
    hardware: HardwareInfo | None = None,
) -> list[BenchmarkResult]:
    """Run ``config`` under each policy and return one result per policy.

    All results share the one ``hardware`` snapshot taken at the start of
    the suite (hardware doesn't change mid-run on this machine).
    """
    hardware = hardware or detect_hardware()

    results: list[BenchmarkResult] = []
    for policy in policies:
        outcome = run_workload(config, policy)
        result = BenchmarkResult.build(
            hardware=hardware,
            config=config,
            policy_description=policy.describe(),
            outcome=outcome,
        )
        results.append(result)

    _check_output_correctness(results)
    return results


def run_budget_sweep(
    config: WorkloadConfig,
    memory_budgets_bytes: list[int],
    include_baseline: bool = True,
    gpu_activation_headroom_fraction: float = DEFAULT_GPU_ACTIVATION_HEADROOM_FRACTION,
    hardware: HardwareInfo | None = None,
) -> list[BenchmarkResult]:
    """Run the same workload once per memory budget in
    ``memory_budgets_bytes`` (plus one unconstrained baseline, unless
    ``include_baseline`` is False), producing a real tradeoff curve instead
    of the two-point comparisons the rest of this package defaults to.

    This is the direct answer to "no sweep across model sizes/budgets yet"
    (flagged repeatedly in README.md/ROADMAP.md) for the budget dimension —
    one real run per budget, each independently measured, not interpolated
    or estimated between points. Sweeping across *model sizes* would mean
    calling this once per model and concatenating the results; not done
    automatically here since each model is a separate multi-GB download
    with its own cost, which a caller should decide about explicitly.

    Budgets are run from largest to smallest (closer to baseline's
    behavior first) — purely for readability when watching it run live;
    results are independent of order.
    """
    policies: list[MemoryPolicy] = []
    if include_baseline:
        policies.append(BaselinePolicy())
    for budget in sorted(memory_budgets_bytes, reverse=True):
        policies.append(
            OffloadPolicy(
                memory_budget_bytes=budget,
                gpu_activation_headroom_fraction=gpu_activation_headroom_fraction,
            )
        )
    return run_suite(config, policies, hardware=hardware)


def run_dtype_sweep(
    config: WorkloadConfig,
    dtypes: list[str | None],
    hardware: HardwareInfo | None = None,
) -> list[BenchmarkResult]:
    """Run ``BaselinePolicy`` once per dtype in ``dtypes`` (e.g.
    ``["fp32", "fp16", "int8", "int4"]``), measuring quantization's actual
    cost/benefit directly — the lever ``OffloadPolicy``'s budget sweep
    doesn't touch. Real quantized loading (``memory/policies.py``'s
    ``_loading_kwargs_for_dtype``) is what makes this a real measurement
    rather than a guess.

    Deliberately does NOT run ``_check_output_correctness`` the way
    ``run_suite``/``run_budget_sweep`` do: quantization is lossy by
    design, so different dtypes producing different generated text is
    expected, not a bug to flag. A factual note is attached instead (see
    ``results/report.py``'s dtype-sweep report), never framed as a
    warning the way real divergence under an unconstrained-vs-offload
    comparison is.
    """
    hardware = hardware or detect_hardware()
    results: list[BenchmarkResult] = []
    for dtype in dtypes:
        dtype_config = replace(config, dtype=dtype)
        outcome = run_workload(dtype_config, BaselinePolicy())
        result = BenchmarkResult.build(
            hardware=hardware,
            config=dtype_config,
            policy_description=BaselinePolicy().describe(),
            outcome=outcome,
        )
        results.append(result)
    return results


def _check_output_correctness(results: list[BenchmarkResult]) -> None:
    successful = [r for r in results if r.outcome.success and r.outcome.generated_text]
    if len(successful) < 2:
        return
    texts = {r.outcome.generated_text for r in successful}
    if len(texts) > 1:
        names = ", ".join(r.policy.get("name", "?") for r in successful)
        warning = (
            f"generated text differs across policies ({names}) for the same "
            "config under greedy decoding — the memory-constrained path may "
            "be producing different output, not just different performance. "
            "Not treated as a benchmark win until this is investigated."
        )
        for r in successful:
            r.warnings.append(warning)
