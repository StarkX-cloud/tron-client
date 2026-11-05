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


def _run(config: WorkloadConfig, policy, isolated: bool):
    """Dispatch to the in-process or subprocess-isolated runner.

    ``run_workload`` is referenced as the module-level name (not imported
    fresh here) so existing tests that monkeypatch
    ``tron_efficiency.benchmarks.engine.run_workload`` keep working
    unchanged when ``isolated=False`` (the default for library/test use —
    see ``run_workload_isolated``'s docstring and ROADMAP.md's "real
    segfault found during matrix-sweep development" entry for why
    isolation exists and why it isn't the default everywhere: it adds a
    full process-startup + torch/transformers-import cost per run, which
    would make every fast unit test in this project slow and would break
    the monkeypatch-based stubbing those tests rely on).
    """
    if isolated:
        from tron_efficiency.workloads.isolated_runner import run_workload_isolated

        return run_workload_isolated(config, policy)
    return run_workload(config, policy)


def run_suite(
    config: WorkloadConfig,
    policies: list[MemoryPolicy],
    hardware: HardwareInfo | None = None,
    isolated: bool = False,
) -> list[BenchmarkResult]:
    """Run ``config`` under each policy and return one result per policy.

    All results share the one ``hardware`` snapshot taken at the start of
    the suite (hardware doesn't change mid-run on this machine).

    ``isolated=True`` runs each policy in its own subprocess (see
    ``workloads/isolated_runner.py``) so a native-level crash in one run
    can't take the others down with it or kill the caller. Off by default
    here (cheap, fast, two-policy comparisons are lower-risk and this
    keeps the function fast/mockable for tests); the CLI turns it on by
    default for the sweep commands specifically, where a crash partway
    through would otherwise lose every result collected so far.
    """
    hardware = hardware or detect_hardware()

    results: list[BenchmarkResult] = []
    for policy in policies:
        outcome = _run(config, policy, isolated)
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
    isolated: bool = False,
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
    return run_suite(config, policies, hardware=hardware, isolated=isolated)


def run_dtype_sweep(
    config: WorkloadConfig,
    dtypes: list[str | None],
    hardware: HardwareInfo | None = None,
    isolated: bool = False,
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
        outcome = _run(dtype_config, BaselinePolicy(), isolated)
        result = BenchmarkResult.build(
            hardware=hardware,
            config=dtype_config,
            policy_description=BaselinePolicy().describe(),
            outcome=outcome,
        )
        results.append(result)
    return results


def run_matrix_sweep(
    config: WorkloadConfig,
    dtypes: list[str | None],
    memory_budgets_bytes: list[int],
    gpu_activation_headroom_fraction: float = DEFAULT_GPU_ACTIVATION_HEADROOM_FRACTION,
    hardware: HardwareInfo | None = None,
    isolated: bool = False,
) -> list[BenchmarkResult]:
    """Cross ``run_budget_sweep``'s axis (memory budget) with
    ``run_dtype_sweep``'s axis (dtype): run ``OffloadPolicy`` once per
    (dtype, budget) pair, the full landscape in one pass instead of two
    separate one-dimensional sweeps a caller would otherwise have to
    reconcile by hand.

    Deliberately does NOT skip combinations known to fail (e.g.
    ``dtype="int4"`` -- unsupported with offload, see
    ``memory/policies.py``'s ``UnsupportedDtypeCombination``): a real
    failure is itself a real data point for the matrix, reported as such
    (``outcome.success=False`` with the actual error), not silently
    omitted. This is also the tool for properly characterizing
    hardware-specific quirks found ad hoc during development (e.g. the
    int8+offload CPU behavior documented in ROADMAP.md) instead of probing
    one budget at a time by hand.

    Strongly consider ``isolated=True`` here specifically: this is exactly
    the function that found a real, reproducible segfault during
    development (``OffloadPolicy``, ``dtype=None``, a 130MB budget on
    this CPU machine — see
    writeup/compute-efficiency-matrix-sweep-crash.md). A crash in one
    (dtype, budget) cell will otherwise kill the whole matrix and lose
    every cell collected so far, not just report that one cell failed.

    Does not include an unconstrained baseline (that's a different axis
    again -- combine with ``run_dtype_sweep(config, dtypes)`` separately if
    a no-offload reference point is needed) and does not run the
    correctness check the way ``run_budget_sweep`` does for a single dtype
    (divergent text across *dtypes* is expected, same reasoning as
    ``run_dtype_sweep``).
    """
    hardware = hardware or detect_hardware()
    results: list[BenchmarkResult] = []
    for dtype in dtypes:
        dtype_config = replace(config, dtype=dtype)
        for budget in sorted(memory_budgets_bytes, reverse=True):
            policy = OffloadPolicy(
                memory_budget_bytes=budget,
                gpu_activation_headroom_fraction=gpu_activation_headroom_fraction,
            )
            outcome = _run(dtype_config, policy, isolated)
            result = BenchmarkResult.build(
                hardware=hardware,
                config=dtype_config,
                policy_description=policy.describe(),
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
