"""Orchestrates baseline vs. memory-constrained runs and checks correctness.

Timing and memory numbers alone aren't enough — if the memory-constrained
path produces different output than the baseline under greedy decoding on
identical input, that's a correctness regression, not just a performance
trade-off. ``run_suite`` checks for this and warns loudly rather than
reporting a "win" that's actually wrong output.
"""

from __future__ import annotations

from tron_efficiency.hardware.detect import HardwareInfo, detect_hardware
from tron_efficiency.memory.policies import MemoryPolicy
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
