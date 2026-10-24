"""Real, hardware-dependent run of a real small model.

Deliberately NOT part of the default `pytest -q` run — loading and running
even a tiny model takes real seconds and may need a first-time download,
matching the existing project convention of keeping the real LoRA benchmark
(`tron/training/benchmark_lora.py`) out of the default suite. Opt in with:

    TRON_EFFICIENCY_RUN_MODEL_TESTS=1 .venv/Scripts/python.exe -m pytest tests/test_efficiency_workload_real.py -q
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TRON_EFFICIENCY_RUN_MODEL_TESTS") != "1",
    reason="real-model test — set TRON_EFFICIENCY_RUN_MODEL_TESTS=1 to run it "
    "(loads EleutherAI/pythia-70m, may download it on first run)",
)


def test_baseline_run_produces_real_measurements():
    from tron_efficiency.memory.policies import BaselinePolicy
    from tron_efficiency.workloads.config import WorkloadConfig
    from tron_efficiency.workloads.runner import run_workload

    config = WorkloadConfig(max_new_tokens=8, batch_size=1, device="cpu")
    outcome = run_workload(config, BaselinePolicy())

    assert outcome.success, outcome.error
    assert outcome.load_time_seconds > 0
    assert outcome.end_to_end_latency_seconds > 0
    assert outcome.tokens_generated == 8
    assert outcome.tokens_per_second > 0
    assert outcome.generated_text and config.prompt in outcome.generated_text


def test_offload_run_on_cpu_uses_disk_offload_mechanism():
    from tron_efficiency.memory.policies import OffloadPolicy
    from tron_efficiency.workloads.config import WorkloadConfig
    from tron_efficiency.workloads.runner import run_workload

    # pythia-70m's weights total ~134MB; a 100MB CPU budget is below that,
    # so accelerate must genuinely spill part of the model to disk (probed
    # empirically — below ~95MB accelerate refuses outright with "you are
    # trying to offload the whole model to disk", above ~134MB nothing
    # would be offloaded at all, so 100MB is the real partial-offload case).
    config = WorkloadConfig(max_new_tokens=8, batch_size=1, device="cpu")
    outcome = run_workload(config, OffloadPolicy(memory_budget_bytes=100 * 1024 * 1024))

    assert outcome.success, outcome.error
    assert outcome.device_map.get("_mechanism") == "cpu-ram-budget-with-disk-offload"
    assert outcome.tokens_generated == 8


def test_baseline_and_offload_produce_identical_greedy_output():
    """Correctness check: constraining memory must not change the output."""
    from tron_efficiency.benchmarks.engine import run_suite
    from tron_efficiency.memory.policies import BaselinePolicy, OffloadPolicy
    from tron_efficiency.workloads.config import WorkloadConfig

    config = WorkloadConfig(max_new_tokens=8, batch_size=1, device="cpu")
    results = run_suite(
        config, [BaselinePolicy(), OffloadPolicy(memory_budget_bytes=100 * 1024 * 1024)]
    )
    assert all(r.outcome.success for r in results), [r.outcome.error for r in results]
    texts = {r.outcome.generated_text for r in results}
    assert len(texts) == 1, f"outputs diverged across policies: {texts}"
    assert not any(
        "generated text differs" in w for r in results for w in r.warnings
    )
