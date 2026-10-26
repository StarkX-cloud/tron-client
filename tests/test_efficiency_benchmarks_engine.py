"""Tests for benchmarks/engine.py's run_suite and run_budget_sweep, with a
stubbed run_workload so these are fast and deterministic -- no model load.
"""

from tron_efficiency.benchmarks import engine as engine_module
from tron_efficiency.hardware.detect import HardwareInfo
from tron_efficiency.workloads.config import WorkloadConfig
from tron_efficiency.workloads.runner import RunOutcome


def _fake_hardware():
    return HardwareInfo(
        platform="fake", python_version="3.11", cpu_count=8,
        cpu_total_memory_bytes=16 * 1024**3, cuda_available=False,
        torch_version="2.x", gpus=[], warnings=[],
    )


def _install_fake_run_workload(monkeypatch, text_by_policy=None):
    """text_by_policy: optional {policy_name: generated_text} override, for
    testing the correctness-divergence check."""

    def fake_run_workload(config, policy):
        text = (text_by_policy or {}).get(policy.name, "same output")
        return RunOutcome(
            success=True,
            policy_name=policy.name,
            load_time_seconds=0.1,
            time_to_first_token_seconds=0.01,
            end_to_end_latency_seconds=0.5,
            tokens_generated=config.max_new_tokens,
            tokens_per_second=config.max_new_tokens / 0.5,
            peak_cpu_memory_bytes=1000,
            generated_text=text,
        )

    monkeypatch.setattr(engine_module, "run_workload", fake_run_workload)


def test_run_budget_sweep_includes_baseline_by_default(monkeypatch):
    _install_fake_run_workload(monkeypatch)
    config = WorkloadConfig(model_name="fake/model")
    results = engine_module.run_budget_sweep(
        config, [50 * 1024 * 1024, 100 * 1024 * 1024], hardware=_fake_hardware()
    )
    policy_names = [r.policy.get("name") for r in results]
    assert policy_names.count("baseline") == 1
    assert policy_names.count("offload") == 2


def test_run_budget_sweep_without_baseline(monkeypatch):
    _install_fake_run_workload(monkeypatch)
    config = WorkloadConfig(model_name="fake/model")
    results = engine_module.run_budget_sweep(
        config, [50 * 1024 * 1024, 100 * 1024 * 1024],
        include_baseline=False, hardware=_fake_hardware(),
    )
    assert all(r.policy.get("name") == "offload" for r in results)
    assert len(results) == 2


def test_run_budget_sweep_orders_budgets_largest_first(monkeypatch):
    _install_fake_run_workload(monkeypatch)
    config = WorkloadConfig(model_name="fake/model")
    budgets = [50 * 1024 * 1024, 200 * 1024 * 1024, 100 * 1024 * 1024]
    results = engine_module.run_budget_sweep(
        config, budgets, include_baseline=False, hardware=_fake_hardware()
    )
    budget_order = [r.policy.get("memory_budget_bytes") for r in results]
    assert budget_order == sorted(budgets, reverse=True)


def test_run_budget_sweep_passes_through_headroom_fraction(monkeypatch):
    _install_fake_run_workload(monkeypatch)
    config = WorkloadConfig(model_name="fake/model")
    results = engine_module.run_budget_sweep(
        config, [100 * 1024 * 1024], include_baseline=False,
        gpu_activation_headroom_fraction=0.3, hardware=_fake_hardware(),
    )
    assert results[0].policy.get("gpu_activation_headroom_fraction") == 0.3


def test_run_suite_flags_divergent_output_across_sweep(monkeypatch):
    # The correctness check in run_suite (reused by run_budget_sweep) must
    # still catch divergence across more than two results.
    _install_fake_run_workload(
        monkeypatch, text_by_policy={"baseline": "text A", "offload": "text B"}
    )
    config = WorkloadConfig(model_name="fake/model")
    results = engine_module.run_budget_sweep(
        config, [50 * 1024 * 1024], hardware=_fake_hardware()
    )
    assert any("differs across policies" in w for r in results for w in r.warnings)
