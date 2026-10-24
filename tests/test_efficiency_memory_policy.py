"""Memory policy logic, tested without loading any real model."""

from tron_efficiency.memory.policies import BaselinePolicy, OffloadPolicy, resolve_max_memory


def test_baseline_policy_describe():
    policy = BaselinePolicy()
    assert policy.describe() == {"name": "baseline"}


def test_offload_policy_describe_includes_budget():
    policy = OffloadPolicy(memory_budget_bytes=256 * 1024 * 1024, offload_folder="/tmp/x")
    desc = policy.describe()
    assert desc["name"] == "offload"
    assert desc["memory_budget_bytes"] == 256 * 1024 * 1024
    assert desc["offload_folder"] == "/tmp/x"


def test_resolve_max_memory_on_cpu_constrains_cpu_and_allows_disk():
    max_memory, mechanism = resolve_max_memory("cpu", 128 * 1024 * 1024)
    assert max_memory == {"cpu": 128 * 1024 * 1024, "disk": True}
    assert mechanism == "cpu-ram-budget-with-disk-offload"


def test_resolve_max_memory_on_cuda_constrains_gpu_zero():
    max_memory, mechanism = resolve_max_memory("cuda", 512 * 1024 * 1024)
    assert max_memory[0] == 512 * 1024 * 1024
    assert max_memory["disk"] is True
    assert mechanism == "gpu-memory-budget"
