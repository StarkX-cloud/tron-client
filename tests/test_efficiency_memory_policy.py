"""Memory policy logic, tested without loading any real model."""

from tron_efficiency.memory.policies import (
    DEFAULT_GPU_ACTIVATION_HEADROOM_FRACTION,
    BaselinePolicy,
    OffloadPolicy,
    resolve_max_memory,
)


def test_baseline_policy_describe():
    policy = BaselinePolicy()
    assert policy.describe() == {"name": "baseline"}


def test_offload_policy_describe_includes_budget():
    policy = OffloadPolicy(memory_budget_bytes=256 * 1024 * 1024, offload_folder="/tmp/x")
    desc = policy.describe()
    assert desc["name"] == "offload"
    assert desc["memory_budget_bytes"] == 256 * 1024 * 1024
    assert desc["offload_folder"] == "/tmp/x"
    assert desc["gpu_activation_headroom_fraction"] == DEFAULT_GPU_ACTIVATION_HEADROOM_FRACTION


def test_resolve_max_memory_on_cpu_constrains_cpu_and_allows_disk():
    # CPU gets no headroom reservation -- see resolve_max_memory's docstring
    # for why (no hard OS-level RAM cap exists for this codebase to collide
    # with, unlike torch.cuda.set_per_process_memory_fraction for VRAM).
    max_memory, mechanism = resolve_max_memory("cpu", 128 * 1024 * 1024)
    assert max_memory == {"cpu": 128 * 1024 * 1024, "disk": True}
    assert mechanism == "cpu-ram-budget-with-disk-offload"


def test_resolve_max_memory_on_cuda_reserves_activation_headroom():
    # Real bug found during GPU validation (writeup/compute-efficiency-gpu-
    # validation.md): handing accelerate the full nominal budget as the GPU
    # weight budget leaves no room for activations/CUDA context overhead,
    # and OOMs even when the model would otherwise fit. Only a fraction of
    # the nominal budget should reach accelerate as the weight budget.
    nominal_budget = 512 * 1024 * 1024
    max_memory, mechanism = resolve_max_memory("cuda", nominal_budget)
    expected_weight_budget = int(nominal_budget * (1 - DEFAULT_GPU_ACTIVATION_HEADROOM_FRACTION))
    assert max_memory[0] == expected_weight_budget
    assert max_memory[0] < nominal_budget
    assert max_memory["disk"] is True
    assert mechanism == "gpu-memory-budget"


def test_resolve_max_memory_on_cuda_respects_custom_headroom_fraction():
    nominal_budget = 1000 * 1024 * 1024
    max_memory, _ = resolve_max_memory(
        "cuda", nominal_budget, gpu_activation_headroom_fraction=0.1
    )
    assert max_memory[0] == int(nominal_budget * 0.9)


def test_offload_policy_describe_reflects_custom_headroom_fraction():
    policy = OffloadPolicy(memory_budget_bytes=256 * 1024 * 1024, gpu_activation_headroom_fraction=0.25)
    assert policy.describe()["gpu_activation_headroom_fraction"] == 0.25
