"""Memory policy logic, tested without loading any real model."""

import pytest

from tron_efficiency.memory.policies import (
    DEFAULT_GPU_ACTIVATION_HEADROOM_FRACTION,
    BaselinePolicy,
    OffloadPolicy,
    UnsupportedDtypeCombination,
    _loading_kwargs_for_dtype,
    _requires_device_map,
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


def test_dtype_none_preserves_old_no_kwargs_behavior():
    # The old behavior (before the dtype-wiring fix) was to pass nothing to
    # from_pretrained at all -- preserved exactly for dtype=None so existing
    # callers/tests that don't care about dtype are unaffected.
    assert _loading_kwargs_for_dtype(None) == {}


def test_dtype_fp_variants_map_to_dtype_kwarg():
    # transformers renamed from_pretrained's torch_dtype= to dtype= (the
    # old name still works but is deprecated as of 5.16.1, the version
    # this was developed against) -- using the current name, not the
    # deprecated one.
    import torch

    assert _loading_kwargs_for_dtype("fp32") == {"dtype": torch.float32}
    assert _loading_kwargs_for_dtype("fp16") == {"dtype": torch.float16}
    assert _loading_kwargs_for_dtype("bf16") == {"dtype": torch.bfloat16}


def test_dtype_int8_produces_quantization_config():
    kwargs = _loading_kwargs_for_dtype("int8")
    assert "quantization_config" in kwargs
    assert kwargs["quantization_config"].load_in_8bit is True


def test_dtype_int8_with_cpu_offload_sets_fp32_cpu_offload_flag():
    kwargs = _loading_kwargs_for_dtype("int8", allow_cpu_offload=True)
    assert kwargs["quantization_config"].llm_int8_enable_fp32_cpu_offload is True


def test_dtype_int4_rejects_cpu_offload_combination():
    # Real bitsandbytes constraint: 4-bit has no fp32-cpu-offload escape
    # hatch the way 8-bit does.
    with pytest.raises(UnsupportedDtypeCombination):
        _loading_kwargs_for_dtype("int4", allow_cpu_offload=True)


def test_dtype_int4_without_offload_is_fine():
    kwargs = _loading_kwargs_for_dtype("int4", allow_cpu_offload=False)
    assert kwargs["quantization_config"].load_in_4bit is True


def test_unrecognized_dtype_raises():
    with pytest.raises(UnsupportedDtypeCombination):
        _loading_kwargs_for_dtype("fp8")


def test_requires_device_map_only_for_quantized_dtypes():
    assert _requires_device_map("int8") is True
    assert _requires_device_map("int4") is True
    assert _requires_device_map("fp16") is False
    assert _requires_device_map(None) is False
