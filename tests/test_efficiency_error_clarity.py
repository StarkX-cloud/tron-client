"""Tests for _clarify_error: turning known-confusing failures into
actionable messages, grounded only in what's actually been measured --
never an invented numeric "minimum budget" (see
writeup/compute-efficiency-gpu-validation.md for why no such universal
number exists yet)."""

from tron_efficiency.workloads.config import WorkloadConfig
from tron_efficiency.workloads.runner import _clarify_error


def test_whole_model_to_disk_error_gets_actionable_guidance():
    exc = ValueError(
        "You are trying to offload the whole model to the disk. Please use the "
        "`disk_offload` function instead."
    )
    config = WorkloadConfig(device="cuda")
    msg = _clarify_error(exc, "offload", config)
    assert "Raise memory_budget_bytes" in msg
    assert "disk_offload" in msg  # original error preserved, not swallowed


def test_cuda_accelerator_error_references_known_quirk():
    exc = RuntimeError("Cannot access accelerator device when none is available.")
    config = WorkloadConfig(device="cpu", dtype="int8")
    msg = _clarify_error(exc, "offload", config)
    assert "CUDA" in msg
    assert "ROADMAP.md" in msg


def test_cuda_oom_on_offload_references_confirmed_data_points():
    exc = RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB.")
    config = WorkloadConfig(device="cuda")
    msg = _clarify_error(exc, "offload", config)
    assert "600MB" in msg and "1.2GB" in msg
    assert "pythia-410m" in msg


def test_cuda_oom_on_baseline_does_not_reference_offload_data_points():
    # The confirmed 600MB/1.2GB data points are specifically about
    # OffloadPolicy's budget -- a baseline OOM has nothing to do with them.
    exc = RuntimeError("CUDA out of memory.")
    config = WorkloadConfig(device="cuda")
    msg = _clarify_error(exc, "baseline", config)
    assert "600MB" not in msg
    assert "out of memory under this policy's budget" in msg


def test_unrecognized_error_passes_through_with_type_name():
    exc = KeyError("something unrelated")
    config = WorkloadConfig(device="cpu")
    msg = _clarify_error(exc, "baseline", config)
    assert msg.startswith("KeyError:")


def test_unsupported_dtype_combination_passes_through_unchanged():
    from tron_efficiency.memory.policies import UnsupportedDtypeCombination

    exc = UnsupportedDtypeCombination("4-bit quantization does not support offload")
    config = WorkloadConfig(device="cuda", dtype="int4")
    msg = _clarify_error(exc, "offload", config)
    assert msg.startswith("UnsupportedDtypeCombination:")
    assert "4-bit quantization does not support offload" in msg
