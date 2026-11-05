"""Memory-management policies for loading a model.

Two real execution paths, not a baseline relabeled with different numbers:

- ``BaselinePolicy`` loads the model straight onto the target device, no
  memory budget — this is what fails first when memory is tight.
- ``OffloadPolicy`` uses Hugging Face Accelerate's real device-map/offload
  machinery (``device_map="auto"`` + ``max_memory`` + ``offload_folder``) to
  keep the model under a stated memory budget, spilling the rest to disk
  (or, on a CUDA machine, to CPU RAM then disk). This is the same mechanism
  Accelerate ships for any user — it is not a TRON invention, and is
  labeled as such everywhere it's reported.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any

from tron_efficiency.workloads.config import WorkloadConfig

# Mirrors tron/training/lora_demo.py's workaround for flaky xet downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Imported once at module load, not inside load_model(): transformers'
# first import in a process is slow (lazy backend init) and must not land
# inside whichever policy happens to run first in a comparison, or that
# policy's "load time" measurement would include the other policy's import
# cost too.
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    policy_name: str
    device_map: Any
    memory_budget_bytes: int | None
    offload_folder: str | None
    dtype: str | None = None


class UnsupportedDtypeCombination(Exception):
    """Raised when a (dtype, policy, device) combination isn't something
    bitsandbytes/transformers actually supports, so the caller fails with a
    clear message instead of a confusing crash deep in from_pretrained."""


def _loading_kwargs_for_dtype(dtype: str | None, allow_cpu_offload: bool = False) -> dict:
    """What to pass to ``from_pretrained`` for a given requested dtype.

    Real gap this closes (found during the GPU validation run — see
    writeup/compute-efficiency-gpu-validation.md): ``BaselinePolicy`` used
    to pass nothing at all, so it silently loaded whatever dtype a
    checkpoint happened to be stored in on the Hub — only coincidentally
    matching what ``recommend()`` suggested. ``dtype=None`` preserves that
    old behavior exactly (no kwargs added) for backward compatibility;
    every other value is now an explicit, real instruction to
    ``from_pretrained``, not a suggestion that gets silently ignored.

    ``int8``/``int4`` use real bitsandbytes quantization (``load_in_8bit``/
    ``load_in_4bit`` via ``BitsAndBytesConfig``) — not a TRON invention,
    the same mechanism anyone using ``transformers`` has access to. A
    quantized model must be placed via ``device_map`` at load time and can
    never be moved with ``.to()`` afterward (transformers raises if you
    try), which is why callers of this function also need
    ``requires_device_map`` below to know not to call ``.to()``.
    """
    if dtype is None:
        return {}
    if dtype in ("fp32", "fp16", "bf16"):
        import torch

        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        # transformers renamed from_pretrained's torch_dtype= kwarg to
        # dtype= (torch_dtype still works but warns as deprecated as of
        # the version this was developed against, 5.16.1).
        return {"dtype": torch_dtype}
    if dtype in ("int8", "int4"):
        from transformers import BitsAndBytesConfig

        if dtype == "int8":
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=allow_cpu_offload
                )
            }
        if allow_cpu_offload:
            raise UnsupportedDtypeCombination(
                "4-bit quantization (int4) does not support combining with CPU/disk "
                "offload -- bitsandbytes only offers the fp32-cpu-offload escape hatch "
                "for 8-bit (int8). Use int8 if you need both quantization and offload, "
                "or drop offload if int4 is required."
            )
        return {"quantization_config": BitsAndBytesConfig(load_in_4bit=True)}
    raise UnsupportedDtypeCombination(f"unrecognized dtype: {dtype!r}")


def _requires_device_map(dtype: str | None) -> bool:
    """Quantized models (int8/int4) must be placed via device_map at load
    time; calling .to(device) on one afterward raises in transformers."""
    return dtype in ("int8", "int4")


class MemoryPolicy:
    """Base class. Subclasses implement ``load_model``."""

    name = "base"

    def load_model(self, config: WorkloadConfig) -> LoadedModel:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name}


class BaselinePolicy(MemoryPolicy):
    """No memory budget — load the whole model onto one device."""

    name = "baseline"

    def load_model(self, config: WorkloadConfig) -> LoadedModel:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        kwargs = _loading_kwargs_for_dtype(config.dtype)

        if _requires_device_map(config.dtype):
            # Quantized weights are placed at load time, not moved after.
            kwargs["device_map"] = "cpu" if config.device == "cpu" else {"": 0}
            model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
            model = model.to(config.device)
        model.eval()
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            policy_name=self.name,
            device_map=config.device,
            memory_budget_bytes=None,
            offload_folder=None,
            dtype=config.dtype,
        )


# Real bug found during GPU validation (writeup/compute-efficiency-gpu-
# validation.md, run 1): handing accelerate's max_memory a GPU weight
# budget equal to the *entire* VRAM budget leaves zero headroom for CUDA's
# own context/cuBLAS overhead and generate()'s activation/KV-cache tensors
# — both the baseline AND the "constrained" path OOM'd under a 600MB cap,
# even though the constrained path's actual weight footprint was far
# smaller. A 1.2GB cap with a 500MB weight budget (~58% headroom) is what
# actually worked end to end on a real T4. Default below reserves 60% of
# the budget for that headroom, deliberately more conservative than the
# ~58% that was empirically proven, since VRAM overhead isn't fixed across
# GPUs/model architectures and erring toward "degrades gracefully" beats
# "OOMs right at the edge." CPU does not get this treatment: nothing in
# this codebase imposes a hard OS-level cap on CPU RAM the way
# torch.cuda.set_per_process_memory_fraction does for VRAM, so there is no
# equivalent hard ceiling for CPU-side activations to collide with.
DEFAULT_GPU_ACTIVATION_HEADROOM_FRACTION = 0.6


def resolve_max_memory(
    device: str,
    memory_budget_bytes: int,
    gpu_activation_headroom_fraction: float = DEFAULT_GPU_ACTIVATION_HEADROOM_FRACTION,
) -> tuple[dict, str]:
    """Pure, testable piece of OffloadPolicy's logic: which memory to
    constrain given the requested device, independent of actually loading a
    model.

    For CUDA, ``memory_budget_bytes`` is the *total* VRAM budget the caller
    wants respected (weights + activations); only
    ``memory_budget_bytes * (1 - gpu_activation_headroom_fraction)`` of it
    is actually handed to accelerate as the weight-placement budget, so
    there's real headroom left for what accelerate itself doesn't account
    for. For CPU, the full budget is used as-is (see module docstring above
    for why the two cases differ).
    """
    if device == "cuda":
        weight_budget_bytes = int(
            memory_budget_bytes * (1 - gpu_activation_headroom_fraction)
        )
        return (
            {0: weight_budget_bytes, "cpu": "16GiB", "disk": True},
            "gpu-memory-budget",
        )
    return {"cpu": memory_budget_bytes, "disk": True}, "cpu-ram-budget-with-disk-offload"


class OffloadPolicy(MemoryPolicy):
    """Constrain resident memory via Accelerate's device_map/offload.

    ``memory_budget_bytes`` caps what accelerate will keep on
    ``config.device`` (GPU VRAM if CUDA is available, otherwise CPU RAM);
    anything that doesn't fit is spilled to ``offload_folder`` on disk. On a
    CUDA machine this is the spec's "GPU-memory-constrained execution" path
    verbatim — note that only part of ``memory_budget_bytes`` is actually
    handed to accelerate as the weight budget; see ``resolve_max_memory``
    for why. On a CPU-only machine (this one, during development) there is
    no GPU budget to enforce, so the budget is applied to CPU RAM instead
    and the excess goes to disk — a genuine, different memory-management
    mechanism, not a relabeled baseline. ``policy_name`` reflects which case
    actually ran.
    """

    name = "offload"

    def __init__(
        self,
        memory_budget_bytes: int,
        offload_folder: str | None = None,
        gpu_activation_headroom_fraction: float = DEFAULT_GPU_ACTIVATION_HEADROOM_FRACTION,
    ):
        self.memory_budget_bytes = memory_budget_bytes
        self.offload_folder = offload_folder
        self.gpu_activation_headroom_fraction = gpu_activation_headroom_fraction

    def load_model(self, config: WorkloadConfig) -> LoadedModel:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        offload_folder = self.offload_folder or tempfile.mkdtemp(
            prefix="tron_efficiency_offload_"
        )
        os.makedirs(offload_folder, exist_ok=True)

        max_memory, effective_mechanism = resolve_max_memory(
            config.device, self.memory_budget_bytes, self.gpu_activation_headroom_fraction
        )

        # int4 + offload isn't a supported bitsandbytes combination (see
        # _loading_kwargs_for_dtype's docstring) -- raises a clear
        # UnsupportedDtypeCombination rather than attempting it and hitting
        # a confusing failure deep inside from_pretrained. The caller
        # (run_workload) catches this the same way it catches ImportError.
        dtype_kwargs = _loading_kwargs_for_dtype(config.dtype, allow_cpu_offload=True)

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=offload_folder,
            **dtype_kwargs,
        )
        model.eval()
        loaded = LoadedModel(
            model=model,
            tokenizer=tokenizer,
            policy_name=self.name,
            device_map=dict(getattr(model, "hf_device_map", {})),
            memory_budget_bytes=self.memory_budget_bytes,
            offload_folder=offload_folder,
            dtype=config.dtype,
        )
        loaded.device_map["_mechanism"] = effective_mechanism
        # The weight-placement budget actually handed to accelerate, which
        # for CUDA is smaller than memory_budget_bytes (headroom reserved
        # for activations — see resolve_max_memory). Surfaced explicitly so
        # a result never implies the full nominal budget was used for
        # weights when it wasn't.
        loaded.device_map["_gpu_weight_budget_bytes"] = (
            max_memory.get(0) if config.device == "cuda" else None
        )
        return loaded

    def describe(self) -> dict:
        return {
            "name": self.name,
            "memory_budget_bytes": self.memory_budget_bytes,
            "offload_folder": self.offload_folder,
            "gpu_activation_headroom_fraction": self.gpu_activation_headroom_fraction,
        }
