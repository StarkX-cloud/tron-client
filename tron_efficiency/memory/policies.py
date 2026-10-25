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
        model = AutoModelForCausalLM.from_pretrained(config.model_name)
        model = model.to(config.device)
        model.eval()
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            policy_name=self.name,
            device_map=config.device,
            memory_budget_bytes=None,
            offload_folder=None,
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

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=offload_folder,
        )
        model.eval()
        loaded = LoadedModel(
            model=model,
            tokenizer=tokenizer,
            policy_name=self.name,
            device_map=dict(getattr(model, "hf_device_map", {})),
            memory_budget_bytes=self.memory_budget_bytes,
            offload_folder=offload_folder,
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
