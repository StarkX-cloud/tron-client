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


def resolve_max_memory(device: str, memory_budget_bytes: int) -> tuple[dict, str]:
    """Pure, testable piece of OffloadPolicy's logic: which memory to
    constrain given the requested device, independent of actually loading a
    model."""
    if device == "cuda":
        return (
            {0: memory_budget_bytes, "cpu": "16GiB", "disk": True},
            "gpu-memory-budget",
        )
    return {"cpu": memory_budget_bytes, "disk": True}, "cpu-ram-budget-with-disk-offload"


class OffloadPolicy(MemoryPolicy):
    """Constrain resident memory via Accelerate's device_map/offload.

    ``memory_budget_bytes`` caps what accelerate will keep on
    ``config.device`` (GPU VRAM if CUDA is available, otherwise CPU RAM);
    anything that doesn't fit is spilled to ``offload_folder`` on disk. On a
    CUDA machine this is the spec's "GPU-memory-constrained execution" path
    verbatim. On a CPU-only machine (this one, during development) there is
    no GPU budget to enforce, so the budget is applied to CPU RAM instead
    and the excess goes to disk — a genuine, different memory-management
    mechanism, not a relabeled baseline. ``policy_name`` reflects which case
    actually ran.
    """

    name = "offload"

    def __init__(self, memory_budget_bytes: int, offload_folder: str | None = None):
        self.memory_budget_bytes = memory_budget_bytes
        self.offload_folder = offload_folder

    def load_model(self, config: WorkloadConfig) -> LoadedModel:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        offload_folder = self.offload_folder or tempfile.mkdtemp(
            prefix="tron_efficiency_offload_"
        )
        os.makedirs(offload_folder, exist_ok=True)

        max_memory, effective_mechanism = resolve_max_memory(
            config.device, self.memory_budget_bytes
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
        return loaded

    def describe(self) -> dict:
        return {
            "name": self.name,
            "memory_budget_bytes": self.memory_budget_bytes,
            "offload_folder": self.offload_folder,
        }
