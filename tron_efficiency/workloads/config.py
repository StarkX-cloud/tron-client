"""Configuration for a single inference workload run."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WorkloadConfig:
    """Everything needed to reproduce one run.

    ``model_name`` defaults to EleutherAI/pythia-70m — the same small model
    already used by tron/training/lora_demo.py, so it's a known-good choice
    for experimentation and may already be cached locally.
    """

    model_name: str = "EleutherAI/pythia-70m"
    prompt: str = "The quick brown fox jumps over the lazy dog, and"
    max_new_tokens: int = 32
    batch_size: int = 1
    device: str = "cpu"  # "cpu" or "cuda" — chosen explicitly, never guessed
    seed: int = 0
    # None = don't pass a dtype/torch_dtype argument at all (the old, pre-dtype-control
    # behavior: whatever the checkpoint happens to be stored as). One of
    # "fp32"/"fp16"/"bf16"/"int8"/"int4" explicitly requests that precision
    # -- see memory/policies.py for how each is actually loaded. Added
    # after discovering (writeup/compute-efficiency-gpu-validation.md) that
    # BaselinePolicy silently loaded whatever dtype a checkpoint happened
    # to be stored in, which only coincidentally matched what the advisor
    # recommended.
    dtype: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
