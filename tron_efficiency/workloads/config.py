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

    def to_dict(self) -> dict:
        return asdict(self)
