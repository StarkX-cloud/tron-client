"""Weight-space merging: train N replicas completely independently to
completion — zero inter-node communication during training at all — then
merge their weight deltas from a shared initialization. This is the
zero-communication alternative to local_sgd.py's periodic syncing.

Implements task arithmetic (Ilharco et al.) as the simple baseline merge,
and a TIES-lite variant (Yadav et al., "TIES-Merging") on top of it. TIES
exists because naive delta-averaging tends to wash out useful signal when
different shards' updates conflict on a parameter's sign — trimming to
the highest-magnitude deltas and electing a majority sign before
averaging keeps more of the task-specific signal that a plain average
would cancel out.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .model import TinyMLP


def _sample_batch(x: np.ndarray, y: np.ndarray, step: int, batch_size: int = 16):
    n = len(x)
    if n <= batch_size:
        return x, y
    rng = np.random.default_rng(step)
    idx = rng.choice(n, size=batch_size, replace=False)
    return x[idx], y[idx]


def train_independently(
    shards: list[tuple[np.ndarray, np.ndarray]],
    model_factory: Callable[[], TinyMLP],
    num_steps: int,
    lr: float,
) -> tuple[list[TinyMLP], np.ndarray]:
    """Each shard's model trains from the same shared init with zero
    communication between shards until this function returns. Returns
    the trained models and the shared init they all started from (needed
    to compute deltas for merging)."""
    models = [model_factory() for _ in shards]
    shared_init = models[0].get_flat_params().copy()
    for m in models[1:]:
        m.set_flat_params(shared_init.copy())

    for m, (x, y) in zip(models, shards):
        for step in range(num_steps):
            batch_x, batch_y = _sample_batch(x, y, step)
            m.train_step(batch_x, batch_y, lr)

    return models, shared_init


def merge_task_arithmetic(models: list[TinyMLP], shared_init: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Average each model's delta from the shared init, scale, add back."""
    deltas = np.stack([m.get_flat_params() - shared_init for m in models], axis=0)
    merged_delta = deltas.mean(axis=0) * scale
    return shared_init + merged_delta


def merge_ties(
    models: list[TinyMLP],
    shared_init: np.ndarray,
    trim_fraction: float = 0.2,
    scale: float = 1.0,
) -> np.ndarray:
    """TIES-lite, per parameter:
    1. Trim: zero out the smallest-magnitude `trim_fraction` of deltas
       across models (keep only the updates that moved the most).
    2. Elect sign: majority vote on sign among the surviving deltas.
    3. Merge: average only the deltas that agree with the elected sign.
    """
    if not 0.0 <= trim_fraction < 1.0:
        raise ValueError("trim_fraction must be in [0, 1)")

    deltas = np.stack([m.get_flat_params() - shared_init for m in models], axis=0)  # (num_models, num_params)

    magnitude = np.abs(deltas)
    threshold = np.quantile(magnitude, trim_fraction, axis=0)
    trimmed = np.where(magnitude >= threshold, deltas, 0.0)

    signs = np.sign(trimmed)
    elected_sign = np.sign(signs.sum(axis=0))  # 0 where votes tie

    agreeing_mask = (signs == elected_sign) & (elected_sign != 0)
    agreeing_deltas = np.where(agreeing_mask, trimmed, 0.0)
    count_agreeing = np.maximum(agreeing_mask.sum(axis=0), 1)
    merged_delta = (agreeing_deltas.sum(axis=0) / count_agreeing) * scale

    return shared_init + merged_delta
