"""The actual Phase 3 comparison: naive sync-every-step training vs.
DiLoCo-style local SGD (many local steps between infrequent outer syncs).

Both functions train the same set of per-shard model replicas from the
same shared initialization, so their final accuracy is directly
comparable. What differs is how often they synchronize — and that
difference is what benchmark.py measures in communication bytes and
wall-clock. This is the mechanism, not a simulation of it: an "outer
sync" here is an actual average of the replicas' real parameter vectors.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .model import TinyMLP

BYTES_PER_FLOAT64 = 8


def _average_flat_params(models: list[TinyMLP]) -> np.ndarray:
    stacked = np.stack([m.get_flat_params() for m in models], axis=0)
    return stacked.mean(axis=0)


def _sample_batch(x: np.ndarray, y: np.ndarray, step: int, batch_size: int = 16):
    n = len(x)
    if n <= batch_size:
        return x, y
    rng = np.random.default_rng(step)
    idx = rng.choice(n, size=batch_size, replace=False)
    return x[idx], y[idx]


def _shared_init_replicas(model_factory: Callable[[], TinyMLP], num_replicas: int) -> list[TinyMLP]:
    models = [model_factory() for _ in range(num_replicas)]
    shared_init = models[0].get_flat_params().copy()
    for m in models[1:]:
        m.set_flat_params(shared_init.copy())
    return models


def train_baseline_sync_every_step(
    shards: list[tuple[np.ndarray, np.ndarray]],
    model_factory: Callable[[], TinyMLP],
    num_steps: int,
    lr: float,
) -> dict:
    """The baseline this benchmark exists to beat: every single local
    step is followed by an outer sync (parameter averaging) — the
    small-scale equivalent of all-reduce-every-step in real distributed
    SGD. Maximum communication, and the accuracy ceiling everything else
    is measured against.
    """
    models = _shared_init_replicas(model_factory, len(shards))
    num_params = models[0].num_params()

    comm_bytes = 0
    start = time.time()

    for step in range(num_steps):
        for m, (x, y) in zip(models, shards):
            batch_x, batch_y = _sample_batch(x, y, step)
            m.train_step(batch_x, batch_y, lr)

        averaged = _average_flat_params(models)
        for m in models:
            m.set_flat_params(averaged.copy())
        comm_bytes += len(shards) * num_params * BYTES_PER_FLOAT64

    return {
        "model": models[0],
        "comm_bytes": comm_bytes,
        "wall_clock_seconds": time.time() - start,
        "num_syncs": num_steps,
    }


def train_local_sgd(
    shards: list[tuple[np.ndarray, np.ndarray]],
    model_factory: Callable[[], TinyMLP],
    num_rounds: int,
    local_steps: int,
    lr: float,
) -> dict:
    """DiLoCo-lite: each shard trains `local_steps` steps completely
    independently — zero communication during that stretch — then all
    replicas are averaged once. Repeated for `num_rounds`. Total
    communication is `num_rounds` syncs instead of
    `num_rounds * local_steps`, which is the reduction this whole
    benchmark is about.
    """
    models = _shared_init_replicas(model_factory, len(shards))
    num_params = models[0].num_params()

    comm_bytes = 0
    start = time.time()

    for round_idx in range(num_rounds):
        for m, (x, y) in zip(models, shards):
            for local_step in range(local_steps):
                global_step = round_idx * local_steps + local_step
                batch_x, batch_y = _sample_batch(x, y, global_step)
                m.train_step(batch_x, batch_y, lr)

        averaged = _average_flat_params(models)
        for m in models:
            m.set_flat_params(averaged.copy())
        comm_bytes += len(shards) * num_params * BYTES_PER_FLOAT64

    return {
        "model": models[0],
        "comm_bytes": comm_bytes,
        "wall_clock_seconds": time.time() - start,
        "num_syncs": num_rounds,
    }
