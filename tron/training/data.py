"""Synthetic classification data, and non-IID sharding across simulated
nodes.

Data is generated locally (Gaussian blobs per class) rather than
downloaded, so the training demo is self-contained and reproducible
offline — no dataset download, no network dependency, same result on
every machine given the same seed.

Sharding is deliberately non-IID (each shard is skewed toward one
"primary" class) because that's the honest hard case for distributed
training: real heterogeneous nodes rarely see identically distributed
data, and identically-distributed shards would make local-SGD and
weight-merging look better than they actually are.
"""
from __future__ import annotations

import numpy as np


def make_classification_dataset(
    num_samples: int,
    num_features: int,
    num_classes: int,
    seed: int = 0,
    class_sep: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """One Gaussian blob per class, centers spread out by `class_sep`."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=class_sep, size=(num_classes, num_features))
    samples_per_class = num_samples // num_classes

    xs, ys = [], []
    for c in range(num_classes):
        xs.append(centers[c] + rng.normal(scale=1.0, size=(samples_per_class, num_features)))
        ys.append(np.full(samples_per_class, c))

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(x))
    return x[perm], y[perm]


def train_test_split(x: np.ndarray, y: np.ndarray, test_fraction: float = 0.2) -> tuple:
    """Split a single generated dataset into train/test.

    Deliberately exists so nobody calls make_classification_dataset()
    twice with two different seeds to get "train" and "test" sets — that
    draws independent random class centers for each call, so train and
    test end up describing two *different* classification problems that
    happen to share a class count. (This produced a real bug during
    development: a model scored ~3% accuracy — worse than the 25% random
    baseline for 4 classes — because "test" data was centered nowhere
    near where "train" had taught it to look.) Always generate one
    dataset and split it with this function instead.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    split_at = int(len(x) * (1.0 - test_fraction))
    return x[:split_at], y[:split_at], x[split_at:], y[split_at:]


def _take(pool: list, n: int) -> list:
    taken = pool[:n]
    del pool[:n]
    return taken


def make_non_iid_shards(
    x: np.ndarray,
    y: np.ndarray,
    num_shards: int,
    num_classes: int,
    skew: float = 0.8,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split (x, y) into `num_shards` shards. Shard i's primary class is
    `i % num_classes`; `skew` fraction of its samples come from that
    class, the rest are spread evenly across the other classes. This is
    the standard non-IID simulation used in federated-learning literature
    (e.g. the shard-based partitioning in McMahan et al.'s FedAvg paper).
    """
    if not 0.0 <= skew <= 1.0:
        raise ValueError("skew must be in [0, 1]")

    rng = np.random.default_rng(seed)
    class_indices = {c: rng.permutation(np.where(y == c)[0]).tolist() for c in range(num_classes)}

    samples_per_shard = len(x) // num_shards

    shards = []
    for shard_idx in range(num_shards):
        primary_class = shard_idx % num_classes
        n_primary = int(samples_per_shard * skew)
        n_rest = samples_per_shard - n_primary

        indices = _take(class_indices[primary_class], n_primary)

        other_classes = [c for c in range(num_classes) if c != primary_class]
        if other_classes:
            per_other = max(1, n_rest // len(other_classes))
            for c in other_classes:
                indices.extend(_take(class_indices[c], per_other))

        needed = samples_per_shard - len(indices)
        if needed > 0:
            leftover_pool = [i for pool in class_indices.values() for i in pool]
            rng.shuffle(leftover_pool)
            indices.extend(_take(leftover_pool, needed))
            # leftover_pool was a *copy* of the remaining indices, not the
            # live pools — remove what we took from the real pools too so
            # a later shard can't draw the same sample twice.
            taken_set = set(indices[-needed:]) if needed else set()
            for c in class_indices:
                class_indices[c] = [i for i in class_indices[c] if i not in taken_set]

        indices = np.array(indices[:samples_per_shard])
        shards.append((x[indices], y[indices]))

    return shards


def class_distribution(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Fraction of samples belonging to each class — used to confirm a
    shard is actually skewed the way make_non_iid_shards claims."""
    counts = np.bincount(y, minlength=num_classes)
    return counts / max(1, counts.sum())
