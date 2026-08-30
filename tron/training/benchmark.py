"""Run this to reproduce Phase 3's benchmark:

    python -m tron.training.benchmark

It compares three ways of training the same model on the same non-IID
data across 4 simulated nodes:

  1. Baseline  — sync every local step (the naive all-reduce-every-step
     approach this benchmark exists to beat).
  2. Local SGD — DiLoCo-style: 10 local steps between each of 20 syncs
     (same 200 total local steps as the baseline, 10x fewer syncs).
  3. Merge     — train fully independently (zero communication during
     training at all), merge weight deltas once at the end via task
     arithmetic and TIES.

The problem is deliberately hard, not the friendliest case: classes are
only weakly separated (class_sep=1.0) and each shard sees ~95% one class
(skew=0.95) — the honest non-IID scenario, not an easy IID one that would
make every method look artificially good. See tests/test_local_sgd.py and
tests/test_training_merge.py for the same numbers pinned down as
regression tests.

This is intentionally small-scale — a hand-written numpy MLP (a few
hundred parameters), synthetic data, seconds of CPU time. The goal here is
to demonstrate the training *algorithms* honestly at a size anyone can
rerun in seconds, not to demonstrate a framework integration or a
frontier-scale model. Scaling the same algorithms up to a real model
(millions to billions of parameters, actual heterogeneous machines over a
real network) is real, substantial future work — see ROADMAP.md. What
this benchmark proves is that the mechanism itself (local SGD's
communication reduction, TIES merging's recovery from non-IID skew) is
implemented correctly and behaves the way the literature says it should,
on a scale where every number in this file can be verified in seconds.
"""
from __future__ import annotations

import numpy as np

from .data import make_classification_dataset, make_non_iid_shards, train_test_split
from .local_sgd import train_baseline_sync_every_step, train_local_sgd
from .merge import merge_task_arithmetic, merge_ties, train_independently
from .model import TinyMLP

NUM_FEATURES = 8
NUM_CLASSES = 4
NUM_SHARDS = 4
HIDDEN_DIM = 16
LR = 0.3


def _factory() -> TinyMLP:
    return TinyMLP(input_dim=NUM_FEATURES, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, seed=42)


def _build_problem():
    x_all, y_all = make_classification_dataset(
        num_samples=2500, num_features=NUM_FEATURES, num_classes=NUM_CLASSES, seed=0, class_sep=1.0,
    )
    x_train, y_train, x_test, y_test = train_test_split(x_all, y_all, test_fraction=0.2)
    shards = make_non_iid_shards(x_train, y_train, num_shards=NUM_SHARDS, num_classes=NUM_CLASSES, skew=0.95, seed=1)
    return shards, x_test, y_test


def run() -> dict:
    shards, x_test, y_test = _build_problem()

    baseline = train_baseline_sync_every_step(shards, _factory, num_steps=200, lr=LR)
    local_sgd = train_local_sgd(shards, _factory, num_rounds=20, local_steps=10, lr=LR)
    models, shared_init = train_independently(shards, _factory, num_steps=200, lr=LR)

    solo_accuracies = [m.accuracy(x_test, y_test) for m in models]

    merged_ta = _factory()
    merged_ta.set_flat_params(merge_task_arithmetic(models, shared_init))

    merged_ties = _factory()
    merged_ties.set_flat_params(merge_ties(models, shared_init, trim_fraction=0.2))

    return {
        "baseline": {
            "accuracy": baseline["model"].accuracy(x_test, y_test),
            "comm_bytes": baseline["comm_bytes"],
            "wall_clock_seconds": baseline["wall_clock_seconds"],
            "num_syncs": baseline["num_syncs"],
        },
        "local_sgd": {
            "accuracy": local_sgd["model"].accuracy(x_test, y_test),
            "comm_bytes": local_sgd["comm_bytes"],
            "wall_clock_seconds": local_sgd["wall_clock_seconds"],
            "num_syncs": local_sgd["num_syncs"],
        },
        "solo_shards": {
            "avg_accuracy": float(np.mean(solo_accuracies)),
            "per_shard_accuracy": solo_accuracies,
            "comm_bytes": 0,
        },
        "merge_task_arithmetic": {
            "accuracy": merged_ta.accuracy(x_test, y_test),
            "comm_bytes": 0,  # zero communication during training; one merge at the end
        },
        "merge_ties": {
            "accuracy": merged_ties.accuracy(x_test, y_test),
            "comm_bytes": 0,
        },
    }


def _print_report(results: dict) -> None:
    b = results["baseline"]
    l = results["local_sgd"]
    s = results["solo_shards"]
    ta = results["merge_task_arithmetic"]
    ties = results["merge_ties"]

    print("=" * 72)
    print("TRON Phase 3 benchmark - non-IID distributed training, 4 shards")
    print("(class_sep=1.0, skew=0.95 - a genuinely hard, not cherry-picked, split)")
    print("=" * 72)
    print(f"{'method':<28}{'accuracy':>10}{'comm bytes':>14}{'syncs':>10}")
    print("-" * 72)
    print(f"{'baseline (sync/step)':<28}{b['accuracy']:>10.3f}{b['comm_bytes']:>14,}{b['num_syncs']:>10}")
    print(f"{'local SGD (10 steps/sync)':<28}{l['accuracy']:>10.3f}{l['comm_bytes']:>14,}{l['num_syncs']:>10}")
    print(f"{'solo shard (avg, no sync)':<28}{s['avg_accuracy']:>10.3f}{s['comm_bytes']:>14,}{'0':>10}")
    print(f"{'merge: task arithmetic':<28}{ta['accuracy']:>10.3f}{ta['comm_bytes']:>14,}{'1':>10}")
    print(f"{'merge: TIES':<28}{ties['accuracy']:>10.3f}{ties['comm_bytes']:>14,}{'1':>10}")
    print("-" * 72)

    comm_reduction = b["comm_bytes"] / l["comm_bytes"]
    acc_cost = b["accuracy"] - l["accuracy"]
    print(f"\nLocal SGD: {comm_reduction:.0f}x less communication than the baseline, "
          f"for {acc_cost:.3f} less accuracy ({b['comm_bytes']:,} to {l['comm_bytes']:,} bytes).")

    merge_gain = ta["accuracy"] - s["avg_accuracy"]
    print(f"Merging: zero communication during training recovers {ta['accuracy']:.3f} accuracy "
          f"vs {s['avg_accuracy']:.3f} average for an unmerged solo shard "
          f"(+{merge_gain:.3f}), vs {b['accuracy']:.3f} for the fully-synced baseline.")


if __name__ == "__main__":
    _print_report(run())
