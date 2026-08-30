"""Tests for Phase 3's naive-sync-every-step baseline vs. DiLoCo-style
local SGD: communication-byte accounting, and the actual accuracy/
communication tradeoff on a fixed, reproducible problem.
"""
import numpy as np
import pytest

from tron.training.data import make_classification_dataset, make_non_iid_shards, train_test_split
from tron.training.local_sgd import BYTES_PER_FLOAT64, train_baseline_sync_every_step, train_local_sgd
from tron.training.model import TinyMLP

NUM_FEATURES = 8
NUM_CLASSES = 4
NUM_SHARDS = 4


def _hard_non_iid_problem():
    """A fixed, reproducible, genuinely hard (not trivially separable)
    non-IID problem — see benchmark.py for why these specific parameters
    were chosen (an easier problem makes every method score ~100% and
    proves nothing; this one was tuned empirically to actually
    differentiate approaches)."""
    x_all, y_all = make_classification_dataset(
        num_samples=2500, num_features=NUM_FEATURES, num_classes=NUM_CLASSES, seed=0, class_sep=1.0,
    )
    x_train, y_train, x_test, y_test = train_test_split(x_all, y_all, test_fraction=0.2)
    shards = make_non_iid_shards(x_train, y_train, num_shards=NUM_SHARDS, num_classes=NUM_CLASSES, skew=0.95, seed=1)
    return shards, x_test, y_test


def _factory():
    return TinyMLP(input_dim=NUM_FEATURES, hidden_dim=16, num_classes=NUM_CLASSES, seed=42)


def test_baseline_comm_bytes_formula():
    shards, _, _ = _hard_non_iid_problem()
    result = train_baseline_sync_every_step(shards, _factory, num_steps=5, lr=0.1)
    num_params = _factory().num_params()
    assert result["comm_bytes"] == NUM_SHARDS * num_params * BYTES_PER_FLOAT64 * 5
    assert result["num_syncs"] == 5


def test_local_sgd_comm_bytes_formula():
    shards, _, _ = _hard_non_iid_problem()
    result = train_local_sgd(shards, _factory, num_rounds=3, local_steps=10, lr=0.1)
    num_params = _factory().num_params()
    # Only 3 syncs happened (once per round), regardless of local_steps.
    assert result["comm_bytes"] == NUM_SHARDS * num_params * BYTES_PER_FLOAT64 * 3
    assert result["num_syncs"] == 3


def test_local_sgd_uses_far_less_communication_for_equal_total_local_steps():
    shards, _, _ = _hard_non_iid_problem()
    total_local_steps = 200

    baseline = train_baseline_sync_every_step(shards, _factory, num_steps=total_local_steps, lr=0.3)
    local_sgd = train_local_sgd(shards, _factory, num_rounds=20, local_steps=10, lr=0.3)

    assert local_sgd["comm_bytes"] == baseline["comm_bytes"] / 10  # 10x fewer syncs, exactly


def test_local_sgd_reaches_comparable_accuracy_to_baseline_with_far_less_communication():
    """The actual Phase 3 benchmark claim, locked to fixed seeds so this
    is a real reproducible number, not a fuzzy "should work" hope. See
    scratch exploration in this file's sibling benchmark.py docstring for
    how these thresholds were derived empirically before being asserted.
    """
    shards, x_test, y_test = _hard_non_iid_problem()

    baseline = train_baseline_sync_every_step(shards, _factory, num_steps=200, lr=0.3)
    local_sgd = train_local_sgd(shards, _factory, num_rounds=20, local_steps=10, lr=0.3)

    baseline_acc = baseline["model"].accuracy(x_test, y_test)
    local_sgd_acc = local_sgd["model"].accuracy(x_test, y_test)

    assert baseline_acc == pytest.approx(0.872, abs=0.01)
    assert local_sgd_acc == pytest.approx(0.856, abs=0.01)

    # The actual claim: ~10x less communication for a small accuracy cost.
    assert local_sgd["comm_bytes"] == baseline["comm_bytes"] / 10
    assert baseline_acc - local_sgd_acc < 0.03
