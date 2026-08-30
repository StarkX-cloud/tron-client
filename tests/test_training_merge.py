"""Tests for Phase 3 weight-space merging."""
import numpy as np
import pytest

from tron.training.data import make_classification_dataset, make_non_iid_shards, train_test_split
from tron.training.merge import merge_task_arithmetic, merge_ties, train_independently
from tron.training.model import TinyMLP


class _FakeModel:
    """Minimal stand-in exposing get_flat_params(), so merge tests don't
    need a real TinyMLP — just numbers to merge."""

    def __init__(self, flat: np.ndarray):
        self._flat = flat

    def get_flat_params(self) -> np.ndarray:
        return self._flat


def test_task_arithmetic_returns_init_when_all_deltas_zero():
    shared_init = np.array([1.0, 2.0, 3.0])
    models = [_FakeModel(shared_init.copy()) for _ in range(3)]
    merged = merge_task_arithmetic(models, shared_init)
    np.testing.assert_allclose(merged, shared_init)


def test_task_arithmetic_averages_deltas():
    shared_init = np.array([0.0, 0.0])
    models = [_FakeModel(np.array([2.0, 0.0])), _FakeModel(np.array([-2.0, 0.0])), _FakeModel(np.array([4.0, 0.0]))]
    merged = merge_task_arithmetic(models, shared_init)
    # deltas: 2, -2, 4 -> mean = 4/3
    np.testing.assert_allclose(merged, [4.0 / 3.0, 0.0])


def test_ties_returns_init_when_all_deltas_zero():
    shared_init = np.array([1.0, 2.0, 3.0])
    models = [_FakeModel(shared_init.copy()) for _ in range(3)]
    merged = merge_ties(models, shared_init)
    np.testing.assert_allclose(merged, shared_init)


def test_ties_rejects_invalid_trim_fraction():
    shared_init = np.array([0.0])
    models = [_FakeModel(np.array([1.0]))]
    with pytest.raises(ValueError):
        merge_ties(models, shared_init, trim_fraction=1.0)
    with pytest.raises(ValueError):
        merge_ties(models, shared_init, trim_fraction=-0.1)


def test_ties_drops_disagreeing_small_minority():
    """Two models push a parameter strongly positive, one pushes it
    slightly negative. TIES should elect the majority (positive) sign and
    the result should end up positive — unlike a naive average, which
    could still land positive here too, but by less (this checks TIES
    doesn't get dragged toward the minority the way naive averaging
    would with a *larger* minority delta)."""
    shared_init = np.array([0.0])
    models = [_FakeModel(np.array([10.0])), _FakeModel(np.array([10.0])), _FakeModel(np.array([-9.0]))]

    naive = merge_task_arithmetic(models, shared_init)
    ties = merge_ties(models, shared_init, trim_fraction=0.0)

    assert ties[0] > 0
    # TIES excludes the disagreeing delta entirely once outvoted, so it
    # should end up further from zero in the majority direction than the
    # naive average, which is dragged down by the minority vote.
    assert ties[0] > naive[0]


# ---------------------------------------------------------------------------
# The actual Phase 3 claim: merging heavily-skewed independently-trained
# models recovers most of what full-communication training would have
# gotten, at zero communication cost during training. Fixed seeds, locked
# numbers — see test_local_sgd.py's sibling test for the same problem
# used to compare against the baseline/local-SGD numbers.
# ---------------------------------------------------------------------------

NUM_FEATURES = 8
NUM_CLASSES = 4
NUM_SHARDS = 4


def _hard_non_iid_problem():
    x_all, y_all = make_classification_dataset(
        num_samples=2500, num_features=NUM_FEATURES, num_classes=NUM_CLASSES, seed=0, class_sep=1.0,
    )
    x_train, y_train, x_test, y_test = train_test_split(x_all, y_all, test_fraction=0.2)
    shards = make_non_iid_shards(x_train, y_train, num_shards=NUM_SHARDS, num_classes=NUM_CLASSES, skew=0.95, seed=1)
    return shards, x_test, y_test


def _factory():
    return TinyMLP(input_dim=NUM_FEATURES, hidden_dim=16, num_classes=NUM_CLASSES, seed=42)


def test_merged_model_beats_average_solo_shard_on_held_out_data():
    """Each shard sees ~95% one class — its solo model generalizes badly
    to the balanced test set. Merging recovers most of what a fully
    synchronized model would have gotten, without any communication
    during training at all.
    """
    shards, x_test, y_test = _hard_non_iid_problem()
    models, shared_init = train_independently(shards, _factory, num_steps=200, lr=0.3)

    solo_accuracies = [m.accuracy(x_test, y_test) for m in models]
    avg_solo_acc = np.mean(solo_accuracies)

    merged = _factory()
    merged.set_flat_params(merge_task_arithmetic(models, shared_init))
    merged_acc = merged.accuracy(x_test, y_test)

    assert avg_solo_acc == pytest.approx(0.5875, abs=0.02)
    assert merged_acc == pytest.approx(0.81, abs=0.02)
    assert merged_acc > avg_solo_acc + 0.15  # not a marginal effect


def test_ties_merge_matches_or_beats_task_arithmetic_here():
    shards, x_test, y_test = _hard_non_iid_problem()
    models, shared_init = train_independently(shards, _factory, num_steps=200, lr=0.3)

    naive = _factory()
    naive.set_flat_params(merge_task_arithmetic(models, shared_init))

    ties = _factory()
    ties.set_flat_params(merge_ties(models, shared_init, trim_fraction=0.2))

    assert ties.accuracy(x_test, y_test) >= naive.accuracy(x_test, y_test) - 0.01
