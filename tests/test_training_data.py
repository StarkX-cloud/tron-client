"""Tests for the Phase 3 synthetic data + non-IID sharding."""
import numpy as np
import pytest

from tron.training.data import (
    class_distribution,
    make_classification_dataset,
    make_non_iid_shards,
    train_test_split,
)


def test_dataset_shapes():
    x, y = make_classification_dataset(num_samples=300, num_features=5, num_classes=3, seed=0)
    assert x.shape == (300, 5)
    assert y.shape == (300,)
    assert set(np.unique(y)) == {0, 1, 2}


def test_dataset_is_reproducible_given_same_seed():
    x1, y1 = make_classification_dataset(num_samples=100, num_features=4, num_classes=2, seed=7)
    x2, y2 = make_classification_dataset(num_samples=100, num_features=4, num_classes=2, seed=7)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_shards_have_no_overlapping_samples():
    x, y = make_classification_dataset(num_samples=400, num_features=4, num_classes=4, seed=0)
    shards = make_non_iid_shards(x, y, num_shards=4, num_classes=4, skew=0.8, seed=1)

    seen_rows = set()
    for shard_x, _ in shards:
        for row in shard_x:
            key = tuple(row.round(10))
            assert key not in seen_rows, "duplicate sample across shards"
            seen_rows.add(key)


def test_shards_are_roughly_equal_sized():
    x, y = make_classification_dataset(num_samples=400, num_features=4, num_classes=4, seed=0)
    shards = make_non_iid_shards(x, y, num_shards=4, num_classes=4, skew=0.8, seed=1)
    sizes = [len(sx) for sx, _ in shards]
    assert max(sizes) - min(sizes) <= 2  # integer-division rounding only


def test_shards_are_skewed_toward_primary_class():
    x, y = make_classification_dataset(num_samples=800, num_features=4, num_classes=4, seed=0)
    shards = make_non_iid_shards(x, y, num_shards=4, num_classes=4, skew=0.8, seed=1)

    for shard_idx, (_, shard_y) in enumerate(shards):
        primary_class = shard_idx % 4
        dist = class_distribution(shard_y, num_classes=4)
        # Should be heavily skewed toward its primary class — much more
        # than the 1/4 an IID shard would show.
        assert dist[primary_class] > 0.5


def test_zero_skew_is_roughly_uniform():
    x, y = make_classification_dataset(num_samples=800, num_features=4, num_classes=4, seed=0)
    shards = make_non_iid_shards(x, y, num_shards=4, num_classes=4, skew=0.0, seed=1)

    for _, shard_y in shards:
        dist = class_distribution(shard_y, num_classes=4)
        # No class should dominate when skew=0.
        assert dist.max() < 0.5


def test_invalid_skew_rejected():
    x, y = make_classification_dataset(num_samples=100, num_features=4, num_classes=2, seed=0)
    with pytest.raises(ValueError):
        make_non_iid_shards(x, y, num_shards=2, num_classes=2, skew=1.5, seed=0)


def test_train_test_split_covers_all_samples_without_overlap():
    x, y = make_classification_dataset(num_samples=100, num_features=4, num_classes=2, seed=0)
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_fraction=0.2)
    assert len(x_train) == 80
    assert len(x_test) == 20
    np.testing.assert_array_equal(x_train, x[:80])
    np.testing.assert_array_equal(x_test, x[80:])


def test_train_test_split_rejects_invalid_fraction():
    x, y = make_classification_dataset(num_samples=10, num_features=2, num_classes=2, seed=0)
    with pytest.raises(ValueError):
        train_test_split(x, y, test_fraction=1.5)
    with pytest.raises(ValueError):
        train_test_split(x, y, test_fraction=0.0)
