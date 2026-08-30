"""Tests for wiring Phase 3's training loop into the Phase 1 spine.

The property that matters most: recording must not change the
computation. If it did, a visualization of "what training did" would be
lying about what training actually did.
"""
import numpy as np
import pytest

from tron.spine import EventLog, ArtifactStore, TaskStatus
from tron.training.data import make_classification_dataset, make_non_iid_shards
from tron.training.local_sgd import train_local_sgd
from tron.training.model import TinyMLP
from tron.training.spine_integration import run_local_sgd_with_spine

NUM_FEATURES = 4
NUM_CLASSES = 2
NUM_SHARDS = 2


def _factory():
    return TinyMLP(input_dim=NUM_FEATURES, hidden_dim=6, num_classes=NUM_CLASSES, seed=7)


def _shards():
    x, y = make_classification_dataset(num_samples=200, num_features=NUM_FEATURES, num_classes=NUM_CLASSES, seed=0)
    return make_non_iid_shards(x, y, num_shards=NUM_SHARDS, num_classes=NUM_CLASSES, skew=0.8, seed=1)


@pytest.fixture
def event_log(tmp_path):
    return EventLog(path=tmp_path / "log.db")


@pytest.fixture
def artifact_store(tmp_path):
    return ArtifactStore(root=tmp_path / "artifacts")


def test_instrumented_run_matches_uninstrumented_run_bit_for_bit(event_log, artifact_store):
    shards = _shards()

    uninstrumented = train_local_sgd(shards, _factory, num_rounds=3, local_steps=5, lr=0.2)
    instrumented = run_local_sgd_with_spine(
        shards, _factory, num_rounds=3, local_steps=5, lr=0.2,
        event_log=event_log, artifact_store=artifact_store,
    )

    np.testing.assert_array_equal(
        uninstrumented["model"].get_flat_params(),
        instrumented["model"].get_flat_params(),
    )
    assert uninstrumented["comm_bytes"] == instrumented["comm_bytes"]
    assert uninstrumented["num_syncs"] == instrumented["num_syncs"]


def test_records_one_completed_task_per_shard_per_round(event_log, artifact_store):
    shards = _shards()
    run_local_sgd_with_spine(
        shards, _factory, num_rounds=3, local_steps=2, lr=0.2,
        event_log=event_log, artifact_store=artifact_store,
    )

    tasks = event_log.snapshot()
    assert len(tasks) == 3 * NUM_SHARDS  # num_rounds * num_shards
    assert all(t.status == TaskStatus.COMPLETED for t in tasks.values())


def test_tasks_are_attributed_to_their_shard_node(event_log, artifact_store):
    shards = _shards()
    result = run_local_sgd_with_spine(
        shards, _factory, num_rounds=2, local_steps=2, lr=0.2,
        event_log=event_log, artifact_store=artifact_store,
    )

    tasks = event_log.snapshot()
    node_ids_seen = {t.node_id for t in tasks.values()}
    assert node_ids_seen == set(result["shard_node_ids"])


def test_output_artifacts_are_retrievable_and_are_real_weight_vectors(event_log, artifact_store):
    shards = _shards()
    run_local_sgd_with_spine(
        shards, _factory, num_rounds=1, local_steps=2, lr=0.2,
        event_log=event_log, artifact_store=artifact_store,
    )

    tasks = event_log.snapshot()
    num_params = _factory().num_params()

    for task in tasks.values():
        assert task.output_hash is not None
        raw = artifact_store.get(task.output_hash)
        weights = np.frombuffer(raw, dtype=np.float64)
        assert weights.shape == (num_params,)


def test_attempt_number_tracks_round_index(event_log, artifact_store):
    shards = _shards()
    run_local_sgd_with_spine(
        shards, _factory, num_rounds=4, local_steps=1, lr=0.2,
        event_log=event_log, artifact_store=artifact_store,
    )
    tasks = event_log.snapshot()
    # Each task's id is content-addressed by (fn_hash, input_hashes,
    # attempt) — attempt is round_idx, so 4 rounds x 2 shards must
    # produce 8 distinct task ids, not fewer (which would mean rounds
    # collided onto the same task id).
    assert len(tasks) == 8
