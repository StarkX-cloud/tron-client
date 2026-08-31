"""Tests for the Phase 1 execution spine: content addressing, dedup,
replay, and lineage-based recovery.
"""
import tempfile
from pathlib import Path

import pytest

from tron.spine import Artifact, ArtifactStore, EventLog, Task, TaskStatus, content_hash


# ---------------------------------------------------------------------------
# Content addressing
# ---------------------------------------------------------------------------

def test_content_hash_is_deterministic():
    assert content_hash(b"hello") == content_hash(b"hello")


def test_content_hash_differs_for_different_content():
    assert content_hash(b"hello") != content_hash(b"world")


def test_artifact_id_matches_its_own_content_hash():
    artifact = Artifact.from_bytes(b"payload")
    assert artifact.artifact_id == content_hash(b"payload")
    assert artifact.size_bytes == len(b"payload")


def test_task_id_is_deterministic_for_identical_work():
    t1 = Task.new(fn_hash="fn-a", input_hashes=("in-1", "in-2"))
    t2 = Task.new(fn_hash="fn-a", input_hashes=("in-2", "in-1"))  # order shouldn't matter
    assert t1.task_id == t2.task_id


def test_task_id_differs_for_different_inputs():
    t1 = Task.new(fn_hash="fn-a", input_hashes=("in-1",))
    t2 = Task.new(fn_hash="fn-a", input_hashes=("in-2",))
    assert t1.task_id != t2.task_id


def test_task_id_differs_by_attempt():
    t1 = Task.new(fn_hash="fn-a", input_hashes=("in-1",), attempt=0)
    t2 = Task.new(fn_hash="fn-a", input_hashes=("in-1",), attempt=1)
    assert t1.task_id != t2.task_id


# ---------------------------------------------------------------------------
# ArtifactStore
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return ArtifactStore(root=tmp_path / "artifacts")


def test_store_put_and_get_roundtrip(store):
    artifact = store.put(b"some bytes")
    assert store.get(artifact.artifact_id) == b"some bytes"


def test_store_dedups_identical_content(store):
    a1 = store.put(b"same content")
    a2 = store.put(b"same content")
    assert a1.artifact_id == a2.artifact_id
    # only one file should exist on disk for this id
    assert store.exists(a1.artifact_id)


def test_store_get_missing_raises(store):
    with pytest.raises(KeyError):
        store.get("does-not-exist")


def test_store_verify_detects_intact_artifact(store):
    artifact = store.put(b"trustworthy")
    assert store.verify(artifact.artifact_id) is True


def test_store_verify_false_for_missing(store):
    assert store.verify("nonexistent") is False


# ---------------------------------------------------------------------------
# ArtifactStore: at-rest encryption (TRON_ARTIFACT_ENCRYPTION_KEY)
# ---------------------------------------------------------------------------

@pytest.fixture
def fernet_key():
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()


@pytest.fixture
def encrypted_store(tmp_path, fernet_key):
    return ArtifactStore(root=tmp_path / "artifacts", encryption_key=fernet_key)


def test_encrypted_store_roundtrip(encrypted_store):
    artifact = encrypted_store.put(b"some secret bytes")
    assert encrypted_store.get(artifact.artifact_id) == b"some secret bytes"


def test_encrypted_store_artifact_id_matches_plaintext_content_hash(encrypted_store):
    """Encryption must be invisible to content-addressing: the id is
    always the hash of the plaintext, on or off, so nothing that
    references an artifact by hash (which is everything) has to care."""
    artifact = encrypted_store.put(b"some secret bytes")
    assert artifact.artifact_id == content_hash(b"some secret bytes")


def test_encrypted_store_bytes_on_disk_are_not_plaintext(tmp_path, fernet_key):
    store = ArtifactStore(root=tmp_path / "artifacts", encryption_key=fernet_key)
    artifact = store.put(b"this must not appear verbatim on disk")
    raw_on_disk = (tmp_path / "artifacts" / artifact.artifact_id[:2] / artifact.artifact_id).read_bytes()
    assert b"this must not appear verbatim on disk" not in raw_on_disk


def test_encrypted_store_wrong_key_cannot_read_another_stores_data(tmp_path, fernet_key):
    from cryptography.fernet import Fernet, InvalidToken

    writer = ArtifactStore(root=tmp_path / "artifacts", encryption_key=fernet_key)
    artifact = writer.put(b"only readable with the right key")

    wrong_key = Fernet.generate_key().decode()
    reader = ArtifactStore(root=tmp_path / "artifacts", encryption_key=wrong_key)
    with pytest.raises(InvalidToken):
        reader.get(artifact.artifact_id)
    assert reader.verify(artifact.artifact_id) is False


def test_enabling_encryption_on_an_existing_store_still_reads_old_plaintext(tmp_path, fernet_key):
    """A real, expected case: turning on TRON_ARTIFACT_ENCRYPTION_KEY on a
    deployment that already has artifacts on disk doesn't retroactively
    encrypt them. Old data must still be readable, not treated as
    corrupted just because the key doesn't "open" it — it was never
    locked in the first place."""
    plain_store = ArtifactStore(root=tmp_path / "artifacts")
    artifact = plain_store.put(b"written before encryption was ever turned on")

    now_encrypted = ArtifactStore(root=tmp_path / "artifacts", encryption_key=fernet_key)
    assert now_encrypted.get(artifact.artifact_id) == b"written before encryption was ever turned on"
    assert now_encrypted.verify(artifact.artifact_id) is True

    # and a fresh write through the now-encrypted store really is encrypted
    new_artifact = now_encrypted.put(b"written after encryption was turned on")
    raw_on_disk = (tmp_path / "artifacts" / new_artifact.artifact_id[:2] / new_artifact.artifact_id).read_bytes()
    assert b"written after encryption was turned on" not in raw_on_disk


def test_no_key_behaves_exactly_as_before(store):
    """The default (no TRON_ARTIFACT_ENCRYPTION_KEY) is unchanged
    plaintext-on-disk — every existing test and deployment is unaffected."""
    artifact = store.put(b"plaintext by default")
    raw_on_disk = (store.root / artifact.artifact_id[:2] / artifact.artifact_id).read_bytes()
    assert raw_on_disk == b"plaintext by default"


# ---------------------------------------------------------------------------
# EventLog: append + replay
# ---------------------------------------------------------------------------

@pytest.fixture
def log(tmp_path):
    return EventLog(path=tmp_path / "log.db")


def test_append_assigns_increasing_seq(log):
    e1 = log.append("task-1", "queued", {"fn_hash": "fn", "input_hashes": []})
    e2 = log.append("task-1", "started", {"node_id": "node-a"})
    assert e2.seq > e1.seq


def test_replay_returns_events_in_order(log):
    log.append("task-1", "queued", {"fn_hash": "fn", "input_hashes": []})
    log.append("task-1", "started", {"node_id": "node-a"})
    log.append("task-1", "completed", {"output_hash": "out-1"})

    events = list(log.replay())
    assert [e.type for e in events] == ["queued", "started", "completed"]


def test_events_for_filters_by_task(log):
    log.append("task-1", "queued", {"fn_hash": "fn", "input_hashes": []})
    log.append("task-2", "queued", {"fn_hash": "fn", "input_hashes": []})
    log.append("task-1", "completed", {"output_hash": "out-1"})

    events = log.events_for("task-1")
    assert len(events) == 2
    assert all(e.task_id == "task-1" for e in events)


# ---------------------------------------------------------------------------
# EventLog: snapshot (replay -> current state)
# ---------------------------------------------------------------------------

def test_snapshot_reconstructs_task_lifecycle(log):
    log.append("task-1", "queued", {"fn_hash": "fn-a", "input_hashes": ["in-1"]})
    log.append("task-1", "assigned", {"node_id": "node-a"})
    log.append("task-1", "started", {"node_id": "node-a"})
    log.append("task-1", "completed", {"output_hash": "out-1"})

    tasks = log.snapshot()
    task = tasks["task-1"]
    assert task.status == TaskStatus.COMPLETED
    assert task.node_id == "node-a"
    assert task.output_hash == "out-1"
    assert task.fn_hash == "fn-a"


def test_snapshot_up_to_seq_rewinds_state(log):
    e1 = log.append("task-1", "queued", {"fn_hash": "fn-a", "input_hashes": []})
    log.append("task-1", "assigned", {"node_id": "node-a"})
    log.append("task-1", "completed", {"output_hash": "out-1"})

    # Rewind to just after "queued" — completed event shouldn't be visible yet.
    early_state = log.snapshot(up_to_seq=e1.seq)
    assert early_state["task-1"].status == TaskStatus.QUEUED

    full_state = log.snapshot()
    assert full_state["task-1"].status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Lineage-based recovery
# ---------------------------------------------------------------------------

def test_recover_orphaned_tasks_finds_running_work_on_dead_node(log):
    log.append("task-1", "queued", {"fn_hash": "fn-a", "input_hashes": ["in-1"]})
    log.append("task-1", "assigned", {"node_id": "node-a"})
    log.append("task-1", "started", {"node_id": "node-a"})

    log.append("task-2", "queued", {"fn_hash": "fn-b", "input_hashes": []})
    log.append("task-2", "assigned", {"node_id": "node-a"})
    log.append("task-2", "started", {"node_id": "node-a"})
    log.append("task-2", "completed", {"output_hash": "out-2"})  # finished before dying

    orphaned = log.recover_orphaned_tasks("node-a")
    orphaned_ids = {t.task_id for t in orphaned}

    assert orphaned_ids == {"task-1"}  # task-2 completed, shouldn't be orphaned


def test_recover_orphaned_tasks_ignores_other_nodes(log):
    log.append("task-1", "queued", {"fn_hash": "fn-a", "input_hashes": []})
    log.append("task-1", "assigned", {"node_id": "node-b"})
    log.append("task-1", "started", {"node_id": "node-b"})

    assert log.recover_orphaned_tasks("node-a") == []


def test_recovered_task_can_be_resubmitted_as_next_attempt(log):
    log.append("task-1", "queued", {"fn_hash": "fn-a", "input_hashes": ["in-1"], "attempt": 0})
    log.append("task-1", "assigned", {"node_id": "node-a"})
    log.append("task-1", "started", {"node_id": "node-a"})

    orphaned = log.recover_orphaned_tasks("node-a")
    assert len(orphaned) == 1

    dead = orphaned[0]
    retry = Task.new(fn_hash=dead.fn_hash, input_hashes=dead.input_hashes, attempt=dead.attempt + 1)
    assert retry.task_id != dead.task_id  # new graph node, not a silent overwrite
    assert retry.fn_hash == dead.fn_hash
    assert retry.input_hashes == dead.input_hashes
