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
