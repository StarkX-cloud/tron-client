"""HTTP-level test for /training/run_demo — the Phase 3 -> Phase 1 wiring
exercised through the actual FastAPI app, not just the underlying
function (which tests/test_spine_integration.py already covers)."""
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    # TRON_SPINE_DIR isolates the on-disk event log / artifact store per
    # test — without it, reloading queue_server does NOT reset those (see
    # queue_server.py's _SPINE_DIR comment), and these tests specifically
    # assert on the *total* contents of the spine log, so leaked state
    # from another test would silently inflate the counts.
    monkeypatch.setenv("TRON_SPINE_DIR", str(tmp_path / "spine"))
    import queue_server
    importlib.reload(queue_server)
    return TestClient(queue_server.app), queue_server


def test_run_demo_returns_a_real_result(client):
    tc, qs = client
    resp = tc.post("/training/run_demo", json={"num_shards": 2, "num_rounds": 2, "local_steps": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["accuracy"] <= 1.0
    assert body["comm_bytes"] > 0
    assert body["num_syncs"] == 2
    assert body["shard_node_ids"] == ["shard-0", "shard-1"]


def test_run_demo_records_into_the_spine_log(client):
    tc, qs = client
    tc.post("/training/run_demo", json={"num_shards": 2, "num_rounds": 2, "local_steps": 2})

    events = tc.get("/spine/events").json()["events"]
    completed = [e for e in events if e["type"] == "completed"]
    assert len(completed) == 4  # num_rounds * num_shards


def test_run_demo_registers_shard_workers_visible_at_workers_endpoint(client):
    tc, qs = client
    tc.post("/training/run_demo", json={"num_shards": 3, "num_rounds": 1, "local_steps": 1})

    workers = tc.get("/workers").json()
    assert set(workers.keys()) == {"shard-0", "shard-1", "shard-2"}
    assert all(w["status"] == "done" for w in workers.values())


def test_run_demo_shard_workers_are_not_matched_real_jobs(client):
    """Shard workers finish as status='done', not 'idle' — they must not
    look like assignable capacity to the Phase 2b match step."""
    tc, qs = client
    tc.post("/training/run_demo", json={"num_shards": 2, "num_rounds": 1, "local_steps": 1})

    tc.post("/submit", json={"function": "unrelated-job", "priority": 5})
    qs._run_match_cycle()

    assert qs.pending_assignment == {}
