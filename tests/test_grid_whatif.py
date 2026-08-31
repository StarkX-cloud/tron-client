"""Tests for POST /scheduler/whatif — the Grid v2 "drag a worker, see what
the scheduler would do" endpoint. It must be non-destructive and must
actually reflect the matcher's real scoring.
"""
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRON_SPINE_DIR", str(tmp_path / "spine"))
    import queue_server
    importlib.reload(queue_server)
    return TestClient(queue_server.app), queue_server


def _register(tc, name):
    resp = tc.post("/register_worker", json={"name": name})
    assert resp.status_code == 200
    return resp.json()["auth_token"]


def _heartbeat(tc, name, token, latency_ms=None, bw=None):
    # Fail-closed once a worker has a token (queue_server.py) — send the
    # one issued at registration, same as a real worker.py client would.
    body = {"worker_name": name}
    if latency_ms is not None:
        body["latency_ms"] = latency_ms
    if bw is not None:
        body["bandwidth_mbps_down"] = bw
    tc.post(f"/heartbeat/{name}", json=body, headers={"X-TRON-AUTH": token})


def test_whatif_is_a_noop_with_no_overrides(client):
    tc, _ = client
    token_a = _register(tc, "a")
    token_b = _register(tc, "b")
    _heartbeat(tc, "a", token_a, latency_ms=5.0)
    _heartbeat(tc, "b", token_b, latency_ms=300.0)
    tc.post("/submit", json={"function": "fn", "priority": 5, "compute_weight": 10})

    r = tc.post("/scheduler/whatif", json={"overrides": {}})
    body = r.json()
    assert body["baseline"] == body["hypothetical"]
    assert body["changed_jobs"] == []


def test_whatif_does_not_touch_real_state(client):
    tc, qs = client
    token = _register(tc, "a")
    _heartbeat(tc, "a", token, latency_ms=5.0)
    r1 = tc.post("/submit", json={"function": "fn", "priority": 5})
    job_id = r1.json()["job_id"]

    tc.post("/scheduler/whatif", json={"overrides": {"a": {"latency_ms": 999.0}}})

    # real topology and queue unchanged
    assert qs.topology.latency("master", "a") == 5.0
    assert [j["id"] for j in qs.job_queue] == [job_id]
    assert qs.pending_assignment == {}


def test_whatif_reflects_the_matchers_scoring(client):
    tc, _ = client
    token_near = _register(tc, "near")
    token_far = _register(tc, "far")
    _heartbeat(tc, "near", token_near, latency_ms=5.0)
    _heartbeat(tc, "far", token_far, latency_ms=300.0)
    r = tc.post("/submit", json={"function": "fn", "priority": 5, "compute_weight": 10})
    job_id = r.json()["job_id"]

    body = tc.post("/scheduler/whatif", json={
        "overrides": {"far": {"latency_ms": 1.0}},  # pull "far" right up to the master
    }).json()

    baseline = {jid: w for jid, w in body["baseline"]}
    hypo = {jid: w for jid, w in body["hypothetical"]}
    assert baseline[job_id] == "near"     # low-latency worker wins normally
    assert hypo[job_id] == "far"          # ...until we drop far's latency
    assert body["changed_jobs"] == [job_id]


def test_whatif_bandwidth_override_changes_a_heavy_transfer_placement(client):
    tc, _ = client
    token_fat = _register(tc, "fat")
    token_thin = _register(tc, "thin")
    _heartbeat(tc, "fat", token_fat, latency_ms=10.0, bw=100.0)
    _heartbeat(tc, "thin", token_thin, latency_ms=10.0, bw=5.0)
    r = tc.post("/submit", json={"function": "fn", "priority": 5, "transfer_bytes": 50_000_000})
    job_id = r.json()["job_id"]

    body = tc.post("/scheduler/whatif", json={
        "overrides": {"thin": {"bandwidth_mbps_down": 400.0}},
    }).json()

    assert {jid: w for jid, w in body["baseline"]}[job_id] == "fat"
    assert {jid: w for jid, w in body["hypothetical"]}[job_id] == "thin"


def test_whatif_with_empty_queue_returns_empty_assignments(client):
    tc, _ = client
    token = _register(tc, "a")
    _heartbeat(tc, "a", token, latency_ms=5.0)
    body = tc.post("/scheduler/whatif", json={"overrides": {"a": {"latency_ms": 1.0}}}).json()
    assert body["baseline"] == []
    assert body["hypothetical"] == []
    assert body["changed_jobs"] == []
    assert body["idle_workers"] == ["a"]
