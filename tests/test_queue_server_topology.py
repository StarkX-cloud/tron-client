"""Integration test: does a heartbeat-reported latency actually change
/next_job's placement decision? Unit tests cover TopologyMap and
GlobalDecisionBrain in isolation; this proves the wiring between
/heartbeat, the topology map, and /next_job's scoring loop is real.
"""
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Import fresh each test so module-level state (job_queue, workers,
    # topology, ...) doesn't leak between tests.
    import queue_server
    importlib.reload(queue_server)
    return TestClient(queue_server.app), queue_server


def _register(client, name):
    resp = client.post("/register_worker", json={"name": name})
    assert resp.status_code == 200
    return resp.json()["auth_token"]


def test_heartbeat_with_latency_is_recorded_in_topology(client):
    tc, qs = client
    _register(tc, "worker-a")

    resp = tc.post("/heartbeat/worker-a", json={"worker_name": "worker-a", "latency_ms": 42.0})
    assert resp.status_code == 200

    assert qs.topology.latency("master", "worker-a") == 42.0


def test_negative_latency_is_ignored_not_recorded(client):
    tc, qs = client
    _register(tc, "worker-a")

    tc.post("/heartbeat/worker-a", json={"worker_name": "worker-a", "latency_ms": -5.0})

    assert qs.topology.latency("master", "worker-a") is None


def test_high_latency_worker_is_routed_the_lighter_of_two_queued_jobs(client):
    """/next_job picks the best-scoring job out of ONE worker's own queue
    in a single call — there's no point in this pull-based architecture
    where two different workers' scores for the same job are ever
    compared (see tron_runtime/global_brain.py's decide() docstring for
    why that matters and what it would take to change). So the
    observable, real effect of a worker's measured latency is: given two
    queued jobs of equal priority but different weight, a high-latency
    worker gets routed to the lighter one, because the latency penalty
    scales with job weight.
    """
    tc, qs = client
    _register(tc, "far")
    tc.post("/heartbeat/far", json={"worker_name": "far", "latency_ms": 400.0})

    tc.post("/submit", json={"function": "heavy-fn", "priority": 5, "compute_weight": 20, "memory_gb": 1})
    tc.post("/submit", json={"function": "light-fn", "priority": 5, "compute_weight": 1, "memory_gb": 1})

    job = tc.get("/next_job/far").json()["job"]
    assert job is not None
    assert job["function"] == "light-fn"


def test_match_cycle_solves_the_cross_worker_case_next_job_alone_cannot(client):
    """This is the actual fix for Phase 2a's documented limitation: two
    workers, two jobs of very different weight — the match step (which
    sees both workers and both jobs at once) should place the heavy job
    on the low-latency worker and the light job on the high-latency one,
    which per-worker /next_job calls alone are structurally unable to do
    (see tron_runtime/global_brain.py's decide() docstring).
    """
    tc, qs = client
    _register(tc, "near")
    _register(tc, "far")
    tc.post("/heartbeat/near", json={"worker_name": "near", "latency_ms": 5.0})
    tc.post("/heartbeat/far", json={"worker_name": "far", "latency_ms": 300.0})

    tc.post("/submit", json={"function": "heavy-fn", "priority": 5, "compute_weight": 20})
    tc.post("/submit", json={"function": "light-fn", "priority": 5, "compute_weight": 1})

    qs._run_match_cycle()

    assert qs.pending_assignment["near"]["function"] == "heavy-fn"
    assert qs.pending_assignment["far"]["function"] == "light-fn"

    # Whichever worker polls next gets exactly its matched job, not
    # whatever its own local queue view would have argmax'd to.
    near_job = tc.get("/next_job/near").json()["job"]
    assert near_job["function"] == "heavy-fn"


def test_match_cycle_is_a_noop_with_no_idle_workers(client):
    tc, qs = client
    _register(tc, "worker-1")
    qs.workers["worker-1"]["status"] = "busy"

    tc.post("/submit", json={"function": "fn"})
    qs._run_match_cycle()

    assert qs.pending_assignment == {}


def test_spine_events_recorded_for_submitted_job(client):
    tc, qs = client
    submit_resp = tc.post("/submit", json={"function": "dummy-fn-payload"})
    job_id = submit_resp.json()["job_id"]

    events_resp = tc.get("/spine/events")
    events = events_resp.json()["events"]
    matching = [e for e in events if e["task_id"] == job_id]

    assert len(matching) >= 1
    assert matching[0]["type"] == "queued"
    assert "fn_hash" in matching[0]["data"]
