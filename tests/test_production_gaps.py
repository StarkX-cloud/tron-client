"""Closing Prompt 4 production gaps: auth, efficiency filter, quarantine, SLA."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from tron.spine import TopologyMap, score_pair
from tron.spine.matcher import INFEASIBLE_SCORE
from tron_pilot.sla import sla_report
from tron_pilot.usage import UsageEvent, UsageLedger


@pytest.fixture
def qs_client(monkeypatch, tmp_path):
    monkeypatch.delenv("TRON_SUBMIT_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TRON_REQUIRE_VERIFIED_PROVIDERS", raising=False)
    monkeypatch.setenv("TRON_PROVIDER_STATS", str(tmp_path / "stats.jsonl"))
    import queue_server

    importlib.reload(queue_server)
    return TestClient(queue_server.app), queue_server


def test_next_job_requires_worker_auth(qs_client):
    client, _ = qs_client
    reg = client.post(
        "/register_worker",
        json={"name": "auth-w", "memory_gb": 4, "efficiency_runtime": False},
    )
    token = reg.json()["auth_token"]
    bad = client.get("/next_job/auth-w")
    assert bad.status_code == 403
    ok = client.get("/next_job/auth-w", headers={"X-TRON-AUTH": token})
    assert ok.status_code == 200
    assert ok.json().get("job") is None


def test_complete_requires_worker_auth(qs_client):
    client, qs = qs_client
    reg = client.post(
        "/register_worker",
        json={"name": "cw", "memory_gb": 8, "efficiency_runtime": True},
    )
    token = reg.json()["auth_token"]
    # Inject a running job directly
    with qs.lock:
        qs.job_store["j1"] = {"id": "j1", "status": "running"}
        qs.running_jobs["j1"] = {
            "worker": "cw",
            "start_time": 0,
            "job": {"id": "j1", "memory_gb": 1, "requeue_count": 0},
        }
        qs.workers["cw"]["status"] = "busy"
        qs.workers["cw"]["active_jobs"] = 1
    bad = client.post("/complete/j1", json={"result": {"ok": True}})
    assert bad.status_code == 403
    good = client.post(
        "/complete/j1",
        json={"result": {"ok": True}},
        headers={"X-TRON-AUTH": token},
    )
    assert good.status_code == 200
    assert good.json().get("worker") == "cw"
    assert qs.job_store["j1"]["worker"] == "cw"


def test_status_gated_when_submit_token_set(monkeypatch, tmp_path):
    monkeypatch.setenv("TRON_SUBMIT_AUTH_TOKEN", "sec")
    monkeypatch.setenv("TRON_PROVIDER_STATS", str(tmp_path / "s.jsonl"))
    import queue_server

    importlib.reload(queue_server)
    client = TestClient(queue_server.app)
    assert client.get("/status/missing").status_code == 401
    r = client.get("/status/missing", headers={"X-TRON-AUTH": "sec"})
    assert r.status_code == 200
    assert r.json()["status"] == "not_found"


def test_matcher_efficiency_runtime_hard_filter():
    topo = TopologyMap()
    job = {"requires_efficiency_runtime": True, "priority": 1}
    assert (
        score_pair(job, "w", {"efficiency_runtime": False, "gpu": False}, topo)
        == INFEASIBLE_SCORE
    )
    assert (
        score_pair(job, "w", {"efficiency_runtime": True, "gpu": False}, topo)
        != INFEASIBLE_SCORE
    )


def test_matcher_quarantine_and_region():
    topo = TopologyMap()
    assert (
        score_pair({"priority": 1}, "w", {"quarantined": True}, topo) == INFEASIBLE_SCORE
    )
    soft = score_pair(
        {"priority": 1, "preferred_location": "us-east"},
        "w",
        {"location": "us-east"},
        topo,
    )
    other = score_pair(
        {"priority": 1, "preferred_location": "us-east"},
        "w",
        {"location": "eu-west"},
        topo,
    )
    assert soft > other
    assert (
        score_pair(
            {
                "priority": 1,
                "preferred_location": "us-east",
                "preferred_location_strict": True,
            },
            "w",
            {"location": "eu-west"},
            topo,
        )
        == INFEASIBLE_SCORE
    )


def test_quarantine_endpoint(qs_client):
    client, qs = qs_client
    client.post("/register_worker", json={"name": "q1", "memory_gb": 4})
    r = client.post("/workers/q1/quarantine", json={"reason": "flaky"})
    assert r.json()["quarantined"] is True
    assert qs.workers["q1"]["quarantined"] is True
    client.post("/workers/q1/clear_quarantine")
    assert qs.workers["q1"]["quarantined"] is False


def test_sla_report(tmp_path):
    ledger = UsageLedger(tmp_path / "u.jsonl")
    for rt, ok in [(1.0, True), (2.0, True), (10.0, True), (0.5, False)]:
        ledger.record(
            UsageEvent(
                customer_id="c",
                event_type="workload_complete" if ok else "workload_failed",
                plan_id=None,
                model_name="m",
                target="local",
                success=ok,
                runtime_seconds=rt,
                cost_usd=0.01 if ok else None,
            )
        )
    report = sla_report(ledger, customer_id="c", deadline_seconds=5.0)
    assert report["events"] == 4
    assert report["successes"] == 3
    assert report["availability"] == 0.75
    assert report["deadline_misses"] == 1
    assert report["runtime_p50_seconds"] is not None
