"""TRON_SUBMIT_AUTH_TOKEN gate on /submit + worker token redaction."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def open_client(monkeypatch):
    monkeypatch.delenv("TRON_SUBMIT_AUTH_TOKEN", raising=False)
    import queue_server

    importlib.reload(queue_server)
    return TestClient(queue_server.app)


@pytest.fixture
def guarded_client(monkeypatch):
    monkeypatch.setenv("TRON_SUBMIT_AUTH_TOKEN", "submit-secret")
    import queue_server

    importlib.reload(queue_server)
    return TestClient(queue_server.app)


def test_submit_open_without_token(open_client):
    r = open_client.post("/submit", json={"function": "fn"})
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_submit_guarded_rejects_missing(guarded_client):
    r = guarded_client.post("/submit", json={"function": "fn"})
    assert r.status_code == 401


def test_submit_guarded_accepts_header(guarded_client):
    r = guarded_client.post(
        "/submit",
        json={"function": "fn"},
        headers={"X-TRON-AUTH": "submit-secret"},
    )
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_workers_redacts_auth_token(open_client):
    reg = open_client.post(
        "/register_worker",
        json={"name": "w1", "memory_gb": 4, "gpu": False, "efficiency_runtime": False},
    )
    assert reg.status_code == 200
    assert "auth_token" in reg.json()
    workers = open_client.get("/workers").json()
    assert "w1" in workers
    assert "auth_token" not in workers["w1"]
