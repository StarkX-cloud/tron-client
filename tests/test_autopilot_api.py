"""Autopilot HTTP API smoke tests."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    monkeypatch.delenv("TRON_AUTOPILOT_API_TOKEN", raising=False)
    monkeypatch.setenv("TRON_COSTTRACE_PATH", str(tmp_path / "cost.jsonl"))
    import tron_autopilot.api as api

    importlib.reload(api)
    return TestClient(api.app)


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_pricing_skus(api_client):
    r = api_client.get("/pricing/skus")
    assert r.status_code == 200
    assert "g2-gpu-rtx4000a1-s" in r.json()["skus"]


def test_plan_endpoint(api_client):
    r = api_client.post(
        "/plan",
        json={
            "model_name": "EleutherAI/pythia-70m",
            "hourly_rate_usd": 0.5,
            "strategy": "fixed",
            "max_new_tokens": 4,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "chosen" in body
    assert body["chosen"]["target"] in ("local", "pooled", "dedicated", "rejected")


def test_api_token_required(monkeypatch):
    monkeypatch.setenv("TRON_AUTOPILOT_API_TOKEN", "api-secret")
    import tron_autopilot.api as api

    importlib.reload(api)
    client = TestClient(api.app)
    assert client.get("/pricing/skus").status_code == 401
    ok = client.get("/pricing/skus", headers={"X-TRON-AUTH": "api-secret"})
    assert ok.status_code == 200
