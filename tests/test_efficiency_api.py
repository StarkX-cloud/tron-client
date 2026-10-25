"""API tests with a stub workload runner — fast, deterministic, no model load.

``run_workload`` is monkeypatched at the ``tron_efficiency.api.app`` module
level (where the route looks it up), per the real signature
``run_workload(config, policy) -> RunOutcome``.
"""

import os

import pytest
from fastapi.testclient import TestClient

from tron_efficiency.workloads.runner import RunOutcome


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRON_EFFICIENCY_RESULTS_DIR", str(tmp_path))
    import tron_efficiency.api.app as app_module

    def fake_run_workload(config, policy):
        return RunOutcome(
            success=True,
            policy_name=policy.name,
            load_time_seconds=0.01,
            time_to_first_token_seconds=0.001,
            end_to_end_latency_seconds=0.02,
            tokens_generated=config.max_new_tokens,
            tokens_per_second=config.max_new_tokens / 0.02,
            peak_cpu_memory_bytes=123456,
            generated_text="stubbed output",
        )

    monkeypatch.setattr(app_module, "run_workload", fake_run_workload)
    return TestClient(app_module.app)


def test_get_hardware(client):
    res = client.get("/hardware")
    assert res.status_code == 200
    body = res.json()
    assert "cuda_available" in body
    assert "platform" in body


def test_create_and_list_run(client):
    res = client.post("/runs", json={"model": "fake/model", "policy": "baseline"})
    assert res.status_code == 200
    body = res.json()
    assert body["outcome"]["success"] is True
    assert body["outcome"]["generated_text"] == "stubbed output"

    res = client.get("/runs")
    assert res.status_code == 200
    assert len(res.json()) == 1


def test_get_run_not_found(client):
    res = client.get("/runs/does-not-exist")
    assert res.status_code == 404


def test_offload_policy_request(client):
    res = client.post(
        "/runs",
        json={"model": "fake/model", "policy": "offload", "memory_budget_mb": 128},
    )
    assert res.status_code == 200
    assert res.json()["policy"]["memory_budget_bytes"] == 128 * 1024 * 1024


def test_unknown_policy_rejected(client):
    res = client.post("/runs", json={"model": "fake/model", "policy": "nonsense"})
    assert res.status_code == 400


def test_export_csv(client):
    create = client.post("/runs", json={"model": "fake/model", "policy": "baseline"})
    run_id = create.json()["run_id"]
    res = client.get(f"/runs/{run_id}/export.csv")
    assert res.status_code == 200
    assert "run_id" in res.text


def test_report_endpoint(client):
    client.post("/runs", json={"model": "fake/model", "policy": "baseline"})
    res = client.get("/report")
    assert res.status_code == 200
    assert "Benchmark report" in res.text


def test_recommend_endpoint(client, monkeypatch):
    import tron_efficiency.api.app as app_module
    from tron_efficiency.advisor.recommend import Recommendation

    def fake_recommend(model_name, hardware=None, gpu_budget_bytes_override=None):
        assert model_name == "fake/model"
        assert gpu_budget_bytes_override == 512 * 1024 * 1024
        return Recommendation(
            model_name=model_name, estimated_param_count=1000, param_count_is_estimate=True,
            device="cpu", dtype="fp32", policy="baseline", memory_budget_bytes=None,
            confidence="medium", rationale=["stub"], warnings=[],
        )

    monkeypatch.setattr(app_module, "_recommend", fake_recommend)
    res = client.get("/recommend", params={"model": "fake/model", "gpu_budget_mb": 512})
    assert res.status_code == 200
    body = res.json()
    assert body["device"] == "cpu"
    assert body["confidence"] == "medium"
