import importlib

from fastapi.testclient import TestClient


import tron.gpu.openai_bridge as openai_bridge


client = TestClient(openai_bridge.app)


def test_chat_completions_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("TRON_API_KEY", "demo-key")
    monkeypatch.setattr(
        openai_bridge.proxy,
        "submit_job_and_wait",
        lambda *args, **kwargs: {
            "status": "completed",
            "result": {"worker": {"result": {"content": "hello from tron"}}},
        },
    )

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid API key"


def test_chat_completions_returns_openai_payload(monkeypatch):
    monkeypatch.delenv("TRON_API_KEY", raising=False)
    monkeypatch.setattr(
        openai_bridge.proxy,
        "submit_job_and_wait",
        lambda *args, **kwargs: {
            "status": "completed",
            "result": {"worker": {"result": {"content": "hello from tron"}}},
        },
    )

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"
    assert response.json()["choices"][0]["message"]["content"] == "hello from tron"
