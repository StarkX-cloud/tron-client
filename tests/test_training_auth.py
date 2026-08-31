"""Tests for TRON_TRAINING_AUTH_TOKEN, the access control on
/training/session* and /training/outcomes.

Found and fixed after a real WAN run against a live deployed master (see
writeup/distributed-training.md): that whole path had zero access
control — anyone who found the master's URL could create sessions
(including LoRA ones, which load a real base model) or submit fabricated
"trained" data as any shard. These tests pin the fix: fail-open with no
secret configured (so every other test file and the loopback example are
unaffected), fail-CLOSED once one is.
"""
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def open_client(tmp_path, monkeypatch):
    """No TRON_TRAINING_AUTH_TOKEN set — the local-dev/test default."""
    monkeypatch.setenv("TRON_SPINE_DIR", str(tmp_path / "spine"))
    monkeypatch.delenv("TRON_TRAINING_AUTH_TOKEN", raising=False)
    import queue_server
    importlib.reload(queue_server)
    return TestClient(queue_server.app), queue_server


@pytest.fixture
def guarded_client(tmp_path, monkeypatch):
    """A secret is configured — the real-deployment case."""
    monkeypatch.setenv("TRON_SPINE_DIR", str(tmp_path / "spine"))
    monkeypatch.setenv("TRON_TRAINING_AUTH_TOKEN", "correct-horse-battery-staple")
    import queue_server
    importlib.reload(queue_server)
    return TestClient(queue_server.app), queue_server


def _create_session(tc, headers=None):
    return tc.post(
        "/training/session",
        json={"session_id": "sess-auth", "num_shards": 2, "num_rounds": 2, "local_steps": 3},
        headers=headers or {},
    )


def test_no_secret_configured_every_endpoint_is_open(open_client):
    tc, _ = open_client
    resp = _create_session(tc)
    assert resp.status_code == 200

    assert tc.get("/training/session/sess-auth").status_code == 200
    assert tc.get("/training/session/sess-auth/shard/0/data").status_code == 200
    assert tc.get("/training/outcomes").status_code == 200


def test_secret_configured_create_session_without_header_is_401(guarded_client):
    tc, _ = guarded_client
    resp = _create_session(tc)
    assert resp.status_code == 401


def test_secret_configured_wrong_token_is_401(guarded_client):
    tc, _ = guarded_client
    resp = _create_session(tc, headers={"X-TRON-AUTH": "not-the-secret"})
    assert resp.status_code == 401


def test_secret_configured_plain_http_is_rejected_even_with_correct_token(guarded_client):
    """The token protects nothing if it can be read off the wire in
    cleartext. X-Forwarded-Proto: http is what a TLS-terminating proxy
    (Render's edge, or any reverse proxy) sets to say the original
    request wasn't HTTPS."""
    tc, _ = guarded_client
    resp = _create_session(
        tc, headers={"X-TRON-AUTH": "correct-horse-battery-staple", "X-Forwarded-Proto": "http"}
    )
    assert resp.status_code == 400


def test_secret_configured_no_forwarded_proto_header_behaves_as_before(guarded_client):
    """Local dev / loopback / TestClient requests carry no
    X-Forwarded-Proto at all — must not be treated as "http"."""
    tc, _ = guarded_client
    resp = _create_session(tc, headers={"X-TRON-AUTH": "correct-horse-battery-staple"})
    assert resp.status_code == 200


def test_secret_configured_forwarded_proto_https_is_fine(guarded_client):
    tc, _ = guarded_client
    resp = _create_session(
        tc, headers={"X-TRON-AUTH": "correct-horse-battery-staple", "X-Forwarded-Proto": "https"}
    )
    assert resp.status_code == 200


def test_secret_configured_correct_token_succeeds(guarded_client):
    tc, _ = guarded_client
    headers = {"X-TRON-AUTH": "correct-horse-battery-staple"}
    resp = _create_session(tc, headers=headers)
    assert resp.status_code == 200

    assert tc.get("/training/session/sess-auth", headers=headers).status_code == 200
    assert tc.get("/training/session/sess-auth/shard/0/data", headers=headers).status_code == 200
    assert tc.get("/training/outcomes", headers=headers).status_code == 200


def test_secret_configured_every_route_individually_rejects_no_header(guarded_client):
    """Not just session creation — every route in the family, including
    the one an attacker would actually want (submitting fabricated
    "trained" data as some other shard)."""
    tc, _ = guarded_client
    headers = {"X-TRON-AUTH": "correct-horse-battery-staple"}
    _create_session(tc, headers=headers)

    assert tc.get("/training/session/sess-auth").status_code == 401
    assert tc.get("/training/session/sess-auth/shard/0/data").status_code == 401
    assert tc.get("/training/session/sess-auth/round/0/init", params={"shard": 0}).status_code == 401
    assert tc.post("/training/session/sess-auth/round/0/shard/0", content=b"\x00").status_code == 401
    assert tc.get("/training/session/sess-auth/result").status_code == 401
    assert tc.get("/training/outcomes").status_code == 401


def test_full_run_through_shard_client_with_auth_token(guarded_client):
    """End-to-end proof: RequestsTransport-shaped auth plumbing (the code
    path real shard processes use) actually gets a run past a guarded
    master, via the same TestClient-backed transport test_distributed_
    training.py uses."""
    import threading

    from tron.training.distributed.shard_client import run_shard

    tc, qs = guarded_client
    token = "correct-horse-battery-staple"
    headers = {"X-TRON-AUTH": token}
    resp = _create_session(tc, headers=headers)
    assert resp.status_code == 200

    class _AuthedTransport:
        def __init__(self, tc, headers):
            self._tc, self._headers = tc, headers

        def get_bytes(self, path):
            r = self._tc.get(path, headers=self._headers)
            if r.status_code == 202:
                return None
            r.raise_for_status()
            return r.content

        def get_json(self, path):
            r = self._tc.get(path, headers=self._headers)
            r.raise_for_status()
            return r.json()

        def post_bytes(self, path, blob):
            r = self._tc.post(path, content=blob, headers=self._headers)
            r.raise_for_status()
            return r.json()

    transport = _AuthedTransport(tc, headers)
    errors = []

    def drive(k):
        try:
            run_shard(transport, "sess-auth", k, poll_interval=0.02, max_wait_seconds=60)
        except Exception as exc:  # noqa: BLE001
            errors.append((k, repr(exc)))

    threads = [threading.Thread(target=drive, args=(k,)) for k in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors

    result = tc.get("/training/session/sess-auth/result", headers=headers)
    assert result.status_code == 200
    assert result.json()["rounds_merged"] == [0, 1]
