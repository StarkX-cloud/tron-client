"""Real, live test proving tron_router.decision.query_pool_status pulls
real measured latency/bandwidth from queue_server.py's /topology
endpoint (added alongside this test) -- not a TestClient mock, since
query_pool_status uses the real `requests` library over real HTTP, same
reasoning as tests/test_router_handoff.py.

Unit-level behavior of decide_placement's use of measured_links (no
network) lives in tests/test_router_decision.py.
"""

from __future__ import annotations

import importlib
import socket
import threading
import time

import pytest
import requests
import uvicorn

from tron_router.decision import query_pool_status


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_master(tmp_path, monkeypatch):
    monkeypatch.setenv("TRON_SPINE_DIR", str(tmp_path / "spine"))
    import queue_server
    importlib.reload(queue_server)

    port = _free_port()
    config = uvicorn.Config(queue_server.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while not getattr(server, "started", False) and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn server never reported started"

    url = f"http://127.0.0.1:{port}"
    yield url

    server.should_exit = True
    thread.join(timeout=5)


def _register(url, name):
    resp = requests.post(f"{url}/register_worker", json={"name": name}, timeout=5)
    resp.raise_for_status()
    return resp.json()["auth_token"]


def _heartbeat(url, name, token, **payload):
    payload.setdefault("worker_name", name)
    return requests.post(
        f"{url}/heartbeat/{name}", json=payload, headers={"X-TRON-AUTH": token}, timeout=5
    )


def test_pool_status_includes_real_measured_latency_for_a_heartbeating_worker(live_master):
    token = _register(live_master, "worker-a")
    resp = _heartbeat(live_master, "worker-a", token, latency_ms=12.5)
    assert resp.status_code == 200

    status = query_pool_status(live_master)

    assert status.reachable
    assert status.alive_worker_count == 1
    assert status.measured_links is not None
    assert status.measured_links["worker-a"]["latency_ms"] == 12.5


def test_pool_status_per_link_latency_is_none_when_never_measured(live_master):
    # Registration alone marks a worker alive (it sets last_heartbeat) --
    # that's a separate real signal from whether its latency/bandwidth
    # has ever actually been measured via a heartbeat payload.
    _register(live_master, "worker-a")

    status = query_pool_status(live_master)

    assert status.reachable
    assert status.alive_worker_count == 1
    assert status.measured_links["worker-a"]["latency_ms"] is None
    assert status.measured_links["worker-a"]["bandwidth_mbps"] is None
