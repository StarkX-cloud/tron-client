"""Phase 4: confirm the Grid's static page is actually served by
queue_server.py. The page itself was verified visually and functionally
in a real browser during development (real derived worker positions from
measured latency, correct time-scrub replay of the event log) — that
part isn't something pytest can check, so this test covers what it can:
the server-side wiring that makes the page reachable at all.
"""
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRON_SPINE_DIR", str(tmp_path / "spine"))
    import queue_server
    importlib.reload(queue_server)
    return TestClient(queue_server.app)


def test_grid_index_is_served(client):
    resp = client.get("/grid/")
    assert resp.status_code == 200
    assert "TRON Grid" in resp.text


def test_grid_vendored_three_js_is_served(client):
    resp = client.get("/grid/three.min.js")
    assert resp.status_code == 200
    assert len(resp.content) > 100_000  # sanity: not an empty/truncated file


def test_grid_vendored_orbit_controls_is_served(client):
    resp = client.get("/grid/OrbitControls.js")
    assert resp.status_code == 200
