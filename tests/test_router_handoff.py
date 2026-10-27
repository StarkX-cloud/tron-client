"""Real, live, end-to-end test for tron_router.handoff.submit_to_pool:
submits a real cloudpickle function to a real uvicorn-served
queue_server.py instance, drives one real worker.py job-fetch/execute/
report cycle against it, and reads the real result back through the
MagicFuture. Every other test in this project that touches submission
either mocks the network or uses FastAPI's in-process TestClient; this
one is the only one that runs a real socket server in a background
thread, specifically because submit_to_pool and worker.py both use the
real `requests` library over HTTP, not an ASGI test transport.

This test also exercises the fix made alongside it: worker.py's
execute_job previously discarded every submitted job's "function"
payload entirely (every job queue_server.py accepts via /submit must
carry one -- /submit rejects a submission without it) and returned a
fake echoed "ok" result instead. Nothing submitted through
@tron.remote or tron_router ever actually ran anywhere before this.
"""

from __future__ import annotations

import importlib
import socket
import threading
import time

import pytest
import uvicorn

from tron_router.decision import JobRequest, PlacementDecision
from tron_router.handoff import HandoffError, submit_to_pool


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_master(tmp_path, monkeypatch):
    """A real queue_server.py FastAPI app, served by a real uvicorn
    server on a real local port in a background thread -- not TestClient,
    because submit_to_pool and worker.py both speak real HTTP."""
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
    yield url, queue_server

    server.should_exit = True
    thread.join(timeout=5)


def _register_and_run_one_worker_cycle(master_url, tmp_path, monkeypatch):
    """Registers a real worker against the live master and runs exactly
    one real fetch/execute/report cycle -- not worker.py's infinite
    polling loop, since the test only needs to prove one job round-trips
    for real."""
    monkeypatch.setenv("TRON_MASTER_URL", master_url)
    monkeypatch.setenv("TRON_WORKER_NAME", "handoff-test-worker")
    monkeypatch.setenv("TRON_AUTH_TOKEN_FILE", str(tmp_path / "worker_auth.json"))
    import worker
    importlib.reload(worker)

    auth_token = worker.register_worker()
    job = worker.fetch_next_job(auth_token)
    assert job is not None, "worker never received the submitted job from /next_job"

    start = time.time()
    result = worker.execute_job(job)
    worker.report_complete(auth_token, job["id"], result, time.time() - start)
    return result


def _add(a, b):
    return a + b


def test_submit_to_pool_rejects_a_non_pooled_decision():
    job = JobRequest(model_name="fake/model")
    not_pooled = PlacementDecision(target="local", confidence="high")
    with pytest.raises(HandoffError, match="not 'pooled'"):
        submit_to_pool(job, not_pooled, _add, func_args=(1, 2))


def test_pooled_decision_is_handed_off_and_a_real_worker_executes_it_for_real(
    live_master, tmp_path, monkeypatch
):
    """The full real chain: decide 'pooled' -> submit_to_pool posts a real
    cloudpickle function to a real /submit -> a real worker.py pulls it
    via /next_job and actually calls it (the fix made alongside this
    test) -> MagicFuture.get() reads the real returned value back."""
    master_url, _qs = live_master
    job = JobRequest(model_name="fake/model")
    decision = PlacementDecision(target="pooled", confidence="low")

    future = submit_to_pool(
        job, decision, _add, func_args=(3, 4), master_url=master_url
    )
    assert future.job_id

    worker_result = _register_and_run_one_worker_cycle(master_url, tmp_path, monkeypatch)
    assert worker_result["status"] == "ok"
    assert worker_result["result"] == 7  # the real deserialized _add(3, 4), not an echoed prompt

    assert future.get(timeout=5) == 7


def test_a_failing_remote_function_surfaces_as_a_real_failed_job(
    live_master, tmp_path, monkeypatch
):
    master_url, _qs = live_master
    job = JobRequest(model_name="fake/model")
    decision = PlacementDecision(target="pooled", confidence="low")

    def _boom():
        raise ValueError("deliberate failure for this test")

    future = submit_to_pool(job, decision, _boom, master_url=master_url)

    worker_result = _register_and_run_one_worker_cycle(master_url, tmp_path, monkeypatch)
    assert worker_result["status"] == "error"
    assert "deliberate failure" in worker_result["error"]
