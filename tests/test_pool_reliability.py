"""Real, live test of a gap this project's docs have repeatedly flagged
as open: "no retry/reliability handling if a pooled worker drops
mid-job." queue_server.py actually already has a watchdog
(_watchdog_loop) that requeues a dead worker's orphaned jobs via the
spine's content-addressed event log (tron/spine/log.py's
recover_orphaned_tasks) -- but nothing in this repo had ever proven,
for real, that a job submitted through tron_router.handoff.submit_to_pool
survives its worker dying mid-job and still completes on a second
worker, with the original MagicFuture still resolving correctly. This
test drives that whole path for real: a real uvicorn master, a real
dead worker (registered, claims the job, then simply stops -- never
heartbeats or completes it), the real watchdog thread detecting it and
requeuing, and a second real worker picking the requeued job up and
actually running the real deserialized function.
"""

from __future__ import annotations

import importlib
import socket
import threading
import time

import pytest
import uvicorn

from tron_router.decision import JobRequest, PlacementDecision
from tron_router.handoff import submit_to_pool


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_master_with_watchdog(tmp_path, monkeypatch):
    """Same real-uvicorn-in-a-thread setup as test_router_handoff.py,
    plus the watchdog thread -- queue_server.py only starts that itself
    under `if __name__ == "__main__"`, which importing the module for
    ASGI serving never triggers. Also shrinks HEARTBEAT_TIMEOUT_SECONDS
    so the test doesn't have to wait the real 20s default to see a
    worker declared dead."""
    monkeypatch.setenv("TRON_SPINE_DIR", str(tmp_path / "spine"))
    import queue_server
    importlib.reload(queue_server)
    queue_server.HEARTBEAT_TIMEOUT_SECONDS = 1.0

    port = _free_port()
    config = uvicorn.Config(queue_server.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    deadline = time.time() + 10
    while not getattr(server, "started", False) and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn server never reported started"

    watchdog_thread = threading.Thread(target=queue_server._watchdog_loop, daemon=True)
    watchdog_thread.start()

    url = f"http://127.0.0.1:{port}"
    yield url, queue_server

    server.should_exit = True
    server_thread.join(timeout=5)


def _new_worker(master_url, tmp_path, monkeypatch, name):
    """A fresh worker module instance registered under its own name --
    reload is necessary because TRON_WORKER_NAME/AUTH_TOKEN_FILE are
    read into module-level constants at import time."""
    monkeypatch.setenv("TRON_MASTER_URL", master_url)
    monkeypatch.setenv("TRON_WORKER_NAME", name)
    monkeypatch.setenv("TRON_AUTH_TOKEN_FILE", str(tmp_path / f"{name}_auth.json"))
    import worker
    importlib.reload(worker)
    auth_token = worker.register_worker()
    return worker, auth_token


def _multiply(a, b):
    return a * b


def test_a_dead_workers_job_is_requeued_and_finished_by_another_real_worker(
    live_master_with_watchdog, tmp_path, monkeypatch
):
    master_url, qs = live_master_with_watchdog
    job = JobRequest(model_name="fake/model")
    decision = PlacementDecision(target="pooled", confidence="low")

    future = submit_to_pool(job, decision, _multiply, func_args=(6, 7), master_url=master_url)
    job_id = future.job_id

    # worker-doomed claims the job via a real /next_job call, then
    # simply disappears -- never heartbeats again, never calls /complete.
    # This is exactly "a pooled worker drops mid-job," not a simulation
    # of it: real registration, real claim, real silence.
    worker_doomed, token_doomed = _new_worker(master_url, tmp_path, monkeypatch, "worker-doomed")
    claimed = worker_doomed.fetch_next_job(token_doomed)
    assert claimed is not None and claimed["id"] == job_id

    with qs.lock:
        assert qs.workers["worker-doomed"]["status"] == "busy"
        assert job_id in qs.running_jobs

    # Wait past HEARTBEAT_TIMEOUT_SECONDS (shrunk to 1.0s) plus the
    # watchdog's own 5s poll interval for it to actually notice and act.
    deadline = time.time() + 15
    while time.time() < deadline:
        with qs.lock:
            back_in_queue = any(j["id"] == job_id for j in qs.job_queue)
        if back_in_queue:
            break
        time.sleep(0.25)
    assert back_in_queue, "watchdog never requeued the dead worker's job"

    with qs.lock:
        assert qs.job_store[job_id]["status"] == "queued"
        assert job_id not in qs.running_jobs
        assert qs.workers["worker-doomed"]["status"] == "offline"

    # A second, live worker picks up the SAME job (same job_id -- the
    # original future is still tracking it) and actually runs it for real.
    worker_rescuer, token_rescuer = _new_worker(master_url, tmp_path, monkeypatch, "worker-rescuer")
    rescued_job = worker_rescuer.fetch_next_job(token_rescuer)
    assert rescued_job is not None and rescued_job["id"] == job_id
    assert rescued_job.get("function") is not None  # the real payload survived the requeue

    start = time.time()
    result = worker_rescuer.execute_job(rescued_job)
    worker_rescuer.report_complete(token_rescuer, job_id, result, time.time() - start)

    assert result["status"] == "ok"
    assert result["result"] == 42  # the real deserialized _multiply(6, 7), survived the requeue

    assert future.get(timeout=5) == 42
