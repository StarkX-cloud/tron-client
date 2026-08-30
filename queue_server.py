"""TRON queue server: worker registration, job scheduling, and the Phase 1
execution spine (content-addressed artifacts + replayable event log).

This file used to also carry a billing/royalty/customer-account layer for a
compute-rental marketplace. That's been cut — see ARCHITECTURE.md for why.
What's left is the actual product: get work onto heterogeneous nodes
reliably, and record everything that happens in a form that can be replayed
(for recovery, for debugging, eventually for the 3D Grid).
"""
from __future__ import annotations

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    _USE_FASTAPI = True
except Exception as import_error:
    # FastAPI or Uvicorn failed to import — provide a minimal fallback HTTP server
    # so the SDK's auto-discovery still has something to talk to.
    _USE_FASTAPI = False
    import traceback
    print(f"[TRON STARTUP] FastAPI import failed: {import_error}")
    traceback.print_exc()
    import http.server
    import socketserver

import json
import os
import time
import uuid
import threading
from pathlib import Path
import asyncio
from collections import defaultdict

# =========================
# ENGINE IMPORTS
# =========================
# Of the original tron_runtime grab-bag, only these three are actually
# used: SessionManager backs /create_session, SwarmManager's scale signal
# and LoadShaper feed /next_job, StreamEngine backs /stream. The other
# eight modules (routing/predictor/auto_scaler/resurrection/memory_mesh/
# simulation/pricing/market engines) were instantiated but never called —
# decoration, not behavior — and have been removed along with their
# imports. GlobalDecisionBrain is no longer one of the stubs: Phase 2
# rewrote it to score placement from measured topology (see
# tron/spine/topology.py) instead of returning the job's priority
# unchanged.

from tron_runtime.session_manager import SessionManager
from tron_runtime.swarm_manager import SwarmManager
from tron_runtime.stream_engine import StreamEngine
from tron_runtime.load_shaper import LoadShaper
from tron_runtime.global_brain import GlobalDecisionBrain

# =========================
# EXECUTION SPINE (Phase 1 + Phase 2)
# =========================
from tron.spine import EventLog, ArtifactStore, Task, content_hash, TopologyMap, match_jobs_to_workers

# Phase 3 -> Phase 1 wiring: lets /training/run_demo record a real
# training run into this server's own spine log, so /grid/ can render it.
from tron.training.data import make_classification_dataset, make_non_iid_shards, train_test_split
from tron.training.model import TinyMLP
from tron.training.spine_integration import run_local_sgd_with_spine

# =========================
# ORCHESTRATOR & GPU IMPORTS
# =========================
try:
    from tron.orchestrator import TrainingOrchestrator, TrainingConfig
    HAS_ORCHESTRATOR = True
except ImportError:
    HAS_ORCHESTRATOR = False

try:
    from tron.gpu import VirtualGPUCluster
    HAS_VGPU = True
except ImportError:
    HAS_VGPU = False

# =========================
# APP
# =========================

class _FallbackApp:
    def add_middleware(self, *args, **kwargs):
        return None

    def get(self, *args, **kwargs):
        return self._noop

    def post(self, *args, **kwargs):
        return self._noop

    def _noop(self, func):
        return func

app = FastAPI() if _USE_FASTAPI else _FallbackApp()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phase 4: the 3D Grid — a passive, time-scrubbable replay of the
# execution spine's event log, served at /grid. It's a static page that
# fetches /workers and /spine/events itself; see tron/grid/index.html.
if _USE_FASTAPI:
    _grid_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tron", "grid")
    if os.path.isdir(_grid_dir):
        app.mount("/grid", StaticFiles(directory=_grid_dir, html=True), name="grid")

# =========================
# STATE INIT
# =========================

sessions = SessionManager()

swarm = SwarmManager()
stream_engine = StreamEngine()
load_shaper = LoadShaper()

# Execution spine: one process-wide event log, artifact store, and
# topology map (measured worker latency, feeding placement decisions).
# TRON_SPINE_DIR lets tests (and operators who want spine data somewhere
# other than the cwd) point this at an isolated directory — without it,
# reloading this module (as the test suite's importlib.reload does to
# reset in-memory state between tests) does NOT reset the on-disk event
# log or artifact store, since EventLog/ArtifactStore open and continue
# an existing file rather than truncate it. That's correct behavior for
# a real server restart (the whole point of a durable log is surviving
# one) — it's specifically a test-isolation footgun otherwise.
_SPINE_DIR = Path(os.environ.get("TRON_SPINE_DIR", ".tron_spine"))
event_log = EventLog(path=_SPINE_DIR / "log.db")
artifact_store = ArtifactStore(root=_SPINE_DIR / "artifacts")
topology = TopologyMap()

global_brain = GlobalDecisionBrain(topology, swarm, load_shaper)

# Initialize integrated orchestrator & vGPU cluster
orchestrator = None
vgpu_cluster = None

if HAS_ORCHESTRATOR:
    try:
        orchestrator = TrainingOrchestrator()
        print("[TRON] ✓ TrainingOrchestrator initialized")
    except Exception as e:
        print(f"[TRON] Warning: Could not initialize TrainingOrchestrator: {e}")

if HAS_VGPU:
    try:
        vgpu_cluster = VirtualGPUCluster(cluster_name="tron-platform-0")
        print("[TRON] ✓ VirtualGPUCluster initialized")
    except Exception as e:
        print(f"[TRON] Warning: Could not initialize VirtualGPUCluster: {e}")

# =========================
# STATE MEMORY
# =========================

lock = threading.Lock()

job_queue = []
job_store = {}
workers = {}
running_jobs = {}

# Phase 2b: jobs the periodic match step (see _match_loop) has already
# decided a specific worker should get, ahead of that worker's next poll.
# This is what makes cross-worker arbitration real — see
# tron/spine/matcher.py for why /next_job's own per-worker loop alone
# can't do this.
pending_assignment = {}

event_bus = defaultdict(list)

HEARTBEAT_TIMEOUT_SECONDS = 20.0
MATCH_INTERVAL_SECONDS = 1.0

# =========================
# EMIT
# =========================

def emit(job_id, event_type, data=None):
    """Emit to the legacy in-memory SSE bus (for existing /stream clients)
    AND append to the durable, replayable spine log — the latter is the
    one that survives a restart and is what recovery/replay reads from.
    """
    event = {
        "job_id": job_id,
        "type": event_type,
        "data": data or {},
        "time": time.time()
    }

    stream_engine.emit(job_id, event_type, data)
    event_log.append(job_id, event_type, data or {})

    with lock:
        event_bus[job_id].append(event)

# =========================
# HOME
# =========================

@app.get("/")
def home():
    return {"status": "TRON_CORE_ONLINE"}

@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# SESSIONS
# =========================

@app.post("/create_session")
def create_session():
    session_id = sessions.create_session()
    return {"session_id": session_id}

@app.get("/session/{session_id}")
def get_session(session_id: str):
    return sessions.get(session_id) or {"status": "not_found"}

# =========================
# WORKERS
# =========================

@app.post("/register_worker")
def register_worker(worker: dict):

    worker_name = worker.get("name") or str(uuid.uuid4().hex[:8])
    auth_token = str(uuid.uuid4().hex)

    with lock:
        workers[worker_name] = {
            "name": worker_name,
            "auth_token": auth_token,
            "gpu": worker.get("gpu") or worker.get("capabilities", {}).get("gpu", False),
            "gpu_name": worker.get("gpu_name"),
            "memory_gb": worker.get("memory_gb") or worker.get("capabilities", {}).get("memory_gb", 4),
            "cuda_cores": worker.get("cuda_cores") or worker.get("capabilities", {}).get("cuda_cores", 1024),
            "location": worker.get("location", "unknown"),
            "load": 0,
            "status": "idle",
            "last_heartbeat": time.time()
        }

    return {
        "ok": True,
        "worker_name": worker_name,
        "auth_token": auth_token
    }


@app.post("/register")
def register(worker: dict):
    """Compatibility alias for older worker bootstraps."""
    return register_worker(worker)


@app.post("/heartbeat/{worker_name}")
def heartbeat(worker_name: str, request: Request, payload: dict = None):
    """Worker heartbeat with optional auth validation.

    If the worker reports `latency_ms` (its own measured round-trip time
    of the *previous* heartbeat call — see worker.py), record it in the
    topology map. This is the one real signal Phase 2's placement scoring
    reads; see tron/spine/topology.py and tron_runtime/global_brain.py.
    """

    auth_token = request.headers.get("X-TRON-AUTH")

    with lock:
        if worker_name not in workers:
            return {"alive": False, "error": "worker not registered"}, 404

        worker = workers[worker_name]

        if worker.get("auth_token") and auth_token:
            if auth_token != worker["auth_token"]:
                return {"alive": False, "error": "invalid auth token"}, 403

        worker["last_heartbeat"] = time.time()
        if payload:
            worker["active_job_id"] = payload.get("active_job_id")
            latency_ms = payload.get("latency_ms")
            if isinstance(latency_ms, (int, float)) and latency_ms >= 0:
                worker["latency_ms"] = latency_ms

    if payload:
        latency_ms = payload.get("latency_ms")
        if isinstance(latency_ms, (int, float)) and latency_ms >= 0:
            topology.record_latency("master", worker_name, float(latency_ms))

    return {"alive": True, "worker_name": worker_name}


@app.post("/heartbeat")
def heartbeat_alias(payload: dict):
    worker_name = payload.get("worker_name")
    if not worker_name:
        return {"alive": False, "error": "missing worker_name"}
    return heartbeat(worker_name)

# =========================
# SUBMIT JOB
# =========================

@app.post("/submit")
def submit(job: dict):

    job_id = str(uuid.uuid4())

    raw_job = {
        "id": job_id,
        "task_type": job.get("task_type", "remote"),
        "prompt": job.get("prompt", ""),
        "priority": int(job.get("priority", 1)),
        "gpu": bool(job.get("gpu", job.get("gpu_required", False))),
        "memory_gb": float(job.get("memory_gb", job.get("min_memory_gb", 1))),
        "submitted_at": time.time(),
        "function": job.get("function"),
        "compute_weight": job.get("compute_weight", 1)
    }

    if raw_job["function"] is None:
        return {"error": "Missing function payload"}

    enriched_job = raw_job.copy()

    # Record the task's function payload as a content-addressed artifact so
    # identical work is recognized as identical, and lost work can be
    # re-derived from its recorded lineage rather than requiring a checkpoint.
    fn_bytes = raw_job["function"].encode("utf-8") if isinstance(raw_job["function"], str) else bytes(raw_job["function"])
    fn_artifact = artifact_store.put(fn_bytes)
    enriched_job["fn_hash"] = fn_artifact.artifact_id

    with lock:
        job_queue.append(enriched_job)

        job_store[job_id] = {
            "id": job_id,
            "status": "queued",
            "submitted_at": time.time(),
            "fn_hash": fn_artifact.artifact_id,
        }

    emit(job_id, "queued", {"fn_hash": fn_artifact.artifact_id, "input_hashes": []})

    return {
        "job_id": job_id,
        "status": "queued"
    }

# =========================
# METRICS
# =========================

@app.get("/metrics")
def metrics():
    return {
        "workers": len(workers),
        "queue_size": len(job_queue),
        "running_jobs": len(running_jobs),
        "completed_jobs": sum(1 for job in job_store.values() if job.get("status") == "completed")
    }


@app.get("/workers")
def get_workers():
    return workers


@app.get("/running")
def get_running():
    return running_jobs


@app.get("/queue")
def get_queue():
    return {"queue": job_queue}


@app.get("/history")
def get_history():
    return {"jobs": list(job_store.values())}


@app.get("/active_jobs")
def get_active_jobs():
    return {"active_jobs": list(running_jobs.values())}

# =========================
# SPINE: replay & recovery
# =========================

@app.get("/spine/events")
def spine_events(since_seq: int = 0):
    """Raw event log tail — this is what the future 3D Grid renders."""
    events = list(event_log.replay(since_seq=since_seq))
    return {
        "events": [
            {
                "seq": e.seq,
                "task_id": e.task_id,
                "type": e.type,
                "data": e.data,
                "timestamp": e.timestamp,
                "node_id": e.node_id,
            }
            for e in events
        ]
    }


@app.get("/spine/task/{job_id}")
def spine_task(job_id: str):
    """Full recorded history for one task."""
    events = event_log.events_for(job_id)
    if not events:
        return {"status": "not_found"}
    return {
        "task_id": job_id,
        "events": [
            {"seq": e.seq, "type": e.type, "data": e.data, "timestamp": e.timestamp, "node_id": e.node_id}
            for e in events
        ],
    }


@app.post("/training/run_demo")
def run_training_demo(payload: dict = None):
    """Run the Phase 3 local-SGD demo, recording it into this server's
    own execution spine (see tron/training/spine_integration.py) so
    /grid/ can render a real training run instead of only generic job
    lifecycle. Synchronous — the default size finishes in well under a
    second (see tron/training/benchmark.py for the same problem run
    standalone with the fuller comparison against a sync-every-step
    baseline and weight merging).
    """
    payload = payload or {}
    num_shards = int(payload.get("num_shards", 4))
    num_rounds = int(payload.get("num_rounds", 10))
    local_steps = int(payload.get("local_steps", 10))

    num_features, num_classes = 8, 4
    x_all, y_all = make_classification_dataset(
        num_samples=2000, num_features=num_features, num_classes=num_classes, seed=0, class_sep=1.0,
    )
    x_train, y_train, x_test, y_test = train_test_split(x_all, y_all, test_fraction=0.2)
    shards = make_non_iid_shards(x_train, y_train, num_shards=num_shards, num_classes=num_classes, skew=0.9, seed=1)

    def factory():
        return TinyMLP(input_dim=num_features, hidden_dim=16, num_classes=num_classes, seed=42)

    with lock:
        for i in range(num_shards):
            name = f"shard-{i}"
            if name not in workers:
                workers[name] = {
                    "name": name, "auth_token": None, "gpu": False, "gpu_name": None,
                    "memory_gb": 4, "cuda_cores": 1024, "location": "in-process-training-demo",
                    "load": 0, "status": "training", "last_heartbeat": time.time(),
                }

    result = run_local_sgd_with_spine(
        shards, factory, num_rounds=num_rounds, local_steps=local_steps, lr=0.3,
        event_log=event_log, artifact_store=artifact_store,
    )

    with lock:
        for name in result["shard_node_ids"]:
            if name in workers:
                # Not "idle" — these are synthetic in-process shard nodes,
                # not real pollable workers, so they shouldn't look like
                # assignable capacity to the Phase 2b match step.
                workers[name]["status"] = "done"

    return {
        "accuracy": result["model"].accuracy(x_test, y_test),
        "comm_bytes": result["comm_bytes"],
        "num_syncs": result["num_syncs"],
        "shard_node_ids": result["shard_node_ids"],
    }

# =========================
# NEXT JOB (scheduling)
# =========================

def _claim_job(worker_name: str, job: dict) -> str:
    """Shared bookkeeping once a specific job has been chosen for a
    specific worker — caller must already hold `lock`. Returns job_id."""
    job_id = job["id"]
    job_store[job_id]["status"] = "running"
    workers[worker_name]["status"] = "busy"
    workers[worker_name]["load"] += job.get("memory_gb", 1)
    running_jobs[job_id] = {
        "worker": worker_name,
        "start_time": time.time(),
        "job": job,
    }
    return job_id


@app.get("/next_job/{worker_name}")
def next_job(worker_name: str):

    claimed_job = None
    scale_state = None
    best_score = None

    with lock:

        if worker_name not in workers:
            return {"job": None}

        # Phase 2b: if the periodic match step (tron/spine/matcher.py)
        # already decided this worker should get a specific job — having
        # considered it against every other idle worker and every queued
        # job at once — honor that instead of re-deciding locally. This
        # is the actual fix for the limitation Phase 2a documented:
        # per-worker argmax alone can never compare two workers' scores
        # for the same job, because each worker's /next_job call only
        # ever sees its own queue view.
        matched = pending_assignment.pop(worker_name, None)
        if matched is not None:
            claimed_job = matched
            job_id = _claim_job(worker_name, claimed_job)
        else:
            if not job_queue:
                return {"job": None}

            worker = workers[worker_name]
            scale_state = swarm.should_scale(len(job_queue))
            shaped = load_shaper.reshape(job_queue, workers)

            best_job = None
            best_index = None
            best_score = -999

            for i, entry in enumerate(shaped):

                job = entry["job"]

                if entry.get("delay") == 1:
                    continue

                try:
                    decision = global_brain.decide(job_queue, workers, job, worker_name=worker_name)
                    score = decision.get("score", 0)
                except Exception:
                    score = job.get("priority", 1)

                if scale_state == "SCALE_UP":
                    score += 2
                elif scale_state == "SCALE_DOWN":
                    score -= 0.5

                gpu_bonus = 1.5 if worker.get("gpu") else 1.0
                memory_factor = max(0.5, 1.0 - worker.get("load", 0) * 0.1)

                score *= gpu_bonus * memory_factor
                score -= job.get("memory_gb", 1) * 0.2

                if score > best_score:
                    best_score = score
                    best_job = job
                    best_index = i

            if best_job is None:
                return {"job": None}

            job_queue.pop(best_index)
            claimed_job = best_job
            job_id = _claim_job(worker_name, claimed_job)

    event_log.append(job_id, "assigned", {"node_id": worker_name})
    emit(job_id, "started", {
        "worker": worker_name,
        "task_type": claimed_job.get("task_type"),
        "swarm_state": scale_state,
        "score": best_score,
        "source": "matched" if scale_state is None else "local",
    })

    return {
        "job": claimed_job,
        "swarm_state": scale_state,
    }

# =========================
# COMPLETE JOB
# =========================

@app.post("/complete/{job_id}")
def complete(job_id: str, result: dict):

    runtime = 0
    worker_name = None

    if isinstance(result, dict) and "result" in result and len(result) == 1:
        # Workers currently wrap payloads as {"result": actual_result}
        result = result["result"]

    with lock:

        if job_id not in job_store:
            return {"ok": False}

        if job_id in running_jobs:
            runtime = time.time() - running_jobs[job_id]["start_time"]
            worker_name = running_jobs[job_id]["worker"]

        result_bytes = json.dumps(result, default=str).encode("utf-8")
        output_artifact = artifact_store.put(result_bytes)

        job_store[job_id].update({
            "status": "completed",
            "result": result,
            "runtime": runtime,
            "output_hash": output_artifact.artifact_id,
        })

        if worker_name:
            workers[worker_name]["status"] = "idle"
            workers[worker_name]["load"] = max(
                0,
                workers[worker_name]["load"]
                - running_jobs[job_id]["job"]["memory_gb"]
            )

            del running_jobs[job_id]

    emit(job_id, "completed", {
        "runtime": runtime,
        "output_hash": output_artifact.artifact_id,
    })

    return {"ok": True}


@app.post("/complete_job")
def complete_job(payload: dict):
    job_id = payload.get("job_id")
    result = payload.get("result", {})
    if not job_id:
        return {"ok": False, "error": "missing job_id"}
    return complete(job_id, result)

# =========================
# MATCH LOOP (Phase 2b): periodic global assignment
# =========================

def _run_match_cycle():
    """One pass of the match step — separated from _match_loop's sleep
    loop so it can be invoked directly (tests) as well as periodically
    (the real server).

    Real bug found running this against genuinely separate machines over
    a real network, not caught by any localhost test: a worker that
    registered once and then never heartbeated again (dead process, lost
    connection — exactly what happens on a real unreliable network) keeps
    status "idle" forever, since only the watchdog clears staleness and
    it only ever looks at *busy* workers. That stale-but-"idle" worker
    was then not just eligible for matching but *favored* — it has no
    measured latency, and unmeasured latency costs zero penalty in
    GlobalDecisionBrain.decide, so it out-scored a real, responsive
    worker with real (nonzero) measured latency. A job got assigned to a
    worker that would never poll for it again, and sat there
    indefinitely — silent, no error, nothing in any log to point at it.
    Filtering to workers with a recent heartbeat closes this.
    """
    with lock:
        if not job_queue:
            return
        now = time.time()
        idle_workers = {
            name: w for name, w in workers.items()
            if w.get("status") == "idle"
            and name not in pending_assignment
            and (now - w.get("last_heartbeat", 0)) <= HEARTBEAT_TIMEOUT_SECONDS
        }
        if not idle_workers:
            return

        assignments = match_jobs_to_workers(list(job_queue), idle_workers, topology)
        for job_id, worker_name in assignments:
            index = next((i for i, j in enumerate(job_queue) if j["id"] == job_id), None)
            if index is None:
                continue  # already claimed by a worker's own /next_job call in the meantime
            job = job_queue.pop(index)
            pending_assignment[worker_name] = job


def _match_loop():
    """Run _run_match_cycle on a fixed interval. Anything it decides is
    placed in `pending_assignment` and served the next time that worker
    calls /next_job — see the comment there for why this exists
    (per-worker argmax alone can't do cross-worker arbitration).
    """
    while True:
        time.sleep(MATCH_INTERVAL_SECONDS)
        _run_match_cycle()


# =========================
# WATCHDOG: recover work from dead workers via the spine
# =========================

def _watchdog_loop():
    while True:
        time.sleep(5)
        now = time.time()
        with lock:
            dead = [
                name for name, w in workers.items()
                if now - w.get("last_heartbeat", now) > HEARTBEAT_TIMEOUT_SECONDS
                and w.get("status") == "busy"
            ]

        for worker_name in dead:
            orphaned = event_log.recover_orphaned_tasks(worker_name)
            for task in orphaned:
                job_id = task.task_id
                with lock:
                    running = running_jobs.pop(job_id, None)
                    if job_id in job_store:
                        job_store[job_id]["status"] = "queued"
                    if running is not None:
                        job_queue.append(running["job"])
                event_log.append(job_id, "requeued", {"reason": "node_timeout", "dead_node": worker_name})
            with lock:
                if worker_name in workers:
                    workers[worker_name]["status"] = "offline"

# =========================
# STREAM
# =========================

@app.get("/stream/{job_id}")
async def stream(job_id: str):

    async def generator():

        last_seen = 0

        yield "retry: 1000\n\n"

        while True:

            events = stream_engine.get(job_id)

            while last_seen < len(events):
                yield f"data: {json.dumps(events[last_seen])}\n\n"
                last_seen += 1

            yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

            if any(e["type"] == "completed" for e in events):
                break

            await asyncio.sleep(1)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }
    return StreamingResponse(generator(), media_type="text/event-stream", headers=headers)

# =========================
# STATUS / RESULT
# =========================

@app.get("/status/{job_id}")
def status(job_id: str):
    return job_store.get(job_id, {"status": "not_found"})


@app.get("/result/{job_id}")
def result(job_id: str):
    return job_store.get(job_id, {"status": "not_found"})

# =========================
# START
# =========================

if __name__ == "__main__":
    host = os.environ.get("TRON_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("TRON_PORT", 9000)))
    reload_flag = os.environ.get("TRON_RELOAD", "false").lower() in ("1", "true", "yes", "on")
    print(f"TRON CORE STARTING on {host}:{port}")

    if _USE_FASTAPI:
        watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
        watchdog_thread.start()
        match_thread = threading.Thread(target=_match_loop, daemon=True)
        match_thread.start()
        uvicorn.run(app, host=host, port=port, reload=reload_flag)
    else:
        class _SimpleHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/" or self.path == "":
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "TRON_CORE_ONLINE"}).encode())
                elif self.path.startswith("/health"):
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok"}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        def run_simple_server(host, port):
            with socketserver.TCPServer((host, port), _SimpleHandler) as httpd:
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    pass

        print("[TRON] FastAPI not available; running minimal fallback server.")
        print("Install full server with: pip install -r requirements.txt")
        run_simple_server(host, port)
