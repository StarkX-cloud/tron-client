try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    import uvicorn
    _USE_FASTAPI = True
except Exception:
    # FastAPI or Uvicorn not installed — provide a minimal fallback HTTP server
    _USE_FASTAPI = False
    import http.server
    import socketserver
    import threading
    import json

import json
import os
import time
import uuid
import threading
import asyncio
from collections import defaultdict

# =========================
# ENGINE IMPORTS (SAFE)
# =========================

from session_manager import SessionManager
from virtual_memory import VirtualMemory
from execution_graph import ExecutionGraph
from routing_engine import RoutingEngine
from predictor_engine import PredictorEngine
from swarm_manager import SwarmManager
from auto_scaler import AutoScaler
from resurrection_engine import ResurrectionEngine
from memory_mesh import MemoryMesh
from stream_engine import StreamEngine
from simulation_engine import SimulationEngine
from pricing_engine import PricingEngine
from market_engine import MarketEngine
from load_shaper import LoadShaper
from global_brain import GlobalDecisionBrain

# =========================
# APP
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# STATE INIT
# =========================

sessions = SessionManager()
vmemory = VirtualMemory()
graphs = ExecutionGraph()

router = RoutingEngine()
predictor = PredictorEngine()
swarm = SwarmManager()
autoscaler = AutoScaler()
resurrection = ResurrectionEngine()
memory_mesh = MemoryMesh()
stream_engine = StreamEngine()

simulation_engine = SimulationEngine()
market = MarketEngine()
pricing_engine = PricingEngine()
load_shaper = LoadShaper()

global_brain = GlobalDecisionBrain(
    pricing_engine,
    market,
    load_shaper,
    swarm,
    simulation_engine
)

provider_router = None

# =========================
# STATE MEMORY
# =========================

lock = threading.Lock()

job_queue = []
job_store = {}
workers = {}
running_jobs = {}

event_bus = defaultdict(list)

# =========================
# EMIT
# =========================

def emit(job_id, event_type, data=None):

    event = {
        "job_id": job_id,
        "type": event_type,
        "data": data or {},
        "time": time.time()
    }

    stream_engine.emit(job_id, event_type, data)

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
# GRAPHS
# =========================

@app.post("/create_graph")
def create_graph():
    graph_id = graphs.create_graph()
    return {"graph_id": graph_id}

@app.get("/graph/{graph_id}")
def get_graph(graph_id: str):
    return graphs.get(graph_id) or {"status": "not_found"}

# =========================
# WORKERS
# =========================

@app.post("/register_worker")
def register_worker(worker: dict):

    with lock:
        workers[worker["name"]] = {
            "gpu": worker.get("gpu", False),
            "memory_gb": worker.get("memory_gb", 4),
            "load": 0,
            "status": "idle",
            "last_heartbeat": time.time()
        }

    return {"ok": True}


@app.post("/heartbeat/{worker_name}")
def heartbeat(worker_name: str):

    with lock:
        if worker_name in workers:
            workers[worker_name]["last_heartbeat"] = time.time()

    return {"alive": True}

# =========================
# SUBMIT JOB (CLEAN + SAFE)
# =========================

@app.post("/submit_ai")
def submit_ai(job: dict):

    job_id = str(uuid.uuid4())

    raw_job = {
        "id": job_id,
        "task_type": job.get("task_type", "inference"),
        "prompt": job.get("prompt", ""),
        "priority": int(job.get("priority", 1)),
        "gpu": bool(job.get("gpu", False)),
        "memory_gb": float(job.get("memory_gb", 2)),
        "submitted_at": time.time()
    }

    enriched_job = raw_job.copy()

    # attach optional session/graph metadata
    session_id = job.get("session_id")
    graph_id = job.get("graph_id")

    if session_id:
        enriched_job["session_id"] = session_id
        sessions.add_job(session_id, enriched_job)

    if graph_id:
        enriched_job["graph_id"] = graph_id
        graphs.add_node(graph_id, enriched_job)

    # pricing
    enriched_job["estimated_cost"] = pricing_engine.compute_price(
        job_queue,
        workers,
        enriched_job
    )

    enriched_job["routing"] = "default"

    # memory context
    try:
        enriched_job = memory_mesh.inject_context(enriched_job)
    except:
        pass

    # predictive layer (SAFE)
    try:
        enriched_job["predicted_runtime"] = predictor.predict_runtime(enriched_job)
        enriched_job["predicted_cost"] = predictor.predict_cost(enriched_job)
    except:
        pass

    # provider routing (ONLY IF AVAILABLE)
    if provider_router:
        try:
            enriched_job["provider"] = provider_router.select(
                enriched_job["prompt"],
                enriched_job["estimated_cost"]
            )
        except:
            enriched_job["provider"] = None

    with lock:
        job_queue.append(enriched_job)

        job_store[job_id] = {
            "id": job_id,
            "status": "queued",
            "submitted_at": time.time(),
            "estimated_cost": enriched_job["estimated_cost"]
        }

    emit(job_id, "queued", enriched_job)

    return {
        "job_id": job_id,
        "status": "queued"
    }


@app.post("/submit")
def submit(job: dict):

    job_id = str(uuid.uuid4())

    raw_job = {
        "id": job_id,
        "task_type": job.get("task_type", "remote"),
        "prompt": job.get("prompt", ""),
        "priority": int(job.get("priority", job.get("priority", 1))),
        "gpu": bool(job.get("gpu", job.get("gpu_required", False))),
        "memory_gb": float(job.get("memory_gb", job.get("min_memory_gb", 1))),
        "submitted_at": time.time(),
        "function": job.get("function"),
        "compute_weight": job.get("compute_weight", 1)
    }

    if raw_job["function"] is None:
        return {"error": "Missing function payload"}

    enriched_job = raw_job.copy()

    # pricing
    enriched_job["estimated_cost"] = pricing_engine.compute_price(
        job_queue,
        workers,
        enriched_job
    )

    enriched_job["routing"] = "remote"

    with lock:
        job_queue.append(enriched_job)

        job_store[job_id] = {
            "id": job_id,
            "status": "queued",
            "submitted_at": time.time(),
            "estimated_cost": enriched_job["estimated_cost"]
        }

    emit(job_id, "queued", enriched_job)

    return {
        "job_id": job_id,
        "status": "queued"
    }

# =========================
# METRICS + DASHBOARD
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

# =========================
# NEXT JOB (STABLE ROUTER CORE)
# =========================

@app.get("/next_job/{worker_name}")
def next_job(worker_name: str):

    with lock:

        if worker_name not in workers:
            return {"job": None}

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
                decision = global_brain.decide(job_queue, workers, job)
                score = decision.get("score", 0)
            except:
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

        job_id = best_job["id"]

        job_store[job_id]["status"] = "running"

        workers[worker_name]["status"] = "busy"
        workers[worker_name]["load"] += best_job.get("memory_gb", 1)

        running_jobs[job_id] = {
            "worker": worker_name,
            "start_time": time.time(),
            "job": best_job
        }

    emit(job_id, "started", {
        "worker": worker_name,
        "task_type": best_job.get("task_type"),
        "swarm_state": scale_state,
        "score": best_score
    })

    return {
        "job": best_job,
        "swarm_state": scale_state
    }

# =========================
# COMPLETE JOB (CLEAN)
# =========================

@app.post("/complete/{job_id}")
def complete(job_id: str, result: dict):

    runtime = 0
    worker_name = None

    with lock:

        if job_id not in job_store:
            return {"ok": False}

        if job_id in running_jobs:
            runtime = time.time() - running_jobs[job_id]["start_time"]
            worker_name = running_jobs[job_id]["worker"]

        job_store[job_id].update({
            "status": "completed",
            "result": result,
            "runtime": runtime,
            "cost": 0.01 + runtime * 0.001
        })

        graph_id = job_store[job_id].get("graph_id")
        if graph_id:
            graphs.update_status(graph_id, job_id, "completed")

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
        "result": result
    })

    return {"ok": True}

# =========================
# STREAM
# =========================

@app.get("/stream/{job_id}")
async def stream(job_id: str):

    async def generator():

        last_seen = 0

        # send an initial retry hint for SSE clients
        yield "retry: 1000\n\n"

        while True:

            events = stream_engine.get(job_id)

            while last_seen < len(events):
                yield f"data: {json.dumps(events[last_seen])}\n\n"
                last_seen += 1

            # keep the connection alive between events
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
# STATUS
# =========================

@app.get("/status/{job_id}")
def status(job_id: str):

    return job_store.get(job_id, {"status": "not_found"})

# =========================
# RESULT
# =========================

@app.get("/result/{job_id}")
def result(job_id: str):

    return job_store.get(job_id, {"status": "not_found"})

# =========================
# START
# =========================

if __name__ == "__main__":
    host = os.environ.get("TRON_HOST", "0.0.0.0")
    port = int(os.environ.get("TRON_PORT", os.environ.get("PORT", 9000)))
    reload_flag = os.environ.get("TRON_RELOAD", "false").lower() in ("1", "true", "yes", "on")
    print(f"TRON CORE STARTING on {host}:{port}")

    if _USE_FASTAPI:
        uvicorn.run(app, host=host, port=port, reload=reload_flag)
    else:
        # Minimal fallback: serve only basic health and root endpoints so SDK can auto-discover
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
