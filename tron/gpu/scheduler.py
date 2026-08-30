"""Extended TRON scheduler with vGPU capability."""
from __future__ import annotations

import time
import uuid
import sqlite3
import threading
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .cluster import VirtualGPUCluster

DATABASE_PATH = Path("tron_vgpu_master.db")
AUTH_TOKEN_HEADER = "X-TRON-AUTH"
HEARTBEAT_TIMEOUT = 20.0

app = FastAPI(title="TRON vGPU Master Scheduler")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lock = threading.Lock()
workers: Dict[str, Dict[str, Any]] = {}
jobs: Dict[str, Dict[str, Any]] = {}
active_jobs: Dict[str, Dict[str, Any]] = {}
active_shards: Dict[str, Dict[str, Any]] = {}
vgpu_cluster = VirtualGPUCluster(cluster_name="tron-vgpu-cluster")


class WorkerRegistration(BaseModel):
    name: str = Field(..., description="Worker unique name")
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    gpu_name: Optional[str] = Field(None, description="Physical GPU name if present")
    vram_gb: Optional[float] = Field(None, description="Available GPU memory in GB")
    cuda_cores: Optional[int] = Field(None, description="Physical CUDA core count")
    network_bandwidth_gbps: Optional[float] = Field(None, description="Network bandwidth to master")
    location: Optional[str] = Field(None, description="Worker public location or tag")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class JobSubmission(BaseModel):
    task_type: str = Field("vgpu_task", description="Type of task")
    payload: Dict[str, Any] = Field(..., description="Task payload")
    runtime_seconds: Optional[float] = Field(None, description="Estimated runtime in seconds")
    priority: int = Field(1, description="Job priority")
    requires_gpu: bool = Field(True, description="GPU required")
    required_vram_gb: float = Field(0.0, description="Required virtual GPU memory in GB")
    required_cuda_cores: int = Field(0, description="Required virtual CUDA cores")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class JobAssignment(BaseModel):
    job_id: str
    task_type: str
    payload: Dict[str, Any]
    assigned_at: float
    billing_rate: float
    payout_rate: float
    synthetic_profile: Dict[str, Any]
    assigned_nodes: Optional[List[str]] = None
    shard_id: Optional[str] = None
    node_id: Optional[str] = None


class HeartbeatPayload(BaseModel):
    worker_name: Optional[str] = None
    active_job_id: Optional[str] = None


class JobCompletion(BaseModel):
    job_id: str
    worker_name: str
    result: Dict[str, Any]
    success: bool = True
    runtime_seconds: Optional[float] = None
    shard_id: Optional[str] = None


# Database helpers

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workers (
                worker_name TEXT PRIMARY KEY,
                auth_token TEXT NOT NULL,
                capabilities TEXT,
                gpu_name TEXT,
                vram_gb REAL,
                cuda_cores INTEGER,
                network_bandwidth_gbps REAL,
                registered_at REAL,
                last_heartbeat REAL,
                status TEXT,
                active_job_id TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                task_type TEXT,
                payload TEXT,
                submitted_at REAL,
                priority INTEGER,
                requires_gpu INTEGER,
                required_vram_gb REAL,
                required_cuda_cores INTEGER,
                assigned_to TEXT,
                assigned_at REAL,
                completed_at REAL,
                runtime_ms INTEGER,
                billed_amount REAL,
                payout_amount REAL,
                status TEXT,
                synthetic_profile TEXT,
                start_time REAL,
                assigned_nodes TEXT,
                completed_nodes TEXT,
                job_shards TEXT,
                result TEXT
            )
            """
        )
        existing_columns = [row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()]
        if "start_time" not in existing_columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN start_time REAL")
        if "assigned_nodes" not in existing_columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN assigned_nodes TEXT")
        if "completed_nodes" not in existing_columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN completed_nodes TEXT")
        if "job_shards" not in existing_columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN job_shards TEXT")
    conn.close()


def json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def json_loads(value: Optional[str]) -> Any:
    if not value:
        return None
    return json.loads(value)


def save_worker_state(worker_name: str, worker_info: Dict[str, Any]) -> None:
    conn = get_db_connection()
    with conn:
        conn.execute(
            "REPLACE INTO workers (worker_name, auth_token, capabilities, gpu_name, vram_gb, cuda_cores, network_bandwidth_gbps, registered_at, last_heartbeat, status, active_job_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                worker_name,
                worker_info["auth_token"],
                json_dumps(worker_info.get("capabilities", {})),
                worker_info.get("gpu_name"),
                worker_info.get("vram_gb"),
                worker_info.get("cuda_cores"),
                worker_info.get("network_bandwidth_gbps"),
                worker_info["registered_at"],
                worker_info["last_heartbeat"],
                worker_info["status"],
                worker_info.get("active_job_id"),
            ),
        )
    conn.close()


def save_job_state(job_id: str, job_record: Dict[str, Any]) -> None:
    conn = get_db_connection()
    with conn:
        conn.execute(
            "REPLACE INTO jobs (job_id, task_type, payload, submitted_at, priority, requires_gpu, required_vram_gb, required_cuda_cores, assigned_to, assigned_at, completed_at, runtime_ms, billed_amount, payout_amount, status, synthetic_profile, start_time, assigned_nodes, completed_nodes, job_shards, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                job_record["task_type"],
                json_dumps(job_record["payload"]),
                job_record["submitted_at"],
                job_record["priority"],
                int(job_record["requires_gpu"]),
                job_record.get("required_vram_gb", 0.0),
                job_record.get("required_cuda_cores", 0),
                job_record.get("assigned_to"),
                job_record.get("assigned_at"),
                job_record.get("completed_at"),
                job_record.get("runtime_ms"),
                job_record.get("billed_amount"),
                job_record.get("payout_amount"),
                job_record["status"],
                json_dumps(job_record.get("synthetic_profile", {})),
                job_record.get("start_time"),
                json_dumps(job_record.get("assigned_nodes", [])),
                json_dumps(job_record.get("completed_nodes", [])),
                json_dumps(job_record.get("job_shards", {})),
                json_dumps(job_record.get("result")) if job_record.get("result") is not None else None,
            ),
        )
    conn.close()


def get_cluster_resource_usage() -> tuple[float, int]:
    """Sum reserved and running vGPU resource consumption from active jobs."""
    used_vram = 0.0
    used_cores = 0
    for job in jobs.values():
        if job["status"] in {"reserved", "running"}:
            used_vram += job["required_vram_gb"]
            used_cores += job["required_cuda_cores"]
    return used_vram, used_cores


def get_available_cluster_resources() -> tuple[float, int]:
    """Return the remaining aggregate vGPU capacity available for new assignments."""
    profile = vgpu_cluster.aggregated_profile()
    used_vram, used_cores = get_cluster_resource_usage()
    return (
        max(profile.aggregated_vram_gb - used_vram, 0.0),
        max(profile.aggregated_cuda_cores - used_cores, 0),
    )


def get_idle_gpu_nodes() -> List[str]:
    return [
        node.node_id
        for node in vgpu_cluster.active_nodes()
        if node.node_id in workers and workers[node.node_id]["status"] == "idle"
    ]


def select_nodes_for_job(required_vram_gb: float, required_cuda_cores: int, preferred_worker: Optional[str] = None) -> List[str]:
    available_nodes = [
        node
        for node in vgpu_cluster.active_nodes()
        if node.node_id in workers and workers[node.node_id]["status"] == "idle"
    ]
    if preferred_worker:
        available_nodes.sort(
            key=lambda node: (
                node.node_id != preferred_worker,
                -node.vram_gb,
                -node.cuda_cores,
            )
        )
    else:
        available_nodes.sort(key=lambda node: (-node.vram_gb, -node.cuda_cores))

    selected: List[str] = []
    total_vram = 0.0
    total_cores = 0
    for node in available_nodes:
        selected.append(node.node_id)
        total_vram += node.vram_gb
        total_cores += node.cuda_cores
        if total_vram >= required_vram_gb and total_cores >= required_cuda_cores:
            return selected
    return []


def reserve_job_nodes(job_record: Dict[str, Any], node_ids: List[str], now: float) -> None:
    job_record["assigned_to"] = node_ids[0] if node_ids else None
    job_record["assigned_nodes"] = node_ids
    job_record["completed_nodes"] = []
    job_record["job_shards"] = {
        node_id: {
            "shard_id": f"{job_record['job_id']}:{node_id}",
            "node_id": node_id,
            "status": "reserved",
            "assigned_at": None,
            "completed_at": None,
            "required_vram_gb": min(job_record["required_vram_gb"], workers[node_id].get("vram_gb", 0.0)),
            "required_cuda_cores": min(job_record["required_cuda_cores"], workers[node_id].get("cuda_cores", 0)),
        }
        for node_id in node_ids
    }
    job_record["status"] = "reserved"
    job_record["assigned_at"] = now
    job_record["start_time"] = now
    save_job_state(job_record["job_id"], job_record)


def find_worker_reserved_shard(worker_name: str) -> Optional[Dict[str, Any]]:
    for job in jobs.values():
        if job["status"] in {"reserved", "running"} and job.get("job_shards"):
            shard = job["job_shards"].get(worker_name)
            if shard and shard["status"] == "reserved":
                return job
    return None


def assign_worker_shard(job_record: Dict[str, Any], worker_name: str, now: float) -> Dict[str, Any]:
    shard = job_record["job_shards"][worker_name]
    shard["status"] = "running"
    shard["assigned_at"] = now
    job_record["status"] = "running"
    save_job_state(job_record["job_id"], job_record)
    return shard


def load_persistent_state() -> None:
    conn = get_db_connection()
    with conn:
        worker_rows = conn.execute("SELECT * FROM workers").fetchall()
        for row in worker_rows:
            workers[row["worker_name"]] = {
                "auth_token": row["auth_token"],
                "capabilities": json_loads(row["capabilities"]),
                "gpu_name": row["gpu_name"],
                "vram_gb": row["vram_gb"],
                "cuda_cores": row["cuda_cores"],
                "network_bandwidth_gbps": row["network_bandwidth_gbps"],
                "registered_at": row["registered_at"],
                "last_heartbeat": row["last_heartbeat"],
                "status": row["status"],
                "active_job_id": row["active_job_id"],
            }
            if row["gpu_name"] and row["vram_gb"] and row["cuda_cores"]:
                vgpu_cluster.register_node(
                    node_id=row["worker_name"],
                    gpu_name=row["gpu_name"],
                    vram_gb=row["vram_gb"],
                    cuda_cores=row["cuda_cores"],
                    network_bandwidth_gbps=row["network_bandwidth_gbps"] or 1.0,
                )

        job_rows = conn.execute("SELECT * FROM jobs").fetchall()
        for row in job_rows:
            assigned_nodes = json_loads(row["assigned_nodes"])
        completed_nodes = json_loads(row["completed_nodes"])
        job_shards = json_loads(row["job_shards"])
        result_data = json_loads(row["result"])
        jobs[row["job_id"]] = {
                "job_id": row["job_id"],
                "task_type": row["task_type"],
                "payload": json_loads(row["payload"]),
                "submitted_at": row["submitted_at"],
                "priority": row["priority"],
                "requires_gpu": bool(row["requires_gpu"]),
                "required_vram_gb": row["required_vram_gb"],
                "required_cuda_cores": row["required_cuda_cores"],
                "assigned_to": row["assigned_to"],
                "assigned_at": row["assigned_at"],
                "completed_at": row["completed_at"],
                "runtime_ms": row["runtime_ms"],
                "billed_amount": row["billed_amount"],
                "payout_amount": row["payout_amount"],
                "status": row["status"],
                "start_time": row["start_time"],
                "assigned_nodes": assigned_nodes if isinstance(assigned_nodes, list) else [],
                "completed_nodes": completed_nodes if isinstance(completed_nodes, list) else [],
                "job_shards": job_shards if isinstance(job_shards, dict) else {},
                "synthetic_profile": json_loads(row["synthetic_profile"]),
                "result": result_data if isinstance(result_data, dict) else {},
            }
    conn.close()


def verify_auth_token(auth_token: str = Header(..., alias=AUTH_TOKEN_HEADER)) -> str:
    with lock:
        for worker_name, worker_info in workers.items():
            if worker_info["auth_token"] == auth_token:
                return worker_name
    raise HTTPException(status_code=401, detail="Invalid authentication token")


@app.on_event("startup")
def startup_event() -> None:
    initialize_db()
    threading.Thread(target=worker_watchdog, daemon=True).start()


@app.post("/register")
def register_worker(payload: WorkerRegistration) -> Dict[str, Any]:
    worker_name = payload.name
    auth_token = str(uuid.uuid4())
    now = time.time()
    worker_info = {
        "auth_token": auth_token,
        "capabilities": payload.capabilities,
        "gpu_name": payload.gpu_name,
        "vram_gb": payload.vram_gb,
        "cuda_cores": payload.cuda_cores,
        "network_bandwidth_gbps": payload.network_bandwidth_gbps,
        "location": payload.location,
        "metadata": payload.metadata,
        "registered_at": now,
        "last_heartbeat": now,
        "status": "idle",
        "active_job_id": None,
    }

    with lock:
        if worker_name in workers:
            raise HTTPException(status_code=409, detail="Worker name already registered")
        workers[worker_name] = worker_info
        save_worker_state(worker_name, worker_info)
        if payload.gpu_name and payload.vram_gb and payload.cuda_cores:
            vgpu_cluster.register_node(
                node_id=worker_name,
                gpu_name=payload.gpu_name,
                vram_gb=payload.vram_gb,
                cuda_cores=payload.cuda_cores,
                network_bandwidth_gbps=payload.network_bandwidth_gbps or 1.0,
            )

    return {"worker_name": worker_name, "auth_token": auth_token}


@app.post("/heartbeat")
def heartbeat(payload: HeartbeatPayload, worker_name: str = Depends(verify_auth_token)) -> Dict[str, Any]:
    now = time.time()
    with lock:
        if worker_name not in workers:
            raise HTTPException(status_code=404, detail="Worker not found")

        workers[worker_name]["last_heartbeat"] = now
        workers[worker_name]["status"] = "busy" if payload.active_job_id else "idle"
        workers[worker_name]["active_job_id"] = payload.active_job_id
        save_worker_state(worker_name, workers[worker_name])

    return {"status": "ok", "time": now}


def can_assign_job(job_record: Dict[str, Any], worker_name: str) -> bool:
    if job_record["status"] != "queued":
        return False
    if job_record["requires_gpu"] and not workers[worker_name]["capabilities"].get("gpu", False):
        return False
    node_ids = select_nodes_for_job(job_record["required_vram_gb"], job_record["required_cuda_cores"], preferred_worker=worker_name)
    return bool(node_ids)


def choose_next_job(worker_name: str) -> Optional[Dict[str, Any]]:
    reserved_job = find_worker_reserved_shard(worker_name)
    if reserved_job:
        return reserved_job

    worker = workers[worker_name]
    candidates = []
    for job in jobs.values():
        if not can_assign_job(job, worker_name):
            continue
        candidates.append(job)

    if not candidates:
        return None

    candidates.sort(key=lambda j: (-j["priority"], j["submitted_at"]))
    return candidates[0]


@app.get("/next_job")
def next_job(auth_token: str = Header(..., alias=AUTH_TOKEN_HEADER)) -> Dict[str, Any]:
    worker_name = verify_auth_token(auth_token)
    now = time.time()
    with lock:
        if worker_name not in workers:
            raise HTTPException(status_code=404, detail="Worker not found")

        worker = workers[worker_name]
        worker["last_heartbeat"] = now
        save_worker_state(worker_name, worker)

        if worker["active_job_id"]:
            return {"job": None, "reason": "worker already running a job"}

        job_record = choose_next_job(worker_name)
        if not job_record:
            return {"job": None}

        if job_record["status"] == "reserved":
            shard = assign_worker_shard(job_record, worker_name, now)
            assigned_nodes = job_record["assigned_nodes"]
        else:
            node_ids = select_nodes_for_job(job_record["required_vram_gb"], job_record["required_cuda_cores"], preferred_worker=worker_name)
            if not node_ids:
                return {"job": None, "reason": "no available node set for this job"}
            reserve_job_nodes(job_record, node_ids, now)
            shard = assign_worker_shard(job_record, worker_name, now)
            assigned_nodes = node_ids

        worker["status"] = "busy"
        worker["active_job_id"] = job_record["job_id"]
        save_worker_state(worker_name, worker)

        active_jobs[job_record["job_id"]] = {
            "worker": worker_name,
            "job_id": job_record["job_id"],
            "start_time": now,
            "billed_rate": 2.50 / 3600.0,
            "payout_rate": 1.00 / 3600.0,
        }
        active_shards[shard["shard_id"]] = {
            "job_id": job_record["job_id"],
            "node_id": worker_name,
            "shard_id": shard["shard_id"],
            "start_time": now,
        }

    assignment = JobAssignment(
        job_id=job_record["job_id"],
        task_type=job_record["task_type"],
        payload=job_record["payload"],
        assigned_at=now,
        billing_rate=2.50,
        payout_rate=1.00,
        synthetic_profile={
            "profile_name": vgpu_cluster.aggregated_profile().profile_name,
            "aggregated_vram_gb": vgpu_cluster.aggregated_profile().aggregated_vram_gb,
            "aggregated_cuda_cores": vgpu_cluster.aggregated_profile().aggregated_cuda_cores,
            "node_count": vgpu_cluster.aggregated_profile().node_count,
            "assigned_nodes": assigned_nodes,
        },
        assigned_nodes=assigned_nodes,
        shard_id=shard.get("shard_id"),
        node_id=worker_name,
    )
    return {"job": assignment.dict()}


@app.post("/submit_job")
def submit_job(payload: JobSubmission) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    now = time.time()
    profile = vgpu_cluster.aggregated_profile()
    if payload.required_vram_gb > profile.aggregated_vram_gb:
        raise HTTPException(status_code=400, detail="Requested vGPU memory exceeds available cluster memory")
    if payload.required_cuda_cores > profile.aggregated_cuda_cores:
        raise HTTPException(status_code=400, detail="Requested virtual CUDA cores exceed available aggregate cores")

    job_record = {
        "job_id": job_id,
        "task_type": payload.task_type,
        "payload": payload.payload,
        "submitted_at": now,
        "priority": payload.priority,
        "requires_gpu": payload.requires_gpu,
        "required_vram_gb": payload.required_vram_gb,
        "required_cuda_cores": payload.required_cuda_cores,
        "assigned_to": None,
        "assigned_at": None,
        "completed_at": None,
        "runtime_ms": None,
        "billed_amount": None,
        "payout_amount": None,
        "status": "queued",
        "synthetic_profile": {
            "profile_name": profile.profile_name,
            "aggregated_vram_gb": profile.aggregated_vram_gb,
            "aggregated_cuda_cores": profile.aggregated_cuda_cores,
            "node_count": profile.node_count,
        },
        "result": None,
    }

    with lock:
        jobs[job_id] = job_record
        save_job_state(job_id, job_record)

    return {"job_id": job_id, "status": "queued", "synthetic_profile": job_record["synthetic_profile"]}


@app.post("/complete_job")
def complete_job(payload: JobCompletion, worker_name: str = Depends(verify_auth_token)) -> Dict[str, Any]:
    now = time.time()
    with lock:
        if payload.job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job_record = jobs[payload.job_id]
        if not job_record.get("job_shards") or worker_name not in job_record["job_shards"]:
            raise HTTPException(status_code=403, detail="Worker is not part of this job")

        shard = job_record["job_shards"][worker_name]
        if shard["status"] != "running":
            raise HTTPException(status_code=400, detail="Shard is not running")

        runtime_ms = int((payload.runtime_seconds if payload.runtime_seconds is not None else 0.0) * 1000)
        if payload.runtime_seconds is None and payload.job_id in active_jobs:
            runtime_ms = int((now - active_jobs[payload.job_id]["start_time"]) * 1000)

        shard["status"] = "completed"
        shard["completed_at"] = now
        shard["runtime_ms"] = runtime_ms

        completed_nodes = job_record.get("completed_nodes")
        if not isinstance(completed_nodes, list):
            completed_nodes = []
        completed_nodes.append(worker_name)
        job_record["completed_nodes"] = completed_nodes

        result_map = job_record.get("result")
        if not isinstance(result_map, dict):
            result_map = {}
        result_map[worker_name] = {
            "shard_id": shard["shard_id"],
            "runtime_ms": runtime_ms,
            "success": payload.success,
            "result": payload.result,
        }
        job_record["result"] = result_map

        if shard["shard_id"] in active_shards:
            del active_shards[shard["shard_id"]]

        assigned_nodes = job_record.get("assigned_nodes")
        if not isinstance(assigned_nodes, list):
            assigned_nodes = []
            job_record["assigned_nodes"] = []

        if len(job_record["completed_nodes"]) == len(assigned_nodes):
            job_record["status"] = "completed" if payload.success else "failed"
            job_record["completed_at"] = now
            job_record["runtime_ms"] = int(
                max(
                    (shard_info or {}).get("runtime_ms", 0)
                    for shard_info in job_record.get("job_shards", {}).values()
                )
            )
            billed_amount = (job_record["runtime_ms"] / 1000.0) * 2.50 / 3600.0
            payout_amount = (job_record["runtime_ms"] / 1000.0) * 1.00 / 3600.0
            job_record["billed_amount"] = billed_amount
            job_record["payout_amount"] = payout_amount
            if payload.job_id in active_jobs:
                del active_jobs[payload.job_id]
        else:
            job_record["status"] = "running"
            billed_amount = 0.0
            payout_amount = 0.0

        save_job_state(payload.job_id, job_record)

        worker = workers[worker_name]
        worker["status"] = "idle"
        worker["active_job_id"] = None
        worker["last_heartbeat"] = now
        save_worker_state(worker_name, worker)

    return {"ok": True, "runtime_ms": runtime_ms, "billed": billed_amount, "payout": payout_amount}


@app.get("/workers")
def get_workers() -> Dict[str, Any]:
    with lock:
        return {"workers": workers, "synthetic_vgpu_profile": vgpu_cluster.aggregated_profile().__dict__}


@app.get("/jobs")
def get_jobs() -> Dict[str, Any]:
    with lock:
        return {"jobs": jobs}


@app.get("/ledger")
def get_ledger() -> Dict[str, Any]:
    conn = get_db_connection()
    with conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY submitted_at DESC").fetchall()
    return {"ledger": [dict(row) for row in rows]}


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "TRON vGPU Scheduler",
        "status": "ok",
        "message": "Use /health for health checks or /docs for the API reference.",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def worker_watchdog() -> None:
    while True:
        now = time.time()
        requeued = []
        with lock:
            stale_workers = [name for name, info in workers.items() if now - info["last_heartbeat"] > HEARTBEAT_TIMEOUT]
            for worker_name in stale_workers:
                worker_info = workers[worker_name]
                active_job_id = worker_info.get("active_job_id")
                if active_job_id and active_job_id in jobs:
                    job_record = jobs[active_job_id]
                    if job_record["status"] in {"running", "reserved"}:
                        stale_shard_ids = [shard_info.get("shard_id") for shard_info in (job_record.get("job_shards") or {}).values()]
                        job_record["status"] = "queued"
                        job_record["assigned_to"] = None
                        job_record["assigned_at"] = None
                        job_record["start_time"] = None
                        job_record["billed_amount"] = None
                        job_record["payout_amount"] = None
                        job_record["assigned_nodes"] = []
                        job_record["completed_nodes"] = []
                        job_record["job_shards"] = {}
                        job_record["result"] = None
                        for shard_id in stale_shard_ids:
                            if shard_id and shard_id in active_shards:
                                del active_shards[shard_id]
                        jobs[active_job_id] = job_record
                        save_job_state(active_job_id, job_record)
                        requeued.append(active_job_id)
                        active_jobs.pop(active_job_id, None)

                worker_info["status"] = "offline"
                worker_info["active_job_id"] = None
                workers[worker_name] = worker_info
                save_worker_state(worker_name, worker_info)

        if requeued:
            print(f"[WATCHDOG] Requeued stale jobs: {requeued}")
        time.sleep(HEARTBEAT_TIMEOUT / 2)


def main() -> None:
    host = "0.0.0.0"
    port = 9002
    initialize_db()
    load_persistent_state()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
