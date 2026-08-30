import hashlib
import os
import time
import uuid
import json
import subprocess
import traceback
from pathlib import Path

import requests

# =========================
# INTEGRATED TRON LAYERS
# =========================
try:
    from tron.gpu import VirtualGPUCluster, VirtualGPUNode
    HAS_VGPU = True
except ImportError:
    HAS_VGPU = False

TRON_MASTER_URL = os.environ.get("TRON_MASTER_URL", "http://127.0.0.1:9000")
WORKER_NAME = os.environ.get("TRON_WORKER_NAME", f"worker-{uuid.uuid4().hex[:8]}")

# The cache file is scoped to (master URL, worker name) — a real bug hit
# during development, not a defensive guess: with a single fixed cache
# path, pointing this same worker at a *different* server reused a token
# issued by (and only meaningful to) the old one. The server didn't
# recognize it, so /heartbeat and /next_job silently failed forever
# (both swallow request errors) while the process kept "running" and
# printing nothing — it looked alive but had never actually registered.
_default_auth_token_file = Path.home() / f".tron_worker_auth_{hashlib.sha256((TRON_MASTER_URL + WORKER_NAME).encode()).hexdigest()[:16]}.json"
AUTH_TOKEN_FILE = Path(os.environ.get("TRON_AUTH_TOKEN_FILE", _default_auth_token_file))
LOCATION = os.environ.get("TRON_LOCATION", "self-hosted")

# GPU cluster instance for this worker
gpu_cluster = None


def detect_gpu():
    """Detect GPU hardware via nvidia-smi and register in vGPU cluster if available."""
    gpu_available = False
    gpu_name = None
    vram_gb = 1
    cuda_cores = 1024
    
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, timeout=5
        )
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if lines:
            first = lines[0].split(",")
            gpu_name = first[0].strip()
            vram_gb = max(1, int(float(first[1].split()[0]) / 1024)) if len(first) > 1 else 1
            gpu_available = True
            cuda_cores = 4096  # Typical NVIDIA GPU core count
            
            # Register in vGPU cluster if available
            if HAS_VGPU:
                try:
                    global gpu_cluster
                    if gpu_cluster is None:
                        gpu_cluster = VirtualGPUCluster(cluster_name=f"tron-worker-{WORKER_NAME}")
                    
                    node_id = f"node-{WORKER_NAME}"
                    node = gpu_cluster.register_node(
                        node_id=node_id,
                        gpu_name=gpu_name,
                        vram_gb=vram_gb,
                        cuda_cores=cuda_cores,
                        network_bandwidth_gbps=1.0
                    )
                    print(f"[WORKER] ✓ GPU registered in vGPU cluster: {node_id}")
                except Exception as e:
                    print(f"[WORKER] Warning: Could not register GPU in vGPU cluster: {e}")
                    
    except Exception:
        pass
    
    return gpu_available, gpu_name, vram_gb, cuda_cores


def load_auth_token():
    if AUTH_TOKEN_FILE.exists():
        try:
            cached = json.loads(AUTH_TOKEN_FILE.read_text())
            # Defense in depth on top of the per-(master, name) cache path
            # above: never trust a cached token whose worker_name doesn't
            # match the one we're running as right now (e.g. if
            # TRON_AUTH_TOKEN_FILE was set manually).
            if cached.get("worker_name") != WORKER_NAME:
                return None
            return cached.get("auth_token")
        except Exception:
            return None
    return None


def save_auth_token(token):
    AUTH_TOKEN_FILE.write_text(json.dumps({"auth_token": token, "worker_name": WORKER_NAME}))


def register_worker():
    gpu_available, gpu_name, vram_gb, cuda_cores = detect_gpu()
    payload = {
        "name": WORKER_NAME,
        "capabilities": {
            "gpu": gpu_available,
            "memory_gb": vram_gb,
            "cuda_cores": cuda_cores,
            "location": LOCATION,
        },
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "cuda_cores": cuda_cores,
        "network_bandwidth_gbps": 1.0,
        "location": LOCATION,
        "metadata": {"bootstrap": "tron-node", "runtime": "python"},
    }
    try:
        resp = requests.post(f"{TRON_MASTER_URL}/register_worker", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        auth_token = data.get("auth_token")
        save_auth_token(auth_token)
        print(f"[WORKER] Registered {WORKER_NAME} with TRON master at {TRON_MASTER_URL}")
        return auth_token
    except Exception as exc:
        print(f"[WORKER] Failed to register worker: {exc}")
        raise


_last_measured_latency_ms = None
_last_measured_bw_down_mbps = None
_last_measured_bw_up_mbps = None

# How many heartbeats between bandwidth probes. A probe moves ~1.25MB, so
# doing it every heartbeat would be wasteful; latency needs every-heartbeat
# sampling, throughput changes far more slowly.
BANDWIDTH_PROBE_EVERY = int(os.environ.get("TRON_BANDWIDTH_PROBE_EVERY", "15"))
_PROBE_DOWN_BYTES = 1024 * 1024
_PROBE_UP_BYTES = 256 * 1024


def bandwidth_probe(auth_token):
    """Measure real throughput to the master: download PROBE_DOWN_BYTES via
    /probe/blob, upload PROBE_UP_BYTES to /probe/sink, divide bytes by
    wall-clock seconds. This is a genuine transfer measurement, not the
    hard-coded network_bandwidth_gbps=1.0 the old registration reported.
    Feeds tron/spine/topology.py's bandwidth map, which matcher.py and
    global_brain.py use to keep heavy-input jobs off thin pipes.
    """
    global _last_measured_bw_down_mbps, _last_measured_bw_up_mbps
    try:
        start = time.time()
        resp = requests.get(
            f"{TRON_MASTER_URL}/probe/blob",
            params={"bytes": _PROBE_DOWN_BYTES},
            headers={"X-TRON-AUTH": auth_token},
            timeout=30,
        )
        resp.raise_for_status()
        elapsed = time.time() - start
        got = len(resp.content)
        if elapsed > 0 and got > 0:
            _last_measured_bw_down_mbps = (got * 8.0) / (elapsed * 1_000_000.0)
    except Exception as exc:
        print(f"[WORKER] bandwidth down-probe failed: {exc}")

    try:
        blob = b"\x00" * _PROBE_UP_BYTES
        start = time.time()
        resp = requests.post(
            f"{TRON_MASTER_URL}/probe/sink",
            headers={"X-TRON-AUTH": auth_token, "Content-Type": "application/octet-stream"},
            data=blob,
            timeout=30,
        )
        resp.raise_for_status()
        elapsed = time.time() - start
        if elapsed > 0:
            _last_measured_bw_up_mbps = (_PROBE_UP_BYTES * 8.0) / (elapsed * 1_000_000.0)
    except Exception as exc:
        print(f"[WORKER] bandwidth up-probe failed: {exc}")


def heartbeat(auth_token):
    """Send a heartbeat, and report the round-trip time of the *previous*
    heartbeat call (plus the most recent bandwidth probe, if any) so the
    master can build up a real (not simulated) picture of this worker's
    network distance — see tron/spine/topology.py and
    tron_runtime/global_brain.py, which use this to prefer lower-latency,
    fatter-pipe workers when scoring job placement.
    """
    global _last_measured_latency_ms
    payload = {"worker_name": WORKER_NAME, "active_job_id": None}
    if _last_measured_latency_ms is not None:
        payload["latency_ms"] = _last_measured_latency_ms
    if _last_measured_bw_down_mbps is not None:
        payload["bandwidth_mbps_down"] = _last_measured_bw_down_mbps
    if _last_measured_bw_up_mbps is not None:
        payload["bandwidth_mbps_up"] = _last_measured_bw_up_mbps

    start = time.time()
    try:
        requests.post(
            f"{TRON_MASTER_URL}/heartbeat/{WORKER_NAME}",
            headers={"X-TRON-AUTH": auth_token},
            json=payload,
            timeout=5,
        )
        _last_measured_latency_ms = (time.time() - start) * 1000.0
    except Exception:
        pass


def fetch_next_job(auth_token):
    try:
        resp = requests.get(f"{TRON_MASTER_URL}/next_job/{WORKER_NAME}", headers={"X-TRON-AUTH": auth_token}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("job")
    except Exception as exc:
        print(f"[WORKER] next_job failed: {exc}")
        return None


def report_complete(auth_token, job_id, result, runtime_seconds):
    try:
        resp = requests.post(
            f"{TRON_MASTER_URL}/complete/{job_id}",
            headers={"X-TRON-AUTH": auth_token},
            json={"result": result},
            timeout=10,
        )
        resp.raise_for_status()
        print(f"[WORKER] Completed job {job_id}")
    except Exception as exc:
        print(f"[WORKER] Failed to report completion for {job_id}: {exc}")


def execute_job(job):
    try:
        payload = job.get("payload", {}) or {}
        prompt = payload.get("prompt") or payload.get("message") or "TRON job"
        print(f"[WORKER] Executing job {job.get('job_id')} prompt={prompt[:80]}")
        return {"status": "ok", "message": prompt, "worker": WORKER_NAME}
    except Exception as exc:
        traceback.print_exc()
        return {"status": "error", "error": str(exc)}


def main():
    auth_token = load_auth_token()
    if not auth_token:
        auth_token = register_worker()

    print(f"[WORKER] Starting polling loop for jobs on {TRON_MASTER_URL}")
    idle_ticks = 0
    while True:
        try:
            job = fetch_next_job(auth_token)
            if job:
                job_id = job.get("job_id")
                start = time.time()
                result = execute_job(job)
                report_complete(auth_token, job_id, result, max(0.1, time.time() - start))
            else:
                if idle_ticks % BANDWIDTH_PROBE_EVERY == 0:
                    bandwidth_probe(auth_token)
                idle_ticks += 1
                heartbeat(auth_token)
                time.sleep(1)
        except KeyboardInterrupt:
            print("[WORKER] Shutting down")
            break
        except Exception as exc:
            print(f"[WORKER] Polling error: {exc}")
            time.sleep(2)


if __name__ == "__main__":
    main()
