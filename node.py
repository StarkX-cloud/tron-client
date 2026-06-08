import requests
import time
import traceback
import threading
import json

from concurrent.futures import ThreadPoolExecutor
from tron.serializer import deserialize

# =========================
# CONFIG
# =========================

QUEUE_URL = "http://127.0.0.1:9000"
WORKER_NAME = "TRON_NODE_01"
MAX_PARALLEL_JOBS = 4

executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS)

# =========================
# SAFE REQUEST WRAPPER
# =========================

def safe_get(url, retries=3):
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=5)
            try:
                return r.json()
            except:
                print("[WARN] Non-JSON response:", r.text[:200])
                return None
        except:
            time.sleep(0.2)
    return None


def safe_post(url, data=None, retries=3):
    for _ in range(retries):
        try:
            r = requests.post(url, json=data, timeout=5)
            try:
                return r.json()
            except:
                return None
        except:
            time.sleep(0.2)
    return None

# =========================
# REGISTER NODE
# =========================

print(f"[REGISTERING] {WORKER_NAME}")

worker_payload = {
    "name": WORKER_NAME,
    "gpu": False,
    "memory_gb": 8
}

resp = safe_post(f"{QUEUE_URL}/register_worker", worker_payload)

print(resp if resp else "[REGISTER FAILED]")

print(f"[{WORKER_NAME}] ONLINE ({MAX_PARALLEL_JOBS} workers)")

# =========================
# STATE
# =========================

active_jobs = set()
lock = threading.Lock()

# =========================
# SAFE RESULT SERIALIZER
# =========================

def safe(obj):
    try:
        json.dumps(obj)
        return obj
    except:
        return str(obj)

# =========================
# EXECUTION CORE
# =========================

def run_task(fn_data, job_id):

    try:
        if isinstance(fn_data, tuple) and len(fn_data) == 3:
            fn, args, kwargs = fn_data
        else:
            fn = fn_data
            args = []
            kwargs = {}

        result = fn(*args, **kwargs) if callable(fn) else fn
        result = safe(result)

        payload = {"output": result}

        safe_post(f"{QUEUE_URL}/complete/{job_id}", payload)

        print(f"[DONE] {job_id}")

    except Exception as e:
        print(f"[ERROR] {job_id} -> {e}")
        traceback.print_exc()

        safe_post(f"{QUEUE_URL}/complete/{job_id}", {
            "error": str(e)
        })

    finally:
        with lock:
            active_jobs.discard(job_id)

# =========================
# JOB EXECUTION WRAPPER
# =========================

def execute_job(job):

    job_id = job.get("id")

    try:
        print(f"[START] {job_id}")

        encoded = job.get("function")
        if not encoded:
            raise ValueError("Missing function payload")

        fn_data = deserialize(encoded)

        run_task(fn_data, job_id)

    except Exception as e:
        print(f"[NODE ERROR SAFE] {job_id} -> {e}")
        traceback.print_exc()

        safe_post(f"{QUEUE_URL}/complete/{job_id}", {
            "error": str(e)
        })

    finally:
        with lock:
            active_jobs.discard(job_id)

# =========================
# MAIN LOOP (STABLE CLUSTER MODE)
# =========================

while True:

    # heartbeat
    safe_post(f"{QUEUE_URL}/heartbeat/{WORKER_NAME}")

    # concurrency control
    with lock:
        if len(active_jobs) >= MAX_PARALLEL_JOBS:
            time.sleep(0.1)
            continue

    # fetch job
    data = safe_get(f"{QUEUE_URL}/next_job/{WORKER_NAME}")

    if not data:
        time.sleep(0.2)
        continue

    job = data.get("job")

    if not job:
        time.sleep(0.1)
        continue

    job_id = job["id"]

    with lock:
        if job_id in active_jobs:
            continue
        active_jobs.add(job_id)

    print(f"[PICKED] {job_id}")

    executor.submit(execute_job, job)