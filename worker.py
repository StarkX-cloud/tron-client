import os
import requests
import time
import random

QUEUE_URL = os.getenv("TRON_URL", os.getenv("TRON_QUEUE_URL", "http://127.0.0.1:9000"))

WORKER_NAME = f"TRON_NODE_{random.randint(1000,9999)}"


# =========================
# REGISTER WORKER
# =========================

def register():
    try:
        worker_payload = {
            "name": WORKER_NAME,
            "gpu": False,
            "memory_gb": 8
        }

        r = requests.post(
            f"{QUEUE_URL}/register_worker",
            json=worker_payload,
            timeout=10
        )

        print(f"[TRON WORKER ONLINE] {WORKER_NAME}")
        print(r.json())

    except Exception as e:
        print("[REGISTER ERROR]", e)


# =========================
# HEARTBEAT
# =========================

def heartbeat():
    try:
        requests.post(
            f"{QUEUE_URL}/heartbeat/{WORKER_NAME}",
            timeout=5
        )
    except Exception:
        pass


# =========================
# GET JOB (SAFE + BACKOFF)
# =========================

def get_job():

    try:
        r = requests.get(
            f"{QUEUE_URL}/next_job/{WORKER_NAME}",
            timeout=30
        )

        return r.json()

    except Exception as e:
        print("[WORKER ERROR]", e)
        time.sleep(2)   # BACKOFF (IMPORTANT FIX)
        return {"job": None}


# =========================
# COMPLETE JOB
# =========================

def complete_job(job_id, result):

    try:
        r = requests.post(
            f"{QUEUE_URL}/complete/{job_id}",
            json=result,
            timeout=10
        )

        print("[COMPLETED]")
        print(r.json())

    except Exception as e:
        print("[COMPLETE ERROR]", e)


# =========================
# COMPUTE
# =========================

def compute(job):

    if not job:
        return

    job_id = job.get("id")

    if not job_id:
        print("[INVALID JOB]")
        return

    print(f"[JOB RECEIVED] {job_id}")
    print(job)

    compute_time = random.randint(3, 6)

    for i in range(compute_time):
        print(f"[COMPUTING] {i+1}s")
        time.sleep(1)

    result = {
        "output": f"Completed task: {job.get('prompt')}",
        "task_type": job.get("task_type"),
        "memory_gb": job.get("memory_gb", 1),
        "worker": WORKER_NAME,
        "provider": job.get("provider")  # ✅ IMPORTANT (for router learning)
    }

    complete_job(job_id, result)


# =========================
# MAIN LOOP
# =========================

def run():

    register()

    while True:

        heartbeat()

        data = get_job()

        job = data.get("job")

        if not job:
            print("[IDLE]")
            time.sleep(2)
            continue

        compute(job)


# =========================
# START
# =========================

if __name__ == "__main__":
    run()