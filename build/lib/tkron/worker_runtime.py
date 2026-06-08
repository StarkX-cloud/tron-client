import requests
import time
import random

QUEUE_URL = "http://127.0.0.1:9000"

WORKER_NAME = "TRON_WORKER_01"

# =========================
# REGISTER WORKER
# =========================

requests.post(
    f"{QUEUE_URL}/register_worker",
    json={
        "name": WORKER_NAME,
        "gpu": False,
        "memory_gb": 8
    }
)

print(f"[CONNECTED] {WORKER_NAME}")

# =========================
# EXECUTION ENGINE
# =========================

def execute(job):

    task_type = job.get("task_type")

    print(f"[EXECUTING] {task_type}")

    # -----------------------------------
    # TRAINING
    # -----------------------------------

    if task_type == "training":

        time.sleep(5)

        return {
            "model_id": f"model_{random.randint(1000,9999)}",
            "accuracy": round(random.uniform(0.80, 0.99), 4),
            "status": "trained"
        }

    # -----------------------------------
    # INFERENCE
    # -----------------------------------

    elif task_type == "inference":

        time.sleep(2)

        return {
            "response": f"AI response to: {job.get('prompt')}",
            "tokens": random.randint(50, 500)
        }

    # -----------------------------------
    # EMBEDDING
    # -----------------------------------

    elif task_type == "embedding":

        time.sleep(1)

        return {
            "vector_size": 1536,
            "embedding_created": True
        }

    # -----------------------------------
    # PIPELINE
    # -----------------------------------

    elif task_type == "pipeline":

        time.sleep(4)

        return {
            "pipeline_complete": True,
            "steps_executed": 4
        }

    # -----------------------------------
    # AGENT
    # -----------------------------------

    elif task_type == "agent":

        time.sleep(3)

        return {
            "agent_status": "completed",
            "actions": random.randint(2, 10)
        }

    # -----------------------------------
    # DEFAULT
    # -----------------------------------

    else:

        time.sleep(1)

        return {
            "status": "unknown_task"
        }

# =========================
# MAIN LOOP
# =========================

while True:

    try:

        r = requests.get(
            f"{QUEUE_URL}/next_job/{WORKER_NAME}",
            timeout=5
        )

        data = r.json()

        if not data["job"]:
            time.sleep(1)
            continue

        job = data["job"]
        job_id = job["id"]

        result = execute(job)

        requests.post(
            f"{QUEUE_URL}/complete/{job_id}",
            json=result,
            timeout=10
        )

        print(f"[COMPLETED] {job_id}")

    except Exception as e:

        print("[WORKER ERROR]", e)
        time.sleep(2)