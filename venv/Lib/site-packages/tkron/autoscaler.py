import time
import requests
import subprocess
import sys

QUEUE_URL = "http://127.0.0.1:9000"

# =========================
# CONFIG
# =========================

MAX_WORKERS = 4
MIN_WORKERS = 1

# =========================
# TRACK SPAWNED WORKERS
# =========================

spawned = []

# =========================
# METRICS FETCH
# =========================

def get_metrics():

    try:

        r = requests.get(
            f"{QUEUE_URL}/metrics",
            timeout=2
        )

        return r.json()

    except:

        return None

# =========================
# SPAWN WORKER
# =========================

def spawn_worker():

    worker_id = f"auto_worker_{len(spawned)}"

    print(
        f"[AUTOSCALER] Spawning {worker_id}"
    )

    proc = subprocess.Popen(
        [
            sys.executable,
            "node.py",
            worker_id
        ]
    )

    spawned.append(proc)

# =========================
# AUTOSCALE LOOP
# =========================

def run_autoscaler():

    while True:

        metrics = get_metrics()

        if not metrics:
            time.sleep(2)
            continue

        queue_size = metrics["queue_size"]
        workers = metrics["workers"]

        # =========================
        # SCALE UP
        # =========================

        if queue_size > workers * 2 and workers < MAX_WORKERS:

            spawn_worker()

        # =========================
        # SCALE DOWN (basic version)
        # =========================

        elif queue_size == 0 and workers > MIN_WORKERS:

            print("[AUTOSCALER] Idle state detected")

        time.sleep(3)

# =========================
# RUN
# =========================

if __name__ == "__main__":

    print("[AUTOSCALER] ONLINE")

    run_autoscaler()