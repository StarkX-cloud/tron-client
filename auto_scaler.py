import time
import uuid


class AutoScaler:

    def __init__(self):

        self.spawned_workers = {}

    # =========================
    # SCALE UP LOGIC
    # =========================

    def scale_up(self, workers, job_queue):

        if len(job_queue) < 15:
            return None

        new_worker_id = f"auto-{uuid.uuid4().hex[:6]}"

        self.spawned_workers[new_worker_id] = {
            "gpu": False,
            "memory_gb": 4,
            "load": 0,
            "status": "idle",
            "auto_spawned": True,
            "created_at": time.time()
        }

        workers[new_worker_id] = self.spawned_workers[new_worker_id]

        print(f"[AUTO-SCALER] Spawned worker {new_worker_id}")

        return new_worker_id

    # =========================
    # SCALE DOWN LOGIC
    # =========================

    def scale_down(self, workers):

        to_remove = []

        for name, w in workers.items():

            if w.get("auto_spawned") and w.get("load", 0) == 0:

                idle_time = time.time() - w.get("created_at", time.time())

                if idle_time > 30:

                    to_remove.append(name)

        for name in to_remove:

            print(f"[AUTO-SCALER] Removing idle worker {name}")

            del workers[name]