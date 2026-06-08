import requests
import time

from tron.serializer import serialize

QUEUE_URL = "http://127.0.0.1:9000"

# =========================
# FUTURE CORE (NEW NAME)
# =========================

class RemoteFuture:

    def __init__(self, job_id):
        self.job_id = job_id

    def status(self):

        try:
            r = requests.get(
                f"{QUEUE_URL}/status/{self.job_id}",
                timeout=5
            )
            return r.json()
        except:
            return {"status": "error"}

    def ready(self):

        return self.status().get("status") == "completed"

    def get(self):

        while True:

            data = self.status()

            if data.get("status") == "completed":
                return data.get("result", {}).get("output")

            if data.get("status") == "failed":
                raise Exception(data.get("error", "failed"))

            time.sleep(0.2)


# =========================
# COMPAT LAYER (FIXES YOUR ERROR)
# =========================

class TronFuture(RemoteFuture):
    pass


# =========================
# REMOTE DECORATOR
# =========================

def remote(fn):

    def wrapper(*args, **kwargs):

        payload = serialize((fn, args, kwargs))

        r = requests.post(
            f"{QUEUE_URL}/submit",
            json={
                "function": payload,
                "gpu_required": False,
                "min_memory_gb": 1,
                "priority": 1,
                "compute_weight": 1
            },
            timeout=10
        )

        job_id = r.json()["job_id"]

        print(f"[TRON REMOTE] {job_id}")

        return RemoteFuture(job_id)

    return wrapper