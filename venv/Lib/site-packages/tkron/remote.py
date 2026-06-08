import requests
import time
import json
from tron.serializer import serialize

QUEUE_URL = "http://127.0.0.1:9000"


# =========================
# REMOTE FUTURE
# =========================

class RemoteFuture:

    def __init__(self, job_id: str):
        self.job_id = job_id

    # -------------------------
    # INTERNAL STATUS FETCH
    # -------------------------
    def _fetch(self):
        try:
            r = requests.get(
                f"{QUEUE_URL}/status/{self.job_id}",
                timeout=5
            )
            return r.json()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    # -------------------------
    # STATUS
    # -------------------------
    def status(self):
        return self._fetch()

    # -------------------------
    # READY CHECK
    # -------------------------
    def ready(self) -> bool:
        return self._fetch().get("status") == "completed"

    # -------------------------
    # BLOCKING RESULT
    # -------------------------
    def get(self, poll: float = 0.2):

        while True:

            data = self._fetch()
            status = data.get("status")

            if status == "completed":
                result = data.get("result", {})
                return result.get("result") or result.get("output")

            if status == "failed":
                raise Exception(data.get("error", "Job failed"))

            time.sleep(poll)


# =========================
# REMOTE DECORATOR
# =========================

def remote(fn):

    def wrapper(*args, **kwargs):

        try:
            payload = serialize((fn, args, kwargs))

            r = requests.post(
                f"{QUEUE_URL}/submit",
                json={
                    "function": payload,
                    "gpu_required": kwargs.pop("gpu_required", False),
                    "min_memory_gb": kwargs.pop("min_memory_gb", 1),
                    "priority": kwargs.pop("priority", 1),
                    "compute_weight": kwargs.pop("compute_weight", 1)
                },
                timeout=10
            )

            data = r.json()

            if "job_id" not in data:
                raise RuntimeError(f"Bad queue response: {data}")

            job_id = data["job_id"]

            print(f"[TRON REMOTE] {job_id}")

            return RemoteFuture(job_id)

        except Exception as e:
            raise RuntimeError(f"REMOTE SUBMIT FAILED: {e}")

    return wrapper