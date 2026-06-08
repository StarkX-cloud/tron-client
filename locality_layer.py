import requests
import time
import threading

class LocalityEngine:

    def __init__(self, endpoint="http://127.0.0.1:9000"):
        self.endpoint = endpoint

    # =========================
    # MAIN ILLUSION ENTRY POINT
    # =========================

    def run(self, task_type, payload=None):

        payload = payload or {}

        # 1. INSTANT ACK (IMPORTANT ILLUSION)
        ack = {
            "status": "processing",
            "message": "running locally..."
        }

        # simulate immediate response
        print("[TRON] running locally...")

        # 2. SEND TO BACKEND (hidden)
        r = requests.post(
            f"{self.endpoint}/submit_ai",
            json={
                "task_type": task_type,
                **payload
            }
        )

        data = r.json()
        job_id = data["job_id"]

        # 3. return LOCAL FEEL handle (NOT infrastructure handle)
        return LocalHandle(self.endpoint, job_id)

# =========================
# LOCAL HANDLE (IMPORTANT)
# =========================

class LocalHandle:

    def __init__(self, endpoint, job_id):
        self.endpoint = endpoint
        self.job_id = job_id

    # =========================
    # BLOCKING RESULT (simple)
    # =========================

    def result(self):

        while True:

            r = requests.get(
                f"{self.endpoint}/status/{self.job_id}"
            )

            data = r.json()

            if data.get("status") == "completed":
                return data

            time.sleep(0.5)

    # =========================
    # STREAM FEEL (HOOK INTO STREAMING LAYER)
    # =========================

    def stream(self):

        r = requests.get(
            f"{self.endpoint}/stream/{self.job_id}"
        )

        return r.json()