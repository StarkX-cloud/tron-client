import requests
import time

QUEUE_URL = "http://127.0.0.1:9000"


# =========================
# FUTURE OBJECT (ASYNC HANDLE)
# =========================

class TronFuture:

    def __init__(self, job_id: str):
        self.job_id = job_id

    def status(self):
        try:
            r = requests.get(f"{QUEUE_URL}/status/{self.job_id}", timeout=5)
            return r.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def ready(self):
        return self.status().get("status") == "completed"

    def result(self):
        while True:
            data = self.status()

            if data.get("status") == "completed":
                return data.get("result")

            if data.get("status") == "failed":
                raise Exception("Job failed")

            time.sleep(0.3)


# =========================
# CORE SDK CLIENT
# =========================

class Tron:

    def __init__(self, endpoint: str = QUEUE_URL):
        self.endpoint = endpoint

    # -------------------------
    # AI JOB SUBMISSION (MAIN API)
    # -------------------------

    def run(
        self,
        task_type: str,
        prompt: str = "",
        model: str = None,
        dataset: str = None,
        gpu: bool = False,
        memory: int = 2,
        priority: int = 1,
        depends_on: list = None
    ):

        payload = {
            "task_type": task_type,
            "prompt": prompt,
            "model": model,
            "dataset": dataset,
            "gpu": gpu,
            "memory": memory,
            "priority": priority,
            "depends_on": depends_on or []
        }

        r = requests.post(
            f"{self.endpoint}/submit_ai",
            json=payload,
            timeout=10
        )

        data = r.json()
        job_id = data["job_id"]

        print(f"[TRON SDK] job submitted → {job_id}")

        return TronFuture(job_id)

    # -------------------------
    # QUICK HELPERS (DX BOOST)
    # -------------------------

    def train(self, model, dataset, **kwargs):
        return self.run(
            task_type="training",
            model=model,
            dataset=dataset,
            **kwargs
        )

    def infer(self, model, prompt, **kwargs):
        return self.run(
            task_type="inference",
            model=model,
            prompt=prompt,
            **kwargs
        )

    def pipeline(self, steps: list, **kwargs):
        return self.run(
            task_type="pipeline",
            prompt=str(steps),
            **kwargs
        )