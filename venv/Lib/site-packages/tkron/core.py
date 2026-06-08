import requests
import cloudpickle
import base64
import uuid
import time

QUEUE_URL = "http://127.0.0.1:9000"


class Tron:

    def __init__(self):

        self.batch_id = str(uuid.uuid4())

    def submit(self, fn, depends_on=None):

        if depends_on is None:
            depends_on = []

        serialized = cloudpickle.dumps(fn)

        encoded = base64.b64encode(
            serialized
        ).decode()

        payload = {
            "function": encoded,
            "gpu_required": getattr(
                fn,
                "gpu_required",
                False
            ),
            "min_memory_gb": getattr(
                fn,
                "min_memory_gb",
                1
            ),
            "batch_id": self.batch_id,
            "depends_on": depends_on
        }

        r = requests.post(
            f"{QUEUE_URL}/submit",
            json=payload
        )

        data = r.json()

        return data["job_id"]

    def wait(self, job_id):

        while True:

            r = requests.get(
                f"{QUEUE_URL}/status/{job_id}"
            )

            data = r.json()

            status = data.get("status")

            print(
                f"{job_id[:8]} -> {status}"
            )

            if status == "completed":
                return data

            time.sleep(2)

    def run(self, *tasks):

        previous_job = None

        for task in tasks:

            job_id = self.submit(
                task,
                depends_on=[
                    previous_job
                ] if previous_job else []
            )

            previous_job = job_id

        self.wait(previous_job)