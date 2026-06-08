import requests
import time

QUEUE_URL = "http://127.0.0.1:9000"

class StreamFuture:

    def __init__(self, job_id):
        self.job_id = job_id

    def watch(self):

        last_status = None

        while True:

            r = requests.get(
                f"{QUEUE_URL}/status/{self.job_id}"
            )

            data = r.json()
            status = data.get("status")

            if status != last_status:

                print(f"[TRON STREAM] {status}")

                last_status = status

            if status == "completed":

                return data.get("result", {}).get("output")

            if status == "failed":
                raise Exception("Job failed")

            time.sleep(0.3)