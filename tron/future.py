import requests
import time

QUEUE_URL = "http://127.0.0.1:9000"


class TronFuture:

    def status(self):
        raise NotImplementedError("Subclasses must implement status()")

    def ready(self):
        return self.status().get("status") == "completed"

    def done(self):
        return self.ready()

    def get(self, poll: float = 0.2):
        raise NotImplementedError("Subclasses must implement get()")

    def result(self):
        return self.get()

    def __await__(self):
        result = self.get()
        if False:
            yield None
        return result

    def __repr__(self):
        return f"<{self.__class__.__name__} job_id={getattr(self, 'job_id', None)}>"


class RemoteFuture(TronFuture):

    def __init__(self, job_id):
        self.job_id = job_id

    def status(self):
        try:
            r = requests.get(
                f"{QUEUE_URL}/status/{self.job_id}",
                timeout=5
            )
            return r.json()
        except Exception:
            return {"status": "error"}

    def get(self, poll: float = 0.2):
        while True:
            data = self.status()
            status = data.get("status")

            if status == "completed":
                result = data.get("result", {})
                return result.get("result") or result.get("output")

            if status == "failed":
                raise Exception(data.get("error", "Job failed"))

            time.sleep(poll)
