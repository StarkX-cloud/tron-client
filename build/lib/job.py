import time


class Job:

    def __init__(self, tron, job_id):

        self.tron = tron
        self.job_id = job_id

    def status(self):

        return self.tron.status(
            self.job_id
        )

    def result(self):

        return self.tron.result(
            self.job_id
        )

    def done(self):
        status = self.status()
        return isinstance(status, dict) and status.get("status") == "completed"

    def wait(self, interval=1):

        while True:

            r = self.tron.result(
                self.job_id
            )

            if not r:
                time.sleep(interval)
                continue

            if r.get("status") == "completed":
                return r

            time.sleep(interval)