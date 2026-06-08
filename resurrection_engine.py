import time


class ResurrectionEngine:

    def __init__(self):

        self.dead_workers = {}
        self.failed_jobs = {}

    # =========================
    # DETECT FAILURE
    # =========================

    def detect_failure(self, workers):

        now = time.time()

        for name, w in list(workers.items()):

            last = w.get("last_heartbeat", now)

            # worker considered dead after 10s silence (prototype rule)
            if now - last > 10:

                print(f"[FAILURE] Worker dead: {name}")

                self.dead_workers[name] = w

                del workers[name]

    # =========================
    # JOB RESCUE
    # =========================

    def rescue_jobs(self, running_jobs, job_queue):

        rescued = 0

        for job_id, job in list(running_jobs.items()):

            worker = job.get("worker")

            if worker in self.dead_workers:

                print(f"[RESCUE] Re-queuing job {job_id}")

                job_queue.append(job["job"])

                self.failed_jobs[job_id] = job

                del running_jobs[job_id]

                rescued += 1

        return rescued

    # =========================
    # HEAL SYSTEM STATE
    # =========================

    def heal(self, workers, running_jobs, job_queue):

        self.detect_failure(workers)

        rescued = self.rescue_jobs(running_jobs, job_queue)

        return {
            "dead_workers": len(self.dead_workers),
            "rescued_jobs": rescued
        }