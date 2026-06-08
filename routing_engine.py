import time
from collections import defaultdict


class RoutingEngine:

    def __init__(self):

        # =========================
        # WORKER MEMORY STORE
        # =========================
        self.worker_stats = defaultdict(lambda: {
            "jobs": 0,
            "total_runtime": 0.0,
            "failures": 0,
            "gpu_success": 0,
            "cpu_success": 0,
            "last_seen": time.time()
        })

    # =========================
    # UPDATE WORKER PROFILE
    # =========================
    def update_worker(self, worker_name, runtime, success=True, gpu_used=False):

        stats = self.worker_stats[worker_name]

        stats["jobs"] += 1
        stats["total_runtime"] += float(runtime)
        stats["last_seen"] = time.time()

        if not success:
            stats["failures"] += 1

        if success and gpu_used:
            stats["gpu_success"] += 1

        if success and not gpu_used:
            stats["cpu_success"] += 1

    # =========================
    # WORKER SCORING ENGINE
    # =========================
    def score_worker(self, worker, job, worker_name=None):

        stats = self.worker_stats[worker_name]

        jobs = max(stats["jobs"], 1)
        failures = stats["failures"]

        avg_runtime = stats["total_runtime"] / jobs
        failure_rate = failures / jobs

        # =========================
        # HARD CONSTRAINTS
        # =========================

        # GPU constraint
        if job.get("gpu") and not worker.get("gpu"):
            return float("-inf")

        # Memory constraint
        if worker.get("memory_gb", 0) < job.get("memory_gb", 0):
            return float("-inf")

        # =========================
        # PERFORMANCE MODEL
        # =========================

        speed_score = 1 / (avg_runtime + 0.1)
        stability_score = 1 - failure_rate

        load_penalty = worker.get("load", 0) * 0.5

        idle_bonus = 2 if worker.get("status") == "idle" else 0
        gpu_bonus = 3 if worker.get("gpu") else 0

        # =========================
        # FINAL SCORE
        # =========================
        score = (
            speed_score * 4.0 +
            stability_score * 3.0 +
            gpu_bonus +
            idle_bonus -
            load_penalty
        )

        return score

    # =========================
    # ROUTE SELECTION
    # =========================
    def select_worker(self, workers, job):

        best_worker = None
        best_score = float("-inf")

        for name, worker in workers.items():

            score = self.score_worker(worker, job, name)

            if score > best_score:
                best_score = score
                best_worker = name

        return best_worker

    # =========================
    # SYSTEM LOAD INSIGHT
    # =========================
    def system_health_bias(self, workers):

        total_load = sum(w.get("load", 0) for w in workers.values())
        avg_load = total_load / max(len(workers), 1)

        if avg_load > 10:
            return "SCALE_UP"

        if avg_load < 3:
            return "SCALE_DOWN"

        return "STABLE"