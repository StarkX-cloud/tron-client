import time


class PredictorEngine:

    def __init__(self):
        self.history = []

    # =========================
    # RECORD JOB DATA
    # =========================
    def record(self, job):

        self.history.append({
            "time": time.time(),
            "memory": job.get("memory_gb", 1),
            "gpu": job.get("gpu", False),
            "priority": job.get("priority", 1),
            "runtime": job.get("runtime", None),
            "cost": job.get("cost", None)
        })

    # =========================
    # PREDICT LOAD SPIKE
    # =========================
    def predict_load(self):

        if len(self.history) < 5:
            return "LOW"

        recent = self.history[-10:]

        avg_memory = sum(h["memory"] for h in recent) / len(recent)

        if avg_memory > 6:
            return "HIGH"

        if avg_memory > 3:
            return "MEDIUM"

        return "LOW"

    # =========================
    # PREDICT GPU PRESSURE
    # =========================
    def gpu_pressure(self):

        gpu_jobs = [
            h for h in self.history
            if h["gpu"]
        ]

        if len(gpu_jobs) > 5:
            return True

        return False

    # =========================
    # 🧠 NEW: RUNTIME PREDICTION
    # =========================
    def predict_runtime(self, job):

        base = 2.0

        # GPU increases runtime complexity
        if job.get("gpu"):
            base += 3.0

        # memory scales linearly
        base += job.get("memory_gb", 1) * 0.6

        # priority reduces latency (higher priority = faster handling)
        priority = job.get("priority", 1)
        base -= (priority * 0.3)

        # historical adaptation (learn from past)
        if len(self.history) > 5:
            avg_runtime = sum(
                h["runtime"] for h in self.history
                if h.get("runtime") is not None
            ) or 2.0

            base = (base * 0.7) + (avg_runtime * 0.3)

        return max(0.5, base)

    # =========================
    # 🧠 NEW: COST PREDICTION
    # =========================
    def predict_cost(self, job):

        runtime = self.predict_runtime(job)

        base_cost = 0.01 if job.get("gpu") else 0.002
        memory_cost = job.get("memory_gb", 1) * 0.001
        priority_cost = job.get("priority", 1) * 0.002
        time_cost = runtime * 0.001

        return round(base_cost + memory_cost + priority_cost + time_cost, 6)

    # =========================
    # 🧠 NEW: BEST WORKER PREDICTION
    # =========================
    def predict_best_worker(self, workers, job):

        best_worker = None
        best_score = -999

        for name, w in workers.items():

            score = 0

            # GPU match
            if job.get("gpu") and w.get("gpu"):
                score += 3

            # memory compatibility
            score -= abs(w.get("load", 0) - job.get("memory_gb", 1)) * 0.2

            # idle bonus
            if w.get("status") == "idle":
                score += 2

            # historical GPU pressure awareness
            if self.gpu_pressure() and not w.get("gpu"):
                score -= 2

            if score > best_score:
                best_score = score
                best_worker = name

        return best_worker