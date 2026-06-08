import copy
import time


class GlobalOptimizer:

    def __init__(self):
        self.history = []

    # =========================
    # SIMULATE SYSTEM STATE
    # =========================
    def simulate(self, job_queue, workers, router, predictor):

        simulated_workers = copy.deepcopy(workers)
        simulated_queue = copy.deepcopy(job_queue)

        score = 0

        for job in simulated_queue:

            worker = router.select_worker(simulated_workers, job)

            if not worker:
                continue

            load = simulated_workers[worker].get("load", 0)
            mem = job.get("memory_gb", 1)

            # simulate assignment
            simulated_workers[worker]["load"] = load + mem

            # system pressure penalty
            score -= simulated_workers[worker]["load"] * 0.2

            # prediction bonus
            if predictor.predict_load() == "HIGH":
                score -= 2
            elif predictor.predict_load() == "LOW":
                score += 1

        return score

    # =========================
    # DECIDE SYSTEM STRATEGY
    # =========================
    def decide_strategy(self, job_queue, workers, router, predictor):

        base_score = self.simulate(job_queue, workers, router, predictor)

        # try alternative routing bias
        alt_score = base_score * 1.1  # placeholder for future policies

        if alt_score > base_score:
            return {
                "mode": "AGGRESSIVE_OPTIMIZATION",
                "score": alt_score
            }

        return {
            "mode": "STABLE",
            "score": base_score
        }