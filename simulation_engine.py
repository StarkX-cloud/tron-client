class SimulationEngine:

    def simulate(self, workers, job):
        results = []

        for worker_name, worker in workers.items():

            predicted_runtime = self.predict_runtime(worker, job)
            predicted_cost = self.predict_cost(worker, job)
            success_prob = self.predict_success(worker, job)

            score = (
                success_prob * 3
                - predicted_runtime * 1.5
                - predicted_cost
            )

            results.append({
                "worker": worker_name,
                "score": score,
                "runtime": predicted_runtime,
                "cost": predicted_cost,
                "success": success_prob
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def predict_runtime(self, worker, job):
        base = job.get("memory_gb", 2)
        gpu_factor = 0.5 if worker.get("gpu") else 1.5
        load_factor = worker.get("load", 1)

        return base * gpu_factor * load_factor

    def predict_cost(self, worker, job):
        return 0.01 + job.get("priority", 1) * 0.002

    def predict_success(self, worker, job):
        load = worker.get("load", 1)
        return max(0.2, 1.0 - (load * 0.1))