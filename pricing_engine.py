class PricingEngine:

    def compute_price(self, job_queue, workers, job):

        base = 0.01

        # system pressure
        pressure = len(job_queue) / max(len(workers), 1)

        # worker scarcity
        avg_load = sum(w.get("load", 0) for w in workers.values()) / max(len(workers), 1)

        # gpu premium
        gpu_factor = 2.0 if job.get("gpu") else 1.0

        # priority boost
        priority = job.get("priority", 1)

        price = (
            base
            + pressure * 0.01
            + avg_load * 0.02
            + priority * 0.005
        ) * gpu_factor

        return round(price, 6)