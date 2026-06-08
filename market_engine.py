class MarketEngine:

    def clear_market(self, jobs, workers):

        allocations = []

        for job in sorted(jobs, key=lambda j: j.get("priority", 1), reverse=True):

            best_worker = None
            best_score = -999

            for name, w in workers.items():

                capacity_factor = max(0.1, 1 - w.get("load", 0))

                score = (
                    job.get("priority", 1) * 2
                    + capacity_factor * 3
                    - w.get("load", 0)
                )

                if score > best_score:
                    best_score = score
                    best_worker = name

            if best_worker:
                allocations.append((job, best_worker))

        return allocations