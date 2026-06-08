class LoadShaper:

    def reshape(self, job_queue, workers):

        shaped_queue = []

        avg_load = sum(w.get("load", 0) for w in workers.values()) / max(len(workers), 1)

        for job in job_queue:

            urgency = job.get("priority", 1)
            gpu = job.get("gpu", False)

            # congestion pressure
            pressure = avg_load * len(workers)

            delay_factor = 0

            if pressure > 5 and urgency < 3:
                delay_factor = 1  # push lower priority jobs back

            shaped_queue.append({
                "job": job,
                "delay": delay_factor
            })

        return shaped_queue