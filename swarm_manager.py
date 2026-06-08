import time


class SwarmManager:

    def __init__(self):

        self.swarm_state = {
            "clusters": {},
            "load_history": []
        }

    # =========================
    # CLUSTER WORKERS
    # =========================

    def build_clusters(self, workers):

        clusters = {
            "low_power": [],
            "balanced": [],
            "high_performance": []
        }

        for name, w in workers.items():

            mem = w.get("memory_gb", 0)
            gpu = w.get("gpu", False)

            if gpu and mem > 8:
                clusters["high_performance"].append(name)

            elif mem > 4:
                clusters["balanced"].append(name)

            else:
                clusters["low_power"].append(name)

        self.swarm_state["clusters"] = clusters

        return clusters

    # =========================
    # SELECT SWARM GROUP
    # =========================

    def select_cluster(self, job):

        if job.get("gpu"):

            return "high_performance"

        if job.get("priority", 1) >= 3:

            return "balanced"

        return "low_power"

    # =========================
    # LOAD AWARE SCALING SIGNAL
    # =========================

    def should_scale(self, queue_size):

        if queue_size > 20:
            return "SCALE_UP"

        if queue_size < 5:
            return "SCALE_DOWN"

        return "STABLE"