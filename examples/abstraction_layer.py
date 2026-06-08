class AbstractionLayerV1:

    def __init__(self, cost_engine=None):

        self.cost_engine = cost_engine

    # =========================
    # MAIN ENTRY
    # =========================

    def process(self, job: dict):

        job = self.normalize_intent(job)
        job = self.add_routing_hints(job)
        job = self.add_cost_preview(job)
        job = self.add_execution_profile(job)

        return job

    # =========================
    # 1. INTENT NORMALIZATION
    # =========================

    def normalize_intent(self, job):

        task = job.get("task_type", "unknown")

        # intelligent overrides
        if "train" in task.lower():
            job["task_type"] = "training"

        if "embed" in task.lower():
            job["task_type"] = "embedding"

        if "agent" in task.lower():
            job["task_type"] = "agent"

        # default safety constraints
        job.setdefault("priority", 1)
        job.setdefault("gpu", False)
        job.setdefault("memory_gb", 2)

        return job

    # =========================
    # 2. ROUTING HINTS
    # =========================

    def add_routing_hints(self, job):

        job["routing"] = {
            "latency_sensitivity": "high" if job["priority"] > 2 else "normal",
            "preferred_worker": "gpu" if job.get("gpu") else "cpu",
            "batch_eligible": job.get("priority", 1) <= 2
        }

        return job

    # =========================
    # 3. COST PREVIEW
    # =========================

    def add_cost_preview(self, job):

        estimated = self.cost_engine(job, runtime=1.0)

        job["estimated_cost"] = estimated

        job["billing"] = {
            "precheck": True,
            "estimated": estimated
        }

        return job

    # =========================
    # 4. EXECUTION PROFILE
    # =========================

    def add_execution_profile(self, job):

        job["execution_profile"] = {
            "is_gpu_heavy": job.get("gpu", False),
            "complexity_score":
                job.get("memory_gb", 1) * 2 +
                job.get("priority", 1) * 3,
            "category": job.get("task_type")
        }

        return job
