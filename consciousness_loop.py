import time


class ConsciousnessLoop:

    def __init__(self, predictor, router, optimizer, memory_mesh):

        self.predictor = predictor
        self.router = router
        self.optimizer = optimizer
        self.memory = memory_mesh

        self.learning_rate = 0.05

    # =========================
    # OBSERVE SYSTEM OUTCOME
    # =========================
    def observe(self, job, runtime, cost, worker):

        signal = {
            "memory": job.get("memory_gb", 1),
            "gpu": job.get("gpu", False),
            "runtime": runtime,
            "cost": cost,
            "worker": worker,
            "time": time.time()
        }

        self.memory.write(
            "consciousness_log",
            signal,
            source=job.get("id")
        )

        return signal

    # =========================
    # ADAPT PREDICTOR WEIGHTS
    # =========================
    def adapt_predictor(self, signal):

        if signal["runtime"] > 5:
            self.predictor.history.append({
                "penalty": "slow_execution"
            })

        if signal["cost"] > 0.02:
            self.predictor.history.append({
                "penalty": "high_cost"
            })

    # =========================
    # ADAPT ROUTING BEHAVIOR
    # =========================
    def adapt_router(self, signal):

        if signal["runtime"] > 5:

            self.router.bias = getattr(self.router, "bias", 1.0) + self.learning_rate

        if signal["worker"] is None:

            self.router.bias = getattr(self.router, "bias", 1.0) - self.learning_rate

    # =========================
    # FULL LEARNING CYCLE
    # =========================
    def learn(self, job, runtime, cost, worker):

        signal = self.observe(job, runtime, cost, worker)

        self.adapt_predictor(signal)
        self.adapt_router(signal)

        return {
            "status": "learned",
            "signal": signal
        }