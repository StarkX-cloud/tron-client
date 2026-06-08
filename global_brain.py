class GlobalDecisionBrain:

    def __init__(self, pricing_engine, market, load_shaper, swarm, simulation_engine):
        self.pricing_engine = pricing_engine
        self.market = market
        self.load_shaper = load_shaper
        self.swarm = swarm
        self.simulation_engine = simulation_engine

        # learning memory
        self.worker_scores = {}
        self.routing_bias = 1.0

    # =========================
    # DECISION ENGINE (YOU ALREADY USE THIS)
    # =========================
    def decide(self, job_queue, workers, job):

        score = job.get("priority", 1)

        return {
            "score": score
        }

    # =========================
    # 🧠 LEARNING LOOP (NEW CORE)
    # =========================
    def learn(self, job, result, runtime, worker_name):

        # initialize worker tracking
        if worker_name not in self.worker_scores:
            self.worker_scores[worker_name] = {
                "jobs": 0,
                "avg_runtime": 0,
                "avg_cost_error": 0
            }

        stats = self.worker_scores[worker_name]

        # update stats
        stats["jobs"] += 1

        # exponential moving average
        alpha = 0.2

        stats["avg_runtime"] = (
            (1 - alpha) * stats["avg_runtime"]
            + alpha * runtime
        )

        # cost efficiency signal
        expected = job.get("memory_gb", 1) * 0.01
        actual = result.get("cost", expected)

        cost_error = actual - expected

        stats["avg_cost_error"] = (
            (1 - alpha) * stats["avg_cost_error"]
            + alpha * cost_error
        )

        # adjust routing bias
        if runtime > 5:
            self.routing_bias *= 0.99
        else:
            self.routing_bias *= 1.01