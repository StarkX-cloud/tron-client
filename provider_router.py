class ProviderRouter:

    def __init__(self, memory, latency_predictor):

        self.memory = memory
        self.latency_predictor = latency_predictor
        self.providers = ["openai", "claude", "gemini", "local"]

    # =========================
    # SIMULATION LAYER
    # =========================
    def simulate(self, provider, prompt, cost_hint=None):

        # predicted latency
        latency = self.latency_predictor.predict(provider, prompt)

        # learned efficiency score
        efficiency = self.memory.score(provider)

        # cost penalty (simple heuristic)
        cost_penalty = 0.0

        if cost_hint:
            cost_penalty = cost_hint

        # complexity penalty
        complexity = len(prompt) / 1000

        # final simulated score (higher is better)
        score = efficiency - (latency * 0.2) - cost_penalty - (complexity * 0.1)

        return {
            "provider": provider,
            "score": score,
            "predicted_latency": latency
        }

    # =========================
    # SELECT BEST PROVIDER
    # =========================
    def select(self, prompt, cost_hint=None):

        best = None
        best_sim = None

        for p in self.providers:

            sim = self.simulate(p, prompt, cost_hint)

            if best is None or sim["score"] > best_sim["score"]:
                best = p
                best_sim = sim

        return best