class LatencyPredictor:

    def __init__(self):

        self.samples = []

    def record(self, provider, latency):

        self.samples.append({
            "provider": provider,
            "latency": latency
        })

    def predict(self, provider, prompt):

        base = 2.0

        if len(prompt) > 2000:
            base += 2.5

        if provider == "gemini":
            base += 1.0

        if provider == "local":
            base += 0.5

        # simple learned adjustment
        history = [
            s["latency"]
            for s in self.samples
            if s["provider"] == provider
        ]

        if history:
            base = (base + sum(history[-5:]) / len(history[-5:])) / 2

        return base