import time
from collections import defaultdict

class ProviderMemory:

    def __init__(self):
        self.stats = defaultdict(list)

    def record(self, provider, latency, cost):

        self.stats[provider].append({
            "latency": latency,
            "cost": cost,
            "time": time.time()
        })

    def score(self, provider):

        data = self.stats.get(provider, [])

        if not data:
            return 1.0

        avg_latency = sum(d["latency"] for d in data) / len(data)
        avg_cost = sum(d["cost"] for d in data) / len(data)

        # lower is better
        return 1 / (avg_latency + avg_cost + 0.0001)