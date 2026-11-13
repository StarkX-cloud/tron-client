"""Per-provider / per-worker performance history from measured outcomes.

Append-only JSONL. Used for availability/reliability reporting — not a
fabricated reputation score.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProviderOutcome:
    worker_name: str
    provider_id: str | None
    success: bool
    runtime_seconds: float
    cost_usd: float | None = None
    requeue_count: float = 0.0
    target: str | None = None
    customer_id: str | None = None
    error: str | None = None
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ProviderStatsStore:
    def __init__(self, path: str | os.PathLike | None = None):
        default = os.environ.get("TRON_PROVIDER_STATS", "tron_provider_stats.jsonl")
        self.path = Path(path or default)

    def record(self, outcome: ProviderOutcome) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(outcome.to_dict(), default=str) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def summary(self, worker_name: str | None = None, provider_id: str | None = None) -> dict:
        rows = self.read_all()
        if worker_name:
            rows = [r for r in rows if r.get("worker_name") == worker_name]
        if provider_id:
            rows = [r for r in rows if r.get("provider_id") == provider_id]
        if not rows:
            return {
                "n": 0,
                "successes": 0,
                "failures": 0,
                "success_rate": None,
                "avg_runtime_seconds": None,
                "avg_cost_usd": None,
                "avg_requeue_count": None,
            }
        successes = [r for r in rows if r.get("success")]
        failures = [r for r in rows if not r.get("success")]
        rts = [float(r["runtime_seconds"]) for r in rows if r.get("runtime_seconds") is not None]
        costs = [
            float(r["cost_usd"])
            for r in successes
            if r.get("cost_usd") is not None
        ]
        rqs = [float(r.get("requeue_count") or 0) for r in rows]
        return {
            "n": len(rows),
            "successes": len(successes),
            "failures": len(failures),
            "success_rate": len(successes) / len(rows),
            "avg_runtime_seconds": (sum(rts) / len(rts)) if rts else None,
            "avg_cost_usd": (sum(costs) / len(costs)) if costs else None,
            "avg_requeue_count": (sum(rqs) / len(rqs)) if rqs else 0.0,
            "worker_name": worker_name,
            "provider_id": provider_id,
        }
