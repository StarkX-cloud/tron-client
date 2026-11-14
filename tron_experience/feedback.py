"""Developer feedback linked to workload execution records."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DeveloperFeedback:
    plan_id: str
    met_requirements: bool
    rating: int | None = None  # 1-5 optional
    notes: str | None = None
    customer_id: str | None = None
    tenant_id: str | None = None
    categories: list[str] = field(default_factory=list)  # e.g. ["latency","cost","correctness"]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class FeedbackStore:
    def __init__(self, path: str | os.PathLike | None = None):
        default = os.environ.get("TRON_EXPERIENCE_FEEDBACK", "tron_experience_feedback.jsonl")
        self.path = Path(path or default)

    def submit(self, feedback: DeveloperFeedback) -> None:
        if feedback.rating is not None and not (1 <= int(feedback.rating) <= 5):
            raise ValueError("rating must be 1..5 when provided")
        # Never accept free-form dumps of prompts/code — notes capped.
        if feedback.notes and len(feedback.notes) > 2000:
            feedback.notes = feedback.notes[:2000]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback.to_dict(), default=str) + "\n")

    def for_plan(self, plan_id: str) -> list[dict]:
        return [r for r in self.read_all() if r.get("plan_id") == plan_id]

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def satisfaction_rate(self, customer_id: str | None = None) -> dict[str, Any]:
        rows = self.read_all()
        if customer_id:
            rows = [r for r in rows if r.get("customer_id") == customer_id]
        if not rows:
            return {"n": 0, "met_requirements_rate": None, "avg_rating": None}
        met = sum(1 for r in rows if r.get("met_requirements"))
        ratings = [int(r["rating"]) for r in rows if r.get("rating") is not None]
        return {
            "n": len(rows),
            "met_requirements_rate": met / len(rows),
            "avg_rating": (sum(ratings) / len(ratings)) if ratings else None,
        }
