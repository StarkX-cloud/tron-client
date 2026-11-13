"""Append-only JSONL logs for plans, outcomes, and measured history.

Plans and outcomes are the audit trail ("why did TRON place this?").
History aggregates successful runs so later predictions use **measured**
runtimes for the same (model, target, policy, device, dtype) key — never
a fabricated default when nothing has been measured yet.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Iterable

from tron_autopilot.types import ExecutionOutcome, WorkloadPlan


def new_plan_id() -> str:
    return f"plan_{uuid.uuid4().hex[:12]}"


class DecisionLog:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def append_plan(self, plan: WorkloadPlan) -> None:
        self._append({"kind": "plan", **plan.to_dict()})

    def append_outcome(self, outcome: ExecutionOutcome) -> None:
        self._append({"kind": "outcome", **outcome.to_dict()})

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

    def _append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")


class RuntimeHistory:
    """Mean measured runtime for successful executions, keyed by config."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else None
        self._samples: dict[str, list[float]] = {}
        if self.path and self.path.exists():
            self._load()

    @staticmethod
    def key(
        model_name: str,
        target: str,
        policy: str | None,
        device: str | None,
        dtype: str | None,
    ) -> str:
        return "|".join(
            [
                model_name,
                target,
                policy or "-",
                device or "-",
                dtype or "-",
            ]
        )

    def record(self, key: str, runtime_seconds: float) -> None:
        self._samples.setdefault(key, []).append(float(runtime_seconds))
        self._persist()

    def mean_runtime(self, key: str) -> float | None:
        samples = self._samples.get(key) or []
        if not samples:
            return None
        return sum(samples) / len(samples)

    def ingest_outcomes(self, outcomes: Iterable[ExecutionOutcome]) -> None:
        for o in outcomes:
            if not o.success:
                continue
            # Outcomes alone don't always carry policy/device; callers that
            # know the plan should call record() with the full key.
            k = self.key(o.plan_id, o.target, None, None, None)
            # Intentionally not used as a primary path — see record().
            _ = k

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: list(v) for k, v in self._samples.items()}
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict):
            self._samples = {
                str(k): [float(x) for x in v]
                for k, v in data.items()
                if isinstance(v, list)
            }


def cost_from_runtime(hourly_rate_usd: float, runtime_seconds: float) -> float:
    return (hourly_rate_usd / 3600.0) * float(runtime_seconds)


def now() -> float:
    return time.time()
