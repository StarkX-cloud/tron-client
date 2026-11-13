"""Multi-metric measured history for adaptive decisions.

Unlike ``RuntimeHistory`` (runtime means only), this store keeps samples of
throughput, peak memory, success, and recovery so the optimizer can prefer
configs that actually worked — never fabricated defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PerformanceSample:
    key: str
    runtime_seconds: float
    success: bool
    cost_usd: float | None = None
    tokens_per_second: float | None = None
    peak_memory_bytes: int | None = None
    peak_gpu_memory_bytes: int | None = None
    gpu_utilization_percent: float | None = None
    batch_size: int = 1
    policy: str | None = None
    device: str | None = None
    dtype: str | None = None
    target: str | None = None
    recovered: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class PerformanceStore:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else None
        self._samples: list[PerformanceSample] = []
        if self.path and self.path.exists():
            self._load()

    @staticmethod
    def key(
        model_name: str,
        *,
        target: str | None,
        policy: str | None,
        device: str | None,
        dtype: str | None,
        batch_size: int = 1,
    ) -> str:
        return "|".join(
            [
                model_name,
                target or "-",
                policy or "-",
                device or "-",
                dtype or "-",
                f"b{int(batch_size)}",
            ]
        )

    def record(self, sample: PerformanceSample) -> None:
        self._samples.append(sample)
        self._persist()

    def samples_for(self, key_prefix: str | None = None) -> list[PerformanceSample]:
        if not key_prefix:
            return list(self._samples)
        return [s for s in self._samples if s.key.startswith(key_prefix)]

    def best_successful_key(
        self,
        model_name: str,
        *,
        metric: str = "tokens_per_second",
        prefer_lower_runtime: bool = True,
    ) -> str | None:
        """Pick the measured config with best throughput (or lowest runtime)."""
        candidates = [s for s in self._samples if s.success and s.key.startswith(model_name)]
        if not candidates:
            return None
        if metric == "tokens_per_second":
            scored = [s for s in candidates if s.tokens_per_second is not None]
            if scored:
                return max(scored, key=lambda s: float(s.tokens_per_second or 0)).key
        if prefer_lower_runtime:
            return min(candidates, key=lambda s: float(s.runtime_seconds)).key
        return candidates[-1].key

    def mean_runtime(self, key: str) -> float | None:
        xs = [s.runtime_seconds for s in self._samples if s.key == key and s.success]
        return (sum(xs) / len(xs)) if xs else None

    def success_rate(self, key: str) -> float | None:
        xs = [s for s in self._samples if s.key == key]
        if not xs:
            return None
        return sum(1 for s in xs if s.success) / len(xs)

    def to_dict(self) -> dict:
        return {"samples": [s.to_dict() for s in self._samples]}

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        rows = data.get("samples") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return
        loaded: list[PerformanceSample] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            loaded.append(
                PerformanceSample(
                    key=str(r.get("key") or ""),
                    runtime_seconds=float(r.get("runtime_seconds") or 0.0),
                    success=bool(r.get("success")),
                    cost_usd=r.get("cost_usd"),
                    tokens_per_second=r.get("tokens_per_second"),
                    peak_memory_bytes=r.get("peak_memory_bytes"),
                    peak_gpu_memory_bytes=r.get("peak_gpu_memory_bytes"),
                    gpu_utilization_percent=r.get("gpu_utilization_percent"),
                    batch_size=int(r.get("batch_size") or 1),
                    policy=r.get("policy"),
                    device=r.get("device"),
                    dtype=r.get("dtype"),
                    target=r.get("target"),
                    recovered=bool(r.get("recovered")),
                    extra=dict(r.get("extra") or {}),
                )
            )
        self._samples = loaded
