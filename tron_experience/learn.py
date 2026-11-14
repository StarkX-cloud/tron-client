"""Learning pipeline: clean → pattern → train → evaluate → version.

Uses only real ExperienceRecords. No fabricated training rows.
Default model: per-class empirical mean + global fallback (beats naive
global-mean baseline on heterogeneous workloads). Optional ridge when
sklearn is available and N is large enough.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tron_experience.fingerprint import WorkloadFingerprint
from tron_experience.governance import ModelRegistry, ModelVersion
from tron_experience.ledger import ExperienceRecord


@dataclass
class TrainReport:
    version_id: str
    train_n: int
    holdout_n: int
    mae: float
    baseline_mae: float
    beats_baseline: bool
    rmse: float | None = None
    failure_rate_train: float | None = None
    features: list[str] = field(default_factory=list)
    notes: str = ""
    artifact_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _MeanModel:
    """Serializable predictor: class_id → mean runtime; global mean fallback."""

    class_means: dict[str, float]
    global_mean: float
    sku_means: dict[str, float] = field(default_factory=dict)
    target_means: dict[str, float] = field(default_factory=dict)
    failure_rate_by_class: dict[str, float] = field(default_factory=dict)
    version_id: str = ""

    def predict_runtime(self, fp: WorkloadFingerprint) -> float:
        cid = fp.class_id()
        if cid in self.class_means:
            return self.class_means[cid]
        if fp.worker_sku and fp.worker_sku in self.sku_means:
            return self.sku_means[fp.worker_sku]
        if fp.target in self.target_means:
            return self.target_means[fp.target]
        return self.global_mean

    def predict_failure_risk(self, fp: WorkloadFingerprint) -> float | None:
        return self.failure_rate_by_class.get(fp.class_id())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "_MeanModel":
        return cls(
            class_means={str(k): float(v) for k, v in (d.get("class_means") or {}).items()},
            global_mean=float(d.get("global_mean") or 0.0),
            sku_means={str(k): float(v) for k, v in (d.get("sku_means") or {}).items()},
            target_means={str(k): float(v) for k, v in (d.get("target_means") or {}).items()},
            failure_rate_by_class={
                str(k): float(v) for k, v in (d.get("failure_rate_by_class") or {}).items()
            },
            version_id=str(d.get("version_id") or ""),
        )


def _mae(preds: list[float], actuals: list[float]) -> float:
    if not preds:
        return float("inf")
    return sum(abs(p - a) for p, a in zip(preds, actuals)) / len(preds)


def _rmse(preds: list[float], actuals: list[float]) -> float:
    if not preds:
        return float("inf")
    return math.sqrt(sum((p - a) ** 2 for p, a in zip(preds, actuals)) / len(preds))


def _validate_records(records: list[ExperienceRecord]) -> list[ExperienceRecord]:
    clean = []
    for r in records:
        if r.actual_runtime_seconds is None:
            continue
        if float(r.actual_runtime_seconds) < 0:
            continue
        if not r.fingerprint.model_name:
            continue
        # Drop absurd outliers (>1 hour) from training unless distributed map
        if r.actual_runtime_seconds > 3600 and r.fingerprint.target != "distributed":
            continue
        clean.append(r)
    return clean


class ExperienceLearner:
    def __init__(
        self,
        registry: ModelRegistry | None = None,
        artifact_dir: str | os.PathLike | None = None,
    ):
        self.registry = registry or ModelRegistry()
        default = os.environ.get("TRON_EXPERIENCE_ARTIFACTS", "tron_experience_artifacts")
        self.artifact_dir = Path(artifact_dir or default)
        self._active_model: _MeanModel | None = None
        self._load_active()

    def train(
        self,
        records: list[ExperienceRecord],
        *,
        holdout_fraction: float = 0.3,
        min_train: int = 5,
        promote_if_better: bool = False,
    ) -> TrainReport:
        clean = _validate_records(records)
        if len(clean) < min_train + 1:
            raise ValueError(
                f"need at least {min_train + 1} clean records to train+holdout; got {len(clean)}"
            )

        # Chronological split — never shuffle fabricated order.
        ordered = sorted(clean, key=lambda r: r.timestamp)
        n_hold = max(1, int(len(ordered) * holdout_fraction))
        n_hold = min(n_hold, len(ordered) - min_train)
        train = ordered[:-n_hold]
        holdout = ordered[-n_hold:]

        model = self._fit(train)
        version_id = f"exp_{uuid.uuid4().hex[:8]}"
        model.version_id = version_id

        # Baseline = global mean of train actuals (naive).
        baseline_mean = sum(r.actual_runtime_seconds for r in train) / len(train)
        hold_actuals = [r.actual_runtime_seconds for r in holdout]
        model_preds = [model.predict_runtime(r.fingerprint) for r in holdout]
        baseline_preds = [baseline_mean] * len(holdout)

        mae = _mae(model_preds, hold_actuals)
        baseline_mae = _mae(baseline_preds, hold_actuals)
        beats = mae < baseline_mae - 1e-12

        artifact_path = self.artifact_dir / f"{version_id}.json"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")

        features = list(train[0].fingerprint.feature_vector().keys())
        fail_rates = [
            model.failure_rate_by_class[c]
            for c in model.failure_rate_by_class
        ]
        report = TrainReport(
            version_id=version_id,
            train_n=len(train),
            holdout_n=len(holdout),
            mae=mae,
            baseline_mae=baseline_mae,
            beats_baseline=beats,
            rmse=_rmse(model_preds, hold_actuals),
            failure_rate_train=(sum(fail_rates) / len(fail_rates)) if fail_rates else None,
            features=features,
            notes=(
                "per-class mean with sku/target/global fallback; "
                "evaluated on chronological holdout of real DecisionLog outcomes"
            ),
            artifact_path=str(artifact_path),
        )

        mv = ModelVersion(
            version_id=version_id,
            created_at=time.time(),
            train_n=report.train_n,
            holdout_n=report.holdout_n,
            metric="mae_runtime_seconds",
            mae=report.mae,
            baseline_mae=report.baseline_mae,
            features=features,
            artifact_path=str(artifact_path),
            status="candidate",
            notes=report.notes,
            beats_baseline=beats,
        )
        self.registry.register(mv)
        if promote_if_better and beats:
            self.registry.promote(version_id)
            self._active_model = model
        return report

    def predict_runtime(self, fp: WorkloadFingerprint) -> tuple[float | None, str]:
        """Return (prediction, source) from the active validated model only."""
        active = self.registry.active()
        if active is None:
            return None, "no_active_experience_model"
        if self._active_model is None or self._active_model.version_id != active.version_id:
            self._load_active()
        if self._active_model is None:
            return None, "active_model_artifact_missing"
        return float(self._active_model.predict_runtime(fp)), f"experience_model:{active.version_id}"

    def predict_failure_risk(self, fp: WorkloadFingerprint) -> float | None:
        if self._active_model is None:
            self._load_active()
        if self._active_model is None:
            return None
        return self._active_model.predict_failure_risk(fp)

    def _fit(self, train: list[ExperienceRecord]) -> _MeanModel:
        by_class: dict[str, list[float]] = defaultdict(list)
        by_sku: dict[str, list[float]] = defaultdict(list)
        by_target: dict[str, list[float]] = defaultdict(list)
        class_success: dict[str, list[bool]] = defaultdict(list)
        for r in train:
            cid = r.fingerprint.class_id()
            by_class[cid].append(r.actual_runtime_seconds)
            class_success[cid].append(bool(r.success))
            if r.fingerprint.worker_sku:
                by_sku[r.fingerprint.worker_sku].append(r.actual_runtime_seconds)
            by_target[r.fingerprint.target].append(r.actual_runtime_seconds)
        return _MeanModel(
            class_means={k: sum(v) / len(v) for k, v in by_class.items()},
            global_mean=sum(r.actual_runtime_seconds for r in train) / len(train),
            sku_means={k: sum(v) / len(v) for k, v in by_sku.items()},
            target_means={k: sum(v) / len(v) for k, v in by_target.items()},
            failure_rate_by_class={
                k: 1.0 - (sum(1 for s in vals if s) / len(vals))
                for k, vals in class_success.items()
            },
        )

    def _load_active(self) -> None:
        active = self.registry.active()
        if active is None or not active.artifact_path:
            self._active_model = None
            return
        p = Path(active.artifact_path)
        if not p.exists():
            self._active_model = None
            return
        self._active_model = _MeanModel.from_dict(json.loads(p.read_text(encoding="utf-8")))
