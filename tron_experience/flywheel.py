"""TRON company intelligence flywheel — measurable operating hierarchy.

The machines execute the work. The data teaches TRON how to execute it
better. This module does not invent growth metrics: every stage is scored
from real ExperienceLedger / ModelRegistry / FeedbackStore / fleet signals.

Hierarchy:
  1. Acquire computational workloads
  2. Accumulate high-quality execution intelligence
  3. Convert experience into better algorithms
  4. Improve compute economics
  5. Attract more workloads and infrastructure
  6. Compound the intelligence advantage
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from tron_experience.feedback import FeedbackStore
from tron_experience.governance import ModelRegistry
from tron_experience.ledger import ExperienceLedger, ExperienceRecord
from tron_experience.stats import intelligence_snapshot


STAGES = (
    ("acquire", "Acquire computational workloads"),
    ("accumulate", "Accumulate high-quality execution intelligence"),
    ("convert", "Convert experience into better algorithms"),
    ("economics", "Improve compute economics"),
    ("attract", "Attract more workloads and infrastructure"),
    ("compound", "Compound the intelligence advantage"),
)


@dataclass
class StageScore:
    stage_id: str
    title: str
    score: float | None  # 0..1 when measurable, else None
    status: str  # strong | building | blocked | unknown
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _quality_rate(records: list[ExperienceRecord]) -> float | None:
    if not records:
        return None
    # High-quality = success + measured actual runtime present
    good = sum(
        1
        for r in records
        if r.success and r.actual_runtime_seconds is not None and r.actual_runtime_seconds >= 0
    )
    return good / len(records)


def _prediction_coverage(records: list[ExperienceRecord]) -> float | None:
    if not records:
        return None
    with_pred = sum(1 for r in records if r.predicted_runtime_seconds is not None)
    return with_pred / len(records)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def score_flywheel(
    ledger: ExperienceLedger | None = None,
    registry: ModelRegistry | None = None,
    feedback: FeedbackStore | None = None,
    *,
    fleet: dict | None = None,
) -> dict[str, Any]:
    """Return the six-stage flywheel with measured KPIs from real stores."""
    ledger = ledger or ExperienceLedger()
    registry = registry or ModelRegistry()
    feedback = feedback or FeedbackStore()
    records = ledger.read_all()
    snap = intelligence_snapshot(ledger=ledger, registry=registry, feedback=feedback)
    versions = registry.list_versions()
    active = registry.active()
    fb = feedback.satisfaction_rate()

    n = len(records)
    classes = snap.get("workload_categories_learned") or 0
    quality = _quality_rate(records)
    coverage = _prediction_coverage(records)

    # Stage 1 — acquire
    acquire_score = _clamp01(n / 50.0) if n else 0.0
    acquire = StageScore(
        stage_id="acquire",
        title=STAGES[0][1],
        score=acquire_score,
        status="strong" if n >= 20 else ("building" if n > 0 else "blocked"),
        metrics={
            "workloads_observed": n,
            "targets": snap.get("targets_seen") or [],
        },
        evidence=[
            "Counted from ExperienceLedger (DecisionLog outcomes), not estimates",
        ],
    )

    # Stage 2 — accumulate
    accumulate_score = None
    if quality is not None:
        accumulate_score = _clamp01(0.6 * quality + 0.4 * min(1.0, classes / 10.0))
    accumulate = StageScore(
        stage_id="accumulate",
        title=STAGES[1][1],
        score=accumulate_score,
        status=(
            "strong"
            if (quality or 0) >= 0.8 and classes >= 2
            else ("building" if n else "blocked")
        ),
        metrics={
            "high_quality_rate": quality,
            "workload_classes": classes,
            "hardware_profiles": snap.get("hardware_performance_profiles") or {},
        },
        evidence=[
            "Fingerprints exclude prompts/code/model I/O (governance)",
            "Quality = successful executions with measured runtime",
        ],
    )

    # Stage 3 — convert
    best_improve = None
    for v in versions:
        pct = None
        if v.get("baseline_mae"):
            pct = (v["baseline_mae"] - v["mae"]) / v["baseline_mae"]
        if pct is not None:
            best_improve = max(best_improve or 0.0, pct)
    convert_score = None
    if active is not None and active.beats_baseline:
        convert_score = _clamp01(0.5 + 0.5 * (best_improve or 0.0))
    elif versions:
        convert_score = 0.25
    else:
        convert_score = 0.0 if n else None
    convert = StageScore(
        stage_id="convert",
        title=STAGES[2][1],
        score=convert_score,
        status=(
            "strong"
            if active and active.beats_baseline
            else ("building" if versions else "blocked")
        ),
        metrics={
            "active_model": active.version_id if active else None,
            "model_versions": len(versions),
            "best_holdout_improvement_pct": (
                (best_improve * 100.0) if best_improve is not None else None
            ),
            "runtime_mae_seconds": (snap.get("prediction_accuracy") or {}).get(
                "runtime_mae_seconds"
            ),
        },
        evidence=[
            "Models promote only when holdout MAE beats naive baseline",
            "No fabricated training rows",
        ],
    )

    # Stage 4 — economics
    cost = snap.get("cost_transparency") or {}
    pred_acc = snap.get("prediction_accuracy") or {}
    cost_mae = pred_acc.get("cost_mae_usd")
    econ_score = None
    if best_improve is not None:
        # Better runtime prediction → tighter cost estimates (same hourly rate).
        econ_score = _clamp01(best_improve)
    elif cost_mae is not None:
        econ_score = _clamp01(1.0 / (1.0 + float(cost_mae) * 1000.0))
    economics = StageScore(
        stage_id="economics",
        title=STAGES[3][1],
        score=econ_score if econ_score is not None else (0.0 if n else None),
        status="strong" if (best_improve or 0) > 0.2 else ("building" if n else "blocked"),
        metrics={
            "total_actual_cost_usd": cost.get("total_actual_cost_usd"),
            "cost_mae_usd": cost_mae,
            "holdout_prediction_improvement_pct": (
                (best_improve * 100.0) if best_improve is not None else None
            ),
            "note": cost.get("note"),
        },
        evidence=[
            "Economic gains attributed only to measured holdout improvement",
            "Not a fabricated fleet-wide $ savings claim",
        ],
    )

    # Stage 5 — attract (workloads + infrastructure)
    alive = None
    providers = None
    if isinstance(fleet, dict):
        alive = fleet.get("alive_count")
        providers = len(fleet.get("workers") or [])
    attract_parts = [acquire_score]
    if alive is not None:
        attract_parts.append(_clamp01(alive / 5.0))
    if fb.get("met_requirements_rate") is not None:
        attract_parts.append(float(fb["met_requirements_rate"]))
    attract_score = sum(attract_parts) / len(attract_parts) if attract_parts else None
    attract = StageScore(
        stage_id="attract",
        title=STAGES[4][1],
        score=attract_score,
        status=(
            "strong"
            if (alive or 0) >= 2 and n >= 10
            else ("building" if n or alive else "blocked")
        ),
        metrics={
            "workloads_observed": n,
            "fleet_alive_count": alive,
            "fleet_workers": providers,
            "developer_satisfaction": fb,
        },
        evidence=[
            "Fleet counts from live capacity when provided",
            "Developer feedback is opt-in structured ratings",
        ],
    )

    # Stage 6 — compound
    # Compounding = active model + growing classes + improving MAE trend
    prior = [v for v in versions if v.get("status") in ("active", "retired", "candidate")]
    prior_sorted = sorted(prior, key=lambda v: v.get("created_at") or 0)
    mae_trend = None
    if len(prior_sorted) >= 2:
        mae_trend = prior_sorted[-2]["mae"] - prior_sorted[-1]["mae"]  # positive = better
    compound_score = 0.0
    if active and active.beats_baseline:
        compound_score += 0.4
    if classes >= 2:
        compound_score += 0.2
    if (mae_trend or 0) > 0:
        compound_score += 0.2
    if (quality or 0) >= 0.8:
        compound_score += 0.2
    compound = StageScore(
        stage_id="compound",
        title=STAGES[5][1],
        score=_clamp01(compound_score) if n else None,
        status=(
            "strong"
            if compound_score >= 0.7
            else ("building" if n else "blocked")
        ),
        metrics={
            "active_beats_baseline": bool(active and active.beats_baseline),
            "workload_classes": classes,
            "mae_improvement_vs_previous_model": mae_trend,
            "prediction_coverage": coverage,
        },
        evidence=[
            "Advantage compounds only when new models keep beating prior baselines",
            "More classes + higher quality data → stronger next training round",
        ],
    )

    stages = [acquire, accumulate, convert, economics, attract, compound]
    scored = [s.score for s in stages if s.score is not None]
    return {
        "thesis": (
            "The machines execute the work. The data teaches TRON how to "
            "execute it better. Execution intelligence is the compounding moat."
        ),
        "hierarchy": [s.to_dict() for s in stages],
        "flywheel_health": (sum(scored) / len(scored)) if scored else None,
        "moat": {
            "what_we_collect": [
                "Workload fingerprints (structural, no prompts/code/I/O)",
                "Predicted vs actual runtime/cost/memory",
                "Placement decisions and recovery outcomes",
                "Hardware/capacity signals and developer feedback",
            ],
            "what_we_never_train_on_without_opt_in": [
                "Customer prompts",
                "Source code / cloudpickle payloads",
                "Model inputs and outputs",
            ],
            "why_it_compounds": (
                "Better predictions → cheaper/faster placements → more "
                "workloads and providers → more execution data → better "
                "algorithms"
            ),
        },
        "honesty": [
            "Stage scores are derived from real ledger/model/feedback/fleet data",
            "No fabricated training rows or artificial growth curves",
            "Learning never bypasses Autopilot hard constraints",
        ],
    }
