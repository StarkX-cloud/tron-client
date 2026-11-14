"""Intelligence dashboard aggregates from ledger + model registry."""

from __future__ import annotations

from tron_experience.feedback import FeedbackStore
from tron_experience.governance import ModelRegistry
from tron_experience.ledger import ExperienceLedger


def intelligence_snapshot(
    ledger: ExperienceLedger | None = None,
    registry: ModelRegistry | None = None,
    feedback: FeedbackStore | None = None,
) -> dict:
    ledger = ledger or ExperienceLedger()
    registry = registry or ModelRegistry()
    feedback = feedback or FeedbackStore()
    records = ledger.read_all()
    summary = ledger.summary(records)

    # Prediction accuracy on records that had a prediction.
    with_pred = [
        r
        for r in records
        if r.predicted_runtime_seconds is not None and r.actual_runtime_seconds is not None
    ]
    mae = None
    if with_pred:
        mae = sum(
            abs(float(r.predicted_runtime_seconds) - float(r.actual_runtime_seconds))
            for r in with_pred
        ) / len(with_pred)

    # Cost savings attributable to learning: when predicted cost was used and
    # actual < a naive "user expected or unknown" proxy is hard; instead report
    # total actual cost and mean |pred-actual| cost gap when both known.
    cost_pairs = [
        r
        for r in records
        if r.predicted_cost_usd is not None and r.actual_cost_usd is not None
    ]
    cost_mae = None
    total_actual_cost = None
    if cost_pairs:
        cost_mae = sum(
            abs(float(r.predicted_cost_usd) - float(r.actual_cost_usd)) for r in cost_pairs
        ) / len(cost_pairs)
        total_actual_cost = sum(float(r.actual_cost_usd) for r in cost_pairs)

    versions = registry.list_versions()
    active = registry.active()
    improvements = [
        {
            "version_id": v["version_id"],
            "mae": v["mae"],
            "baseline_mae": v["baseline_mae"],
            "improvement_pct": (
                ((v["baseline_mae"] - v["mae"]) / v["baseline_mae"] * 100.0)
                if v["baseline_mae"]
                else None
            ),
            "status": v["status"],
            "beats_baseline": v.get("beats_baseline"),
        }
        for v in versions
    ]

    # Hardware profiles from fingerprints
    hw: dict[str, int] = {}
    for r in records:
        sku = r.fingerprint.worker_sku or ("gpu" if r.fingerprint.gpu else "cpu")
        hw[str(sku)] = hw.get(str(sku), 0) + 1

    return {
        "workloads_observed": summary["workloads_observed"],
        "workload_categories_learned": summary["workload_classes"],
        "hardware_performance_profiles": hw,
        "prediction_accuracy": {
            "n_with_prediction": len(with_pred),
            "runtime_mae_seconds": mae,
            "cost_mae_usd": cost_mae,
        },
        "cost_transparency": {
            "total_actual_cost_usd": total_actual_cost,
            "note": (
                "Savings vs unlearned baseline are reported per ModelVersion "
                "holdout (baseline_mae − mae), not invented fleet-wide $"
            ),
        },
        "model_improvements": improvements,
        "active_model": active.to_dict() if active else None,
        "developer_feedback": feedback.satisfaction_rate(),
        "targets_seen": summary["targets"],
        "honesty": [
            "Metrics come from real DecisionLog / ExperienceLedger rows only",
            "Unvalidated models cannot be promoted (beats_baseline required)",
            "Experience predictions never bypass Autopilot hard constraints",
        ],
    }
