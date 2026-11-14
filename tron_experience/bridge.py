"""Wire Experience Engine predictions into Economic Autopilot — soft only.

Predictions may fill missing runtime estimates. They never mark candidates
feasible, skip reliability/memory/budget gates, or override security.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tron_experience.fingerprint import WorkloadFingerprint
from tron_experience.learn import ExperienceLearner

if TYPE_CHECKING:
    from tron_autopilot.types import WorkloadSpec

_LEARNER: ExperienceLearner | None = None


def experience_enabled() -> bool:
    return os.environ.get("TRON_EXPERIENCE_ENABLED", "").lower() in ("1", "true", "yes")


def get_learner() -> ExperienceLearner:
    global _LEARNER
    if _LEARNER is None:
        _LEARNER = ExperienceLearner()
    return _LEARNER


def predict_runtime_for_spec(
    spec: "WorkloadSpec",
    target: str,
    policy: str | None,
    device: str | None,
    dtype: str | None,
) -> tuple[float | None, str]:
    """Soft prediction used only when history mean is unavailable."""
    if not experience_enabled():
        return None, "experience_disabled"
    fp = WorkloadFingerprint(
        model_name=spec.model_name,
        target=target,
        policy=policy,
        device=device,
        dtype=dtype,
        batch_size=int(getattr(spec, "batch_size", 1) or 1),
        map_op=getattr(spec, "map_op", None),
        parallel_workers=getattr(spec, "max_parallel_workers", None),
        reliability=getattr(spec, "reliability", "tolerant") or "tolerant",
        requires_efficiency_runtime=bool(
            getattr(spec, "requires_efficiency_runtime", False)
        ),
        region=getattr(spec, "preferred_region", None),
    )
    return get_learner().predict_runtime(fp)


def record_experience_from_feedback(plan, outcome, ledger=None) -> None:
    """Append an ExperienceRecord after Autopilot.record_feedback."""
    from tron_experience.fingerprint import fingerprint_from_plan
    from tron_experience.ledger import ExperienceLedger, ExperienceRecord

    ledger = ledger or ExperienceLedger()
    plan_d = plan.to_dict() if hasattr(plan, "to_dict") else plan
    oc = outcome.to_dict() if hasattr(outcome, "to_dict") else outcome
    fp = fingerprint_from_plan(plan_d, outcome=oc)
    chosen = plan_d.get("chosen") or {}
    src = (chosen.get("source_signals") or {}).get("prediction_source")
    ledger.append(
        ExperienceRecord(
            plan_id=str(oc.get("plan_id") or plan_d.get("plan_id") or ""),
            fingerprint=fp,
            predicted_runtime_seconds=oc.get("predicted_runtime_seconds"),
            actual_runtime_seconds=float(oc.get("actual_runtime_seconds") or 0),
            predicted_cost_usd=oc.get("predicted_cost_usd"),
            actual_cost_usd=oc.get("actual_cost_usd"),
            predicted_peak_memory_bytes=oc.get("predicted_peak_memory_bytes"),
            actual_peak_memory_bytes=oc.get("actual_peak_memory_bytes"),
            prediction_source=src,
            success=bool(oc.get("success", True)),
            strategy=oc.get("strategy"),
            target=oc.get("target"),
            requeue_or_recovery=dict(oc.get("recovery") or {}),
            optimizations={
                "policy": fp.policy,
                "dtype": fp.dtype,
                "device": fp.device,
            },
        )
    )
