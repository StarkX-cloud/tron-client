#!/usr/bin/env python3
"""Prove Experience Engine learning beats naive baseline on real logs.

Uses artifacts/autopilot_decisions.jsonl (real Autopilot bench outcomes).
No fabricated training rows. Writes artifacts/experience_engine_proof.json.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from tron_experience.governance import ModelRegistry
from tron_experience.ledger import ExperienceLedger
from tron_experience.learn import ExperienceLearner
from tron_experience.stats import intelligence_snapshot


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    src = root / "artifacts" / "autopilot_decisions.jsonl"
    if not src.exists():
        raise SystemExit(f"missing real decision log: {src}")

    tmp = Path(tempfile.mkdtemp(prefix="tron-exp-"))
    ledger = ExperienceLedger(tmp / "ledger.jsonl")
    n = ledger.ingest_decision_log(src)
    records = ledger.read_all()
    print(f"ingested {n} outcomes → {len(records)} ledger records from {src}")

    learner = ExperienceLearner(
        registry=ModelRegistry(tmp / "models.json"),
        artifact_dir=tmp / "artifacts",
    )
    report = learner.train(records, holdout_fraction=0.3, promote_if_better=True)

    # Unseen-workload check: holdout classes not fully identical to train —
    # chronological split already used; also evaluate on real-model log if present.
    real_src = root / "artifacts" / "autopilot_real_decisions.jsonl"
    unseen = {}
    if real_src.exists():
        ledger2 = ExperienceLedger(tmp / "ledger_real.jsonl")
        ledger2.ingest_decision_log(real_src)
        real_recs = [r for r in ledger2.read_all() if r.actual_runtime_seconds]
        if real_recs and learner.registry.active():
            preds = []
            actuals = []
            baseline = sum(r.actual_runtime_seconds for r in records) / len(records)
            for r in real_recs:
                p, src = learner.predict_runtime(r.fingerprint)
                if p is None:
                    continue
                preds.append(p)
                actuals.append(r.actual_runtime_seconds)
            if preds:
                mae_m = sum(abs(p - a) for p, a in zip(preds, actuals)) / len(preds)
                mae_b = sum(abs(baseline - a) for a in actuals) / len(actuals)
                unseen = {
                    "source": str(real_src),
                    "n": len(preds),
                    "experience_mae": mae_m,
                    "naive_global_mean_mae": mae_b,
                    "beats_naive_on_unseen": mae_m < mae_b,
                }

    snap = intelligence_snapshot(ledger=ledger, registry=learner.registry)
    artifact = {
        "success": bool(report.beats_baseline),
        "train_report": report.to_dict(),
        "unseen_real_model_eval": unseen,
        "intelligence": {
            k: snap[k]
            for k in (
                "workloads_observed",
                "workload_categories_learned",
                "prediction_accuracy",
                "model_improvements",
                "honesty",
            )
        },
        "constraints": [
            "Predictions are soft estimates only",
            "Promotion requires beats_baseline",
            "No prompts/code/model I/O in training features",
            "Source rows: real Autopilot DecisionLog artifacts",
        ],
    }
    out = root / "artifacts" / "experience_engine_proof.json"
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    md = root / "artifacts" / "experience_engine_proof.md"
    md.write_text(
        "# Experience Engine proof\n\n"
        f"- beats baseline (holdout): `{artifact['success']}`\n"
        f"- MAE model={report.mae:.6f} baseline={report.baseline_mae:.6f}\n"
        f"- train_n={report.train_n} holdout_n={report.holdout_n}\n"
        f"- unseen: `{json.dumps(unseen)}`\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2)[:2000])
    return 0 if artifact["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
