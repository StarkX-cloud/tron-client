"""Experience Engine unit + real-log learning tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tron_experience.feedback import DeveloperFeedback, FeedbackStore
from tron_experience.fingerprint import WorkloadFingerprint, fingerprint_from_plan
from tron_experience.governance import GovernancePolicy, ModelRegistry, ModelVersion
from tron_experience.ledger import ExperienceLedger, ExperienceRecord
from tron_experience.learn import ExperienceLearner
from tron_experience.stats import intelligence_snapshot


def test_fingerprint_excludes_prompts():
    plan = {
        "plan_id": "p1",
        "spec": {
            "model_name": "m",
            "batch_size": 2,
            "prompt": "SECRET PROMPT SHOULD NOT ENTER FINGERPRINT",
            "reliability": "tolerant",
        },
        "chosen": {
            "target": "local",
            "policy": "baseline",
            "device": "cpu",
            "dtype": "fp32",
        },
    }
    fp = fingerprint_from_plan(plan)
    d = fp.to_dict()
    assert "prompt" not in d
    assert fp.class_id().startswith("wf_")


def test_ledger_ingest_and_tenant_filter(tmp_path):
    log = tmp_path / "dec.jsonl"
    # Minimal real-shaped plan+outcome pairs
    rows = []
    for i in range(3):
        rows.append(
            {
                "kind": "plan",
                "plan_id": f"p{i}",
                "spec": {"model_name": f"model-{i%2}", "costtrace_task": "customer:acme"},
                "chosen": {
                    "target": "local",
                    "policy": "baseline",
                    "device": "cpu",
                    "dtype": "fp32",
                    "source_signals": {"prediction_source": "user_expected_runtime"},
                },
                "pool_status": {"alive_count": 0},
            }
        )
        rows.append(
            {
                "kind": "outcome",
                "plan_id": f"p{i}",
                "success": True,
                "target": "local",
                "strategy": "autopilot",
                "actual_runtime_seconds": 0.1 * (i + 1),
                "actual_cost_usd": 0.001,
                "predicted_runtime_seconds": 0.1,
                "predicted_cost_usd": 0.001,
                "timestamp": 1000 + i,
            }
        )
    log.write_text("\n".join(__import__("json").dumps(r) for r in rows), encoding="utf-8")
    gov = GovernancePolicy(allow_shared_learning=False)
    ledger = ExperienceLedger(tmp_path / "led.jsonl", governance=gov)
    assert ledger.ingest_decision_log(log) == 3
    assert len(ledger.read_all(customer_id="acme")) == 3
    assert len(ledger.read_all(customer_id="other")) == 0


def test_learner_beats_or_reports_honestly_on_real_artifact(tmp_path):
    src = Path("artifacts/autopilot_decisions.jsonl")
    if not src.exists():
        pytest.skip("autopilot_decisions.jsonl not present")
    ledger = ExperienceLedger(tmp_path / "led.jsonl")
    ledger.ingest_decision_log(src)
    records = ledger.read_all()
    assert len(records) >= 10
    learner = ExperienceLearner(
        registry=ModelRegistry(tmp_path / "models.json"),
        artifact_dir=tmp_path / "art",
    )
    report = learner.train(records, holdout_fraction=0.3, promote_if_better=True)
    assert report.train_n >= 5
    assert report.holdout_n >= 1
    # Either beats baseline (preferred) or remains candidate without silent promote
    if report.beats_baseline:
        assert learner.registry.active() is not None
        assert report.mae < report.baseline_mae
    else:
        assert learner.registry.active() is None


def test_promote_requires_beats_baseline(tmp_path):
    reg = ModelRegistry(tmp_path / "m.json")
    reg.register(
        ModelVersion(
            version_id="v1",
            created_at=1.0,
            train_n=10,
            holdout_n=3,
            metric="mae_runtime_seconds",
            mae=1.0,
            baseline_mae=0.5,
            features=["model_name"],
            artifact_path=None,
            status="candidate",
            beats_baseline=False,
        )
    )
    with pytest.raises(PermissionError):
        reg.promote("v1")


def test_feedback_store(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.submit(
        DeveloperFeedback(
            plan_id="p1", met_requirements=True, rating=5, customer_id="c1"
        )
    )
    s = store.satisfaction_rate(customer_id="c1")
    assert s["n"] == 1
    assert s["met_requirements_rate"] == 1.0


def test_bridge_respects_disabled(monkeypatch):
    monkeypatch.delenv("TRON_EXPERIENCE_ENABLED", raising=False)
    from tron_autopilot.types import WorkloadSpec
    from tron_experience.bridge import predict_runtime_for_spec

    spec = WorkloadSpec(model_name="m", hourly_rate_usd=0.5)
    rt, src = predict_runtime_for_spec(spec, "local", "baseline", "cpu", "fp32")
    assert rt is None
    assert src == "experience_disabled"


def test_intelligence_snapshot(tmp_path):
    ledger = ExperienceLedger(tmp_path / "l.jsonl")
    fp = WorkloadFingerprint(
        model_name="m",
        target="local",
        policy="baseline",
        device="cpu",
        dtype="fp32",
        worker_sku="cpu-lab",
    )
    ledger.append(
        ExperienceRecord(
            plan_id="p",
            fingerprint=fp,
            predicted_runtime_seconds=0.2,
            actual_runtime_seconds=0.25,
            predicted_cost_usd=0.01,
            actual_cost_usd=0.012,
            success=True,
        )
    )
    snap = intelligence_snapshot(
        ledger=ledger, registry=ModelRegistry(tmp_path / "m.json")
    )
    assert snap["workloads_observed"] == 1
    assert "cpu-lab" in snap["hardware_performance_profiles"]


def test_flywheel_six_stages(tmp_path):
    from tron_experience.flywheel import score_flywheel

    ledger = ExperienceLedger(tmp_path / "l.jsonl")
    for i in range(5):
        ledger.append(
            ExperienceRecord(
                plan_id=f"p{i}",
                fingerprint=WorkloadFingerprint(
                    model_name=f"m{i%2}",
                    target="local" if i % 2 == 0 else "pooled",
                    policy="baseline",
                    device="cpu",
                    dtype="fp32",
                ),
                predicted_runtime_seconds=0.1 * (i + 1),
                actual_runtime_seconds=0.1 * (i + 1) * 1.05,
                predicted_cost_usd=0.01,
                actual_cost_usd=0.011,
                success=True,
                timestamp=1000 + i,
            )
        )
    fw = score_flywheel(
        ledger=ledger,
        registry=ModelRegistry(tmp_path / "m.json"),
        fleet={"alive_count": 2, "workers": [{}, {}]},
    )
    assert len(fw["hierarchy"]) == 6
    assert fw["hierarchy"][0]["stage_id"] == "acquire"
    assert fw["hierarchy"][-1]["stage_id"] == "compound"
    assert "execution intelligence" in fw["thesis"].lower() or "data teaches" in fw["thesis"].lower()
    assert fw["flywheel_health"] is not None
