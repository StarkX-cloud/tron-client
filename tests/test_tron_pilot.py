"""Unit tests for tron_pilot (provider, usage, security, client body)."""

from __future__ import annotations

from tron_pilot.provider import ProviderRegistry
from tron_pilot.provider_stats import ProviderOutcome, ProviderStatsStore
from tron_pilot.security import current_trust_model, redact_secrets
from tron_pilot.usage import UsageEvent, UsageLedger, usage_report


def test_provider_register_verify_challenge(tmp_path):
    reg = ProviderRegistry(tmp_path / "p.json")
    rec, token = reg.register(
        worker_name="w1",
        master_url="http://127.0.0.1:9",
        region="lab",
        register_with_master=False,
    )
    assert token is None
    assert not rec.verified
    ch = reg.issue_challenge(rec.provider_id)
    verified = reg.verify(rec.provider_id, verified_by="test", challenge=ch)
    assert verified.verified
    assert verified.verified_by == "test"
    rows = reg.list_providers(verified_only=True)
    assert len(rows) == 1
    assert "auth_token" not in rows[0] or rows[0].get("auth_token") == "***"


def test_provider_challenge_mismatch(tmp_path):
    reg = ProviderRegistry(tmp_path / "p.json")
    rec, _ = reg.register(
        worker_name="w1", master_url="http://x", register_with_master=False
    )
    reg.issue_challenge(rec.provider_id)
    try:
        reg.verify(rec.provider_id, challenge="wrong")
        assert False, "expected PermissionError"
    except PermissionError:
        pass


def test_usage_ledger_and_report(tmp_path):
    ledger = UsageLedger(tmp_path / "u.jsonl")
    ledger.record(
        UsageEvent(
            customer_id="c1",
            event_type="workload_complete",
            plan_id="p1",
            model_name="m",
            target="local",
            success=True,
            runtime_seconds=1.5,
            cost_usd=0.01,
        )
    )
    ledger.record(
        UsageEvent(
            customer_id="c1",
            event_type="workload_failed",
            plan_id="p2",
            model_name="m",
            target="pooled",
            success=False,
            runtime_seconds=0.2,
            cost_usd=None,
        )
    )
    report = usage_report(ledger, customer_id="c1")
    assert report["events"] == 2
    assert report["successes"] == 1
    assert report["failures"] == 1
    assert abs(report["total_cost_usd_estimate"] - 0.01) < 1e-9
    assert "not a TRON-issued invoice" in report["note"]


def test_provider_stats_summary(tmp_path):
    store = ProviderStatsStore(tmp_path / "s.jsonl")
    store.record(
        ProviderOutcome(
            worker_name="w1",
            provider_id="prov_1",
            success=True,
            runtime_seconds=1.0,
            cost_usd=0.001,
            requeue_count=0,
        )
    )
    store.record(
        ProviderOutcome(
            worker_name="w1",
            provider_id="prov_1",
            success=False,
            runtime_seconds=0.5,
            requeue_count=2,
            error="boom",
        )
    )
    s = store.summary(worker_name="w1")
    assert s["n"] == 2
    assert s["successes"] == 1
    assert s["success_rate"] == 0.5


def test_trust_model_and_redact(monkeypatch):
    monkeypatch.delenv("TRON_SUBMIT_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TRON_AUTOPILOT_API_TOKEN", raising=False)
    tm = current_trust_model()
    assert tm.submit_auth_required is False
    assert "rlimit" in tm.sandbox
    assert redact_secrets({"auth_token": "secret", "name": "w"})["auth_token"] == "***"
