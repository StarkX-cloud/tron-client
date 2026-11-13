"""SKU pricing + costtrace ledger wiring."""

from __future__ import annotations

import json
from pathlib import Path

from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.execute import execute_plan
from tron_autopilot.pricing import hourly_usd_for_sku, resolve_hourly_rate
from tron_autopilot.types import WorkloadSpec


def test_sku_table_known():
    assert hourly_usd_for_sku("g2-gpu-rtx4000a1-s") == 0.52
    assert hourly_usd_for_sku("no-such-sku") is None


def test_resolve_priority():
    assert resolve_hourly_rate(explicit=1.0, sku="g2-gpu-rtx4000a1-s") == 1.0
    assert resolve_hourly_rate(sku="g2-gpu-rtx4000a1-s") == 0.52
    assert resolve_hourly_rate(advertised=0.2, sku="g2-gpu-rtx4000a1-s") == 0.2


def test_costtrace_records_compute(tmp_path, monkeypatch):
    log = tmp_path / "cost.jsonl"
    monkeypatch.setenv("TRON_COSTTRACE_PATH", str(log))
    ap = EconomicAutopilot()
    spec = WorkloadSpec(
        model_name="EleutherAI/pythia-70m",
        hourly_rate_usd=1.0,
        costtrace_task="unit-test-compute",
        max_new_tokens=1,
        expected_runtime_seconds=0.01,
    )
    plan = ap.plan(spec, strategy="fixed")

    def _quick():
        return {"ok": True}

    outcome = execute_plan(plan, func=_quick, strategy="fixed")
    assert outcome.success
    assert log.exists()
    lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    assert lines
    assert lines[-1]["task"] == "unit-test-compute"
    assert lines[-1]["success"] is True
    assert lines[-1]["cost_usd"] is not None
