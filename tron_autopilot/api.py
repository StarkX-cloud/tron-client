"""HTTP control plane for Economic Autopilot (AI-company integration surface).

Run: ``uvicorn tron_autopilot.api:app --host 0.0.0.0 --port 9100``

Optional gate: ``TRON_AUTOPILOT_API_TOKEN`` → require ``X-TRON-AUTH`` header.
"""

from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.execute import execute_plan
from tron_autopilot.fleet import query_fleet_capacity
from tron_autopilot.pricing import GPU_SKU_HOURLY_USD
from tron_autopilot.types import WorkloadSpec

app = FastAPI(title="TRON Economic Autopilot", version="0.2.0")
API_TOKEN = os.environ.get("TRON_AUTOPILOT_API_TOKEN") or None


def _require_api_auth(x_tron_auth: str | None) -> None:
    if not API_TOKEN:
        return
    if not x_tron_auth or not hmac.compare_digest(x_tron_auth, API_TOKEN):
        raise HTTPException(status_code=401, detail="missing or invalid X-TRON-AUTH")


class PlanRequest(BaseModel):
    model_name: str
    hourly_rate_usd: float = 0.5
    budget_usd: float | None = None
    deadline_seconds: float | None = None
    reliability: str = "tolerant"
    pooled_hourly_rate_usd: float | None = None
    expected_runtime_seconds: float | None = None
    requires_multi_node: bool = False
    master_url: str | None = None
    costtrace_task: str | None = None
    prompt: str = "The quick brown fox jumps over the lazy dog, and"
    max_new_tokens: int = 32
    requires_efficiency_runtime: bool = False
    preferred_region: str | None = None
    preferred_location_strict: bool = False
    strategy: str = "autopilot"
    # Distributed Execution Engine v1
    parallelizable: bool = False
    partition_items: list | None = None
    map_op: str = "sum_squares"
    min_shard_size: int = 1
    max_parallel_workers: int | None = None
    bytes_per_item: float = 64.0


class RunRequest(PlanRequest):
    efficiency: bool = True
    wait_timeout_seconds: float = 600.0


def _spec_from(req: PlanRequest, *, efficiency: bool | None = None) -> WorkloadSpec:
    # When efficiency=True (HTTP /run default), require efficiency_runtime
    # workers unless the caller explicitly cleared the flag.
    needs_eff = req.requires_efficiency_runtime
    if efficiency is True and not needs_eff:
        needs_eff = True
    return WorkloadSpec(
        model_name=req.model_name,
        hourly_rate_usd=req.hourly_rate_usd,
        budget_usd=req.budget_usd,
        deadline_seconds=req.deadline_seconds,
        reliability=req.reliability,
        pooled_hourly_rate_usd=req.pooled_hourly_rate_usd,
        expected_runtime_seconds=req.expected_runtime_seconds,
        requires_multi_node=req.requires_multi_node,
        master_url=req.master_url,
        costtrace_task=req.costtrace_task,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        requires_efficiency_runtime=needs_eff,
        preferred_region=req.preferred_region,
        preferred_location_strict=req.preferred_location_strict,
        parallelizable=req.parallelizable,
        partition_items=req.partition_items,
        map_op=req.map_op,
        min_shard_size=req.min_shard_size,
        max_parallel_workers=req.max_parallel_workers,
        bytes_per_item=req.bytes_per_item,
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "tron_autopilot"}


@app.get("/pricing/skus")
def list_skus(x_tron_auth: str | None = Header(default=None, alias="X-TRON-AUTH")):
    _require_api_auth(x_tron_auth)
    return {"skus": GPU_SKU_HOURLY_USD, "note": "pinned snapshot — not a live feed"}


@app.get("/fleet")
def fleet(
    master_url: str | None = None,
    x_tron_auth: str | None = Header(default=None, alias="X-TRON-AUTH"),
):
    _require_api_auth(x_tron_auth)
    return query_fleet_capacity(master_url=master_url).to_dict()


@app.post("/plan")
def plan(
    req: PlanRequest,
    x_tron_auth: str | None = Header(default=None, alias="X-TRON-AUTH"),
):
    _require_api_auth(x_tron_auth)
    ap = EconomicAutopilot()
    plan_obj = ap.plan(_spec_from(req), strategy=req.strategy)
    return plan_obj.to_dict()


@app.post("/run")
def run(
    req: RunRequest,
    x_tron_auth: str | None = Header(default=None, alias="X-TRON-AUTH"),
):
    _require_api_auth(x_tron_auth)
    ap = EconomicAutopilot()
    spec = _spec_from(req, efficiency=req.efficiency)
    plan_obj = ap.plan(spec, strategy=req.strategy)
    outcome = execute_plan(
        plan_obj,
        efficiency=req.efficiency,
        wait_timeout_seconds=req.wait_timeout_seconds,
        strategy=req.strategy,
    )
    ap.record_feedback(plan_obj, outcome)
    return {"plan": plan_obj.to_dict(), "outcome": outcome.to_dict()}


@app.get("/trust")
def trust(x_tron_auth: str | None = Header(default=None, alias="X-TRON-AUTH")):
    """Declare auth/sandbox trust boundaries (Prompt 4)."""
    _require_api_auth(x_tron_auth)
    from tron_pilot.security import current_trust_model

    return current_trust_model().__dict__


@app.get("/providers")
def providers(
    verified_only: bool = False,
    x_tron_auth: str | None = Header(default=None, alias="X-TRON-AUTH"),
):
    _require_api_auth(x_tron_auth)
    from tron_pilot.provider import ProviderRegistry

    return {"providers": ProviderRegistry().list_providers(verified_only=verified_only)}


@app.get("/providers/{provider_id}/stats")
def provider_stats(
    provider_id: str,
    x_tron_auth: str | None = Header(default=None, alias="X-TRON-AUTH"),
):
    _require_api_auth(x_tron_auth)
    from tron_pilot.provider import ProviderRegistry
    from tron_pilot.provider_stats import ProviderStatsStore

    reg = ProviderRegistry()
    rec = reg.get(provider_id)
    stats = ProviderStatsStore().summary(
        worker_name=rec.worker_name if rec else None,
        provider_id=provider_id,
    )
    return {"provider": rec.to_dict() if rec else None, "stats": stats}


@app.get("/usage")
def usage(
    customer_id: str | None = None,
    since: float | None = None,
    until: float | None = None,
    x_tron_auth: str | None = Header(default=None, alias="X-TRON-AUTH"),
):
    """Customer usage report hook — estimates only, no custody."""
    _require_api_auth(x_tron_auth)
    from tron_pilot.usage import UsageLedger, usage_report

    return usage_report(
        UsageLedger(), customer_id=customer_id, since=since, until=until
    )


@app.get("/usage/sla")
def usage_sla(
    customer_id: str | None = None,
    deadline_seconds: float | None = None,
    since: float | None = None,
    until: float | None = None,
    x_tron_auth: str | None = Header(default=None, alias="X-TRON-AUTH"),
):
    """Service-level measurement (p50/p95, availability) — not credits."""
    _require_api_auth(x_tron_auth)
    from tron_pilot.sla import sla_report
    from tron_pilot.usage import UsageLedger

    return sla_report(
        UsageLedger(),
        customer_id=customer_id,
        deadline_seconds=deadline_seconds,
        since=since,
        until=until,
    )


# Static Autopilot console (same pattern as tron_efficiency /dashboard).
_dashboard_dir = os.path.join(os.path.dirname(__file__), "dashboard")
if os.path.isdir(_dashboard_dir):
    from fastapi.staticfiles import StaticFiles

    app.mount(
        "/dashboard",
        StaticFiles(directory=_dashboard_dir, html=True),
        name="autopilot_dashboard",
    )
