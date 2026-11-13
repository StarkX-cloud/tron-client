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
    strategy: str = "autopilot"


class RunRequest(PlanRequest):
    efficiency: bool = True
    wait_timeout_seconds: float = 600.0


def _spec_from(req: PlanRequest) -> WorkloadSpec:
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
        requires_efficiency_runtime=req.requires_efficiency_runtime,
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
    spec = _spec_from(req)
    plan_obj = ap.plan(spec, strategy=req.strategy)
    outcome = execute_plan(
        plan_obj,
        efficiency=req.efficiency,
        wait_timeout_seconds=req.wait_timeout_seconds,
        strategy=req.strategy,
    )
    ap.record_feedback(outcome)
    return {"plan": plan_obj.to_dict(), "outcome": outcome.to_dict()}
