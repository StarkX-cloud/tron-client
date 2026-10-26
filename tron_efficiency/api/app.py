"""Standalone FastAPI app for the Compute Efficiency Engine.

Deliberately not a router bolted onto ``queue_server.py`` — this is a
separate local tool with zero coupling to the job-queue/orchestration
server; it can run on its own port and even in a process that never
imports ``tron``/``queue_server`` at all.

``run_workload`` is imported as a module-level name (rather than called via
a fully-qualified path inline) so tests can monkeypatch
``tron_efficiency.api.app.run_workload`` to avoid loading a real model.
"""

from __future__ import annotations

import io
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tron_efficiency.advisor.recommend import recommend as _recommend
from tron_efficiency.benchmarks.engine import run_budget_sweep as _run_budget_sweep
from tron_efficiency.hardware.detect import detect_hardware
from tron_efficiency.memory.policies import BaselinePolicy, OffloadPolicy
from tron_efficiency.results.report import generate_report, generate_sweep_report
from tron_efficiency.results.schema import BenchmarkResult
from tron_efficiency.results.writer import list_runs, load_run, save_run, write_csv_rows
from tron_efficiency.workloads.config import WorkloadConfig
from tron_efficiency.workloads.runner import run_workload

app = FastAPI(title="TRON Compute Efficiency Engine", version="0.1.0")


class RunRequest(BaseModel):
    model: str = "EleutherAI/pythia-70m"
    prompt: str = WorkloadConfig().prompt
    max_new_tokens: int = 32
    batch_size: int = 1
    device: str = "cpu"
    policy: str = "baseline"
    memory_budget_mb: int = 100
    dtype: str | None = None


def _build_policy(req: RunRequest):
    if req.policy == "baseline":
        return BaselinePolicy()
    if req.policy == "offload":
        return OffloadPolicy(memory_budget_bytes=req.memory_budget_mb * 1024 * 1024)
    raise HTTPException(status_code=400, detail=f"unknown policy: {req.policy}")


@app.get("/hardware")
def get_hardware():
    return detect_hardware().to_dict()


@app.get("/recommend")
def get_recommendation(model: str, gpu_budget_mb: int | None = None):
    gpu_budget_bytes = gpu_budget_mb * 1024 * 1024 if gpu_budget_mb else None
    return _recommend(model, gpu_budget_bytes_override=gpu_budget_bytes).to_dict()


@app.post("/runs")
def create_run(req: RunRequest):
    hardware = detect_hardware()
    config = WorkloadConfig(
        model_name=req.model,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        batch_size=req.batch_size,
        device=req.device,
        dtype=req.dtype,
    )
    policy = _build_policy(req)
    outcome = run_workload(config, policy)
    result = BenchmarkResult.build(hardware, config, policy.describe(), outcome)
    save_run(result)
    return result.to_dict()


class SweepRequest(BaseModel):
    model: str = "EleutherAI/pythia-70m"
    prompt: str = WorkloadConfig().prompt
    max_new_tokens: int = 32
    batch_size: int = 1
    device: str = "cpu"
    dtype: str | None = None
    memory_budget_mb_list: list[int]
    include_baseline: bool = True


@app.post("/sweep")
def create_sweep(req: SweepRequest):
    config = WorkloadConfig(
        model_name=req.model,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        batch_size=req.batch_size,
        device=req.device,
        dtype=req.dtype,
    )
    budgets_bytes = [mb * 1024 * 1024 for mb in req.memory_budget_mb_list]
    results = _run_budget_sweep(config, budgets_bytes, include_baseline=req.include_baseline)
    for r in results:
        save_run(r)
    return {
        "results": [r.to_dict() for r in results],
        "report": generate_sweep_report(results),
    }


@app.get("/runs")
def get_runs():
    return [r.to_dict() for r in list_runs()]


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    try:
        return load_run(run_id).to_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="run not found")


@app.get("/runs/{run_id}/export.csv")
def export_run_csv(run_id: str):
    try:
        result = load_run(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="run not found")
    buf = io.StringIO()
    write_csv_rows([result], buf)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={run_id}.csv"},
    )


@app.get("/report", response_class=PlainTextResponse)
def get_report(hourly_rate: float | None = None):
    return generate_report(list_runs(), hourly_rate_usd=hourly_rate)


_dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")
if os.path.isdir(_dashboard_dir):
    app.mount(
        "/dashboard", StaticFiles(directory=_dashboard_dir, html=True), name="dashboard"
    )
