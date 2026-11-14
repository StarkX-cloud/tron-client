"""Measured metrics for Adaptive Compute Intelligence benchmarks.

Every field is either measured or explicitly ``None`` — never a fabricated
zero for GPU util / peak VRAM on CPU-only hosts.
"""

from __future__ import annotations

from typing import Iterable

from tron_autopilot.types import ExecutionOutcome


def sample_gpu_utilization() -> float | None:
    """Return current GPU utilization percent, or None if unavailable.

    Tries ``torch.cuda.utilization(0)`` then ``nvidia-smi``. On CPU-only
    hosts this honestly returns ``None``.
    """
    try:
        import torch

        if torch.cuda.is_available():
            # Available on recent PyTorch builds with NVML.
            if hasattr(torch.cuda, "utilization"):
                return float(torch.cuda.utilization(0))
    except Exception:  # noqa: BLE001
        pass
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        line = out.decode("utf-8", errors="replace").strip().splitlines()[0]
        return float(line.strip())
    except Exception:  # noqa: BLE001
        return None


def aggregate_run_metrics(
    label: str,
    outcomes: list[ExecutionOutcome],
    *,
    wall_seconds: float | None = None,
) -> dict:
    """Prompt 3 benchmark aggregates from measured outcomes only."""
    successes = [o for o in outcomes if o.success]
    failures = [o for o in outcomes if not o.success]
    n = len(outcomes)

    total_wall = wall_seconds
    if total_wall is None:
        total_wall = sum(float(o.actual_runtime_seconds or 0.0) for o in outcomes)

    cost_known = [
        float(o.actual_cost_usd)
        for o in successes
        if o.actual_cost_usd is not None
    ]
    cost_per_success = (
        (sum(cost_known) / len(successes)) if successes and cost_known else None
    )

    successes_per_hour = None
    if total_wall and total_wall > 0 and successes:
        successes_per_hour = len(successes) / (total_wall / 3600.0)

    def _avg(vals: Iterable[float | int | None]) -> float | None:
        xs = [float(v) for v in vals if v is not None]
        return (sum(xs) / len(xs)) if xs else None

    peak_gpu = _avg(
        o.metrics.get("peak_gpu_memory_allocated_bytes")
        for o in successes
        if o.metrics
    )
    peak_cpu = _avg(
        o.actual_peak_memory_bytes for o in successes if o.actual_peak_memory_bytes
    )
    tok_s = _avg(
        o.metrics.get("tokens_per_second") for o in successes if o.metrics
    )
    gpu_util = _avg(
        o.metrics.get("gpu_utilization_percent") for o in successes if o.metrics
    )
    completion = _avg(o.actual_runtime_seconds for o in outcomes)

    # Communication overhead: prefer metrics, else predicted, else None.
    comm_vals = []
    for o in outcomes:
        m = o.metrics or {}
        if m.get("actual_comm_seconds") is not None:
            comm_vals.append(float(m["actual_comm_seconds"]))
        elif m.get("comm_predicted_seconds") is not None:
            comm_vals.append(float(m["comm_predicted_seconds"]))
    avg_comm = _avg(comm_vals)

    recovered = 0
    recovery_attempts = 0
    for o in outcomes:
        rec = o.recovery or {}
        rq = rec.get("requeue_count_observed")
        if rq is None and "shard_requeues" in rec:
            rq = sum(int(x) for x in (rec.get("shard_requeues") or []))
        if rq is not None:
            recovery_attempts += 1
            if float(rq) > 0 and o.success:
                recovered += 1
    recovery_rate = (
        (recovered / recovery_attempts) if recovery_attempts else None
    )

    return {
        "label": label,
        "total_runs": n,
        "successes": len(successes),
        "failures": len(failures),
        "failure_rate": (len(failures) / n) if n else None,
        "successes_per_hour": successes_per_hour,
        "cost_per_successful_usd": cost_per_success,
        "peak_gpu_memory_bytes": peak_gpu,
        "peak_cpu_memory_bytes": peak_cpu,
        "gpu_utilization_percent": gpu_util,
        "throughput_tokens_per_second": tok_s,
        "avg_completion_seconds": completion,
        "total_wall_seconds": total_wall,
        "communication_overhead_seconds": avg_comm,
        "recovery_rate": recovery_rate,
        "notes": _honesty_notes(peak_gpu, gpu_util, avg_comm),
    }


def _honesty_notes(
    peak_gpu: float | None, gpu_util: float | None, comm: float | None
) -> list[str]:
    notes = []
    if peak_gpu is None:
        notes.append("peak GPU memory unavailable (CPU-only or not reported)")
    if gpu_util is None:
        notes.append("GPU utilization unavailable (no NVML/CUDA sampler)")
    if comm is None:
        notes.append("communication overhead not measured for this path")
    notes.append(
        "independent GPUs are NOT claimed as one unified VRAM pool"
    )
    return notes
