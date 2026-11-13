"""Run a real efficiency workload on a pooled worker (cloudpickle job).

The worker process must have advertised ``efficiency_runtime=True`` and
actually be able to import torch/transformers/accelerate. Autopilot
checks fleet capacity before submitting; this module is the payload.
"""

from __future__ import annotations


def run_efficiency_job(
    model_name: str,
    policy: str,
    device: str,
    dtype: str | None,
    memory_budget_bytes: int | None,
    max_new_tokens: int = 16,
    prompt: str = "The quick brown fox jumps over the lazy dog, and",
    hourly_rate_usd: float = 0.0,
) -> dict:
    """Executed on the worker. Returns a JSON-serializable outcome dict."""
    from tron_efficiency.memory.policies import BaselinePolicy, OffloadPolicy
    from tron_efficiency.workloads.config import WorkloadConfig
    from tron_efficiency.workloads.runner import run_workload

    if policy == "offload":
        if not memory_budget_bytes:
            return {
                "success": False,
                "error": "offload policy requires memory_budget_bytes on the worker",
            }
        mem_policy = OffloadPolicy(memory_budget_bytes=int(memory_budget_bytes))
    else:
        mem_policy = BaselinePolicy()

    config = WorkloadConfig(
        model_name=model_name,
        prompt=prompt,
        max_new_tokens=int(max_new_tokens),
        batch_size=1,
        device=device or "cpu",
        dtype=dtype,
    )
    import time

    start = time.perf_counter()
    outcome = run_workload(config, mem_policy)
    wall = time.perf_counter() - start
    measured = None
    if outcome.load_time_seconds is not None and outcome.end_to_end_latency_seconds is not None:
        measured = outcome.load_time_seconds + outcome.end_to_end_latency_seconds
    runtime = measured if measured is not None else wall
    peak = outcome.peak_gpu_memory_allocated_bytes or outcome.peak_cpu_memory_bytes
    cost = (hourly_rate_usd / 3600.0) * runtime if hourly_rate_usd else None
    return {
        "success": outcome.success,
        "error": outcome.error,
        "actual_runtime_seconds": runtime,
        "actual_cost_usd": cost,
        "actual_peak_memory_bytes": peak,
        "metrics": {
            "workload": "pooled_efficiency.run_efficiency_job",
            "policy_name": outcome.policy_name,
            "device": config.device,
            "dtype": config.dtype,
            "load_time_seconds": outcome.load_time_seconds,
            "end_to_end_latency_seconds": outcome.end_to_end_latency_seconds,
            "tokens_per_second": outcome.tokens_per_second,
            "tokens_generated": outcome.tokens_generated,
            "peak_cpu_memory_bytes": outcome.peak_cpu_memory_bytes,
            "peak_gpu_memory_allocated_bytes": outcome.peak_gpu_memory_allocated_bytes,
            "wall_clock_seconds": wall,
        },
        "generated_text": outcome.generated_text,
    }
