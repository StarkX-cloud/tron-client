#!/usr/bin/env python3
"""Adaptive Compute Intelligence benchmark: optimized vs unoptimized.

Identical workload + hardware; only the optimization profile differs.

    python3 -m benchmarks.adaptive_compute
    python3 -m benchmarks.adaptive_compute --real-model --trials 1
    python3 -m benchmarks.adaptive_compute --out artifacts/adaptive_compute_benchmark.json

Metrics (Prompt 3): successes/hour, cost/success, peak GPU mem, GPU util,
throughput, completion time, communication overhead, failure/recovery rates.

Honesty: on CPU-only hosts GPU util / peak VRAM are None — never fabricated.
Independent GPUs are never claimed as one VRAM pool.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from tron_efficiency.hardware.detect import detect_hardware

from tron_adaptive.loop import AdaptiveIntelligence
from tron_adaptive.metrics import aggregate_run_metrics
from tron_adaptive.performance_store import PerformanceStore
from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.store import DecisionLog, RuntimeHistory
from tron_autopilot.types import WorkloadSpec


def _synthetic(duration: float = 0.05, allocate_mb: int = 2) -> dict:
    buf = bytearray(allocate_mb * 1024 * 1024)
    for i in range(0, len(buf), 4096):
        buf[i] = i % 256
    time.sleep(duration)
    return {"ok": True, "checksum": sum(buf[::4096]) % 10_000}


def run_benchmark(
    *,
    trials: int = 2,
    hourly_rate: float = 0.5,
    real_model: bool = False,
    model: str = "EleutherAI/pythia-70m",
    max_new_tokens: int = 16,
    out: Path | None = None,
) -> dict:
    hardware = detect_hardware()
    tmp = Path(tempfile.mkdtemp(prefix="tron-adaptive-"))
    store = PerformanceStore(tmp / "perf.json")
    ap = EconomicAutopilot(
        decision_log=DecisionLog(tmp / "decisions.jsonl"),
        history=RuntimeHistory(tmp / "history.json"),
        hardware=hardware,
    )
    brain = AdaptiveIntelligence(autopilot=ap, store=store, hardware=hardware)

    base_spec = WorkloadSpec(
        model_name=model if real_model else "synthetic/adaptive",
        hourly_rate_usd=hourly_rate,
        expected_runtime_seconds=0.2 if not real_model else None,
        prompt="The quick brown fox jumps over the lazy dog, and",
        max_new_tokens=max_new_tokens,
        reliability="tolerant",
    )

    modes = ("unoptimized", "optimized")
    results: dict[str, dict] = {}
    tradeoffs: dict[str, list] = {}
    warmup_note = None

    # Shared warmup so cold HF download / first-load does not favor the
    # second mode. Discarded from aggregates.
    if real_model:
        print("[adaptive-bench] warmup (discarded)…")
        _p, _plan, _o, _s = brain.run(
            base_spec,
            mode="unoptimized",
            strategy="autopilot",
            efficiency=True,
            wait_timeout_seconds=600.0,
        )
        warmup_note = {
            "success": _o.success,
            "runtime_seconds": _o.actual_runtime_seconds,
            "error": _o.error,
        }
        print(f"[adaptive-bench] warmup done success={_o.success}")

    for mode in modes:
        outcomes = []
        profiles = []
        t0 = time.perf_counter()
        for i in range(trials):
            if real_model:
                profile, plan, outcome, sample = brain.run(
                    base_spec,
                    mode=mode,
                    strategy="autopilot",
                    efficiency=True,
                    wait_timeout_seconds=600.0,
                )
            else:
                profile, plan, outcome, sample = brain.run(
                    base_spec,
                    mode=mode,
                    strategy="autopilot",
                    efficiency=False,
                    func=_synthetic,
                    func_kwargs={"duration": 0.08, "allocate_mb": 2},
                    wait_timeout_seconds=60.0,
                )
            outcomes.append(outcome)
            profiles.append(profile.to_dict())
            _ = sample
        wall = time.perf_counter() - t0
        agg = aggregate_run_metrics(mode, outcomes, wall_seconds=wall)
        agg["profiles"] = profiles
        results[mode] = agg
        tradeoffs[mode] = profiles[-1].get("tradeoffs") if profiles else []

    comparison = {
        "benchmark": "adaptive_compute_optimized_vs_unoptimized",
        "model": base_spec.model_name,
        "real_model": real_model,
        "trials_per_mode": trials,
        "hardware": {
            "cuda_available": hardware.cuda_available,
            "cpu_count": hardware.cpu_count,
            "cpu_total_memory_bytes": hardware.cpu_total_memory_bytes,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "total_memory_bytes": g.total_memory_bytes,
                }
                for g in hardware.gpus
            ],
        },
        "modes": results,
        "tradeoffs": tradeoffs,
        "delta": _delta(results.get("unoptimized"), results.get("optimized")),
        "warmup": warmup_note,
        "honesty": [
            "Same workload and host for both modes.",
            "Unoptimized = fixed baseline/fp32/batch=1.",
            "Optimized = advisor + measured performance history + batch/device levers.",
            "Real-model runs share a discarded warmup to avoid cold-cache bias.",
            "Does not claim unified VRAM across independent GPUs.",
            "GPU util / peak VRAM are None when CUDA/NVML unavailable.",
        ],
        "performance_store_samples": len(store.samples_for()),
    }

    text = _format_md(comparison)
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
        out.with_suffix(".md").write_text(text, encoding="utf-8")
    print(text)
    return comparison


def _delta(unopt: dict | None, opt: dict | None) -> dict:
    if not unopt or not opt:
        return {}

    def ratio(a, b):
        if a is None or b is None or a == 0:
            return None
        return b / a

    return {
        "completion_time_ratio_opt_over_unopt": ratio(
            unopt.get("avg_completion_seconds"), opt.get("avg_completion_seconds")
        ),
        "cost_per_success_ratio_opt_over_unopt": ratio(
            unopt.get("cost_per_successful_usd"), opt.get("cost_per_successful_usd")
        ),
        "successes_per_hour_ratio_opt_over_unopt": ratio(
            unopt.get("successes_per_hour"), opt.get("successes_per_hour")
        ),
        "throughput_ratio_opt_over_unopt": ratio(
            unopt.get("throughput_tokens_per_second"),
            opt.get("throughput_tokens_per_second"),
        ),
    }


def _format_md(c: dict) -> str:
    lines = [
        "# Adaptive Compute Intelligence benchmark",
        "",
        f"- model: `{c['model']}`",
        f"- real_model: {c['real_model']}",
        f"- trials/mode: {c['trials_per_mode']}",
        f"- cuda_available: {c['hardware']['cuda_available']}",
        "",
        "| Mode | Successes/hr | Cost/success ($) | Avg completion (s) | "
        "Peak GPU mem | GPU util % | Throughput tok/s | Fail rate | Recovery rate | Comm (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("unoptimized", "optimized"):
        r = c["modes"][mode]
        lines.append(
            f"| {mode} | {_f(r.get('successes_per_hour'))} | "
            f"{_f(r.get('cost_per_successful_usd'))} | "
            f"{_f(r.get('avg_completion_seconds'))} | "
            f"{_f(r.get('peak_gpu_memory_bytes'), as_int=True)} | "
            f"{_f(r.get('gpu_utilization_percent'))} | "
            f"{_f(r.get('throughput_tokens_per_second'))} | "
            f"{_f(r.get('failure_rate'))} | "
            f"{_f(r.get('recovery_rate'))} | "
            f"{_f(r.get('communication_overhead_seconds'))} |"
        )
    lines.append("")
    lines.append("## Deltas (optimized / unoptimized)")
    for k, v in (c.get("delta") or {}).items():
        lines.append(f"- {k}: {_f(v)}")
    lines.append("")
    lines.append("## Trade-offs")
    for mode, ts in (c.get("tradeoffs") or {}).items():
        lines.append(f"### {mode}")
        for t in ts or ["—"]:
            lines.append(f"- {t}")
    lines.append("")
    lines.append("## Honesty")
    for h in c.get("honesty") or []:
        lines.append(f"- {h}")
    return "\n".join(lines) + "\n"


def _f(v, as_int: bool = False) -> str:
    if v is None:
        return "—"
    if as_int:
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--hourly-rate", type=float, default=0.5)
    p.add_argument("--real-model", action="store_true")
    p.add_argument("--model", default="EleutherAI/pythia-70m")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument(
        "--out",
        default="artifacts/adaptive_compute_benchmark.json",
    )
    args = p.parse_args()
    run_benchmark(
        trials=args.trials,
        hourly_rate=args.hourly_rate,
        real_model=args.real_model,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        out=Path(args.out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
