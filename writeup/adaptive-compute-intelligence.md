# TRON Adaptive Compute Intelligence (Prompt 3)

**Status:** implemented as `tron_adaptive/` on top of Autopilot + efficiency + distributed.  
**Honesty bar:** predicted → actual → updated decisions from **measured** samples. No unified-VRAM claim across independent GPUs.

## What it reuses

| Capability | Source |
|---|---|
| Memory policy (baseline / offload) | `tron_efficiency.memory.policies` |
| Quantization (fp16/int8/int4 when supported) | advisor + BitsAndBytes paths |
| Heterogeneous CPU/GPU | `detect_hardware` + advisor `device` |
| Independent-task partitioning | `tron_autopilot.distributed` |
| Network-aware refusal of bad parallelization | `distributed.cost_model` |
| Runtime feedback | `EconomicAutopilot.record_feedback` + `RuntimeHistory` |

## What it adds

| Module | Role |
|---|---|
| `tron_adaptive/performance_store.py` | Multi-metric history (tok/s, peak mem, success, recovery) |
| `tron_adaptive/optimizer.py` | Unoptimized vs optimized lever profiles |
| `tron_adaptive/batching.py` | Dynamic batch size from headroom / history |
| `tron_adaptive/loop.py` | Apply profile → plan → execute → close loop |
| `tron_adaptive/metrics.py` | Prompt 3 aggregates + optional GPU util sampler |
| `benchmarks/adaptive_compute.py` | Optimized vs unoptimized A/B harness |

## Feedback loop

```
OptimizationProfile
    → WorkloadSpec overrides (force_policy/dtype/device, batch_size)
    → EconomicAutopilot.plan (local|pooled|distributed)
    → execute_plan (efficiency / distributed / callable)
    → record_feedback (RuntimeHistory)
    → close_feedback_loop (PerformanceStore)
    → next optimized_profile prefers measured best key
```

## Benchmark artifacts

| Artifact | Content |
|---|---|
| `artifacts/adaptive_compute_benchmark.json` | Synthetic identical-workload A/B |
| `artifacts/adaptive_compute_real_model_benchmark.json` | pythia-70m CPU A/B (warmed) |

Metrics reported: successes/hour, cost/success, peak GPU mem, GPU util %, throughput, completion time, communication overhead, failure rate, recovery rate.

On CPU-only hosts GPU util and peak VRAM are **None** (not zero).

### Recorded CPU pythia-70m result (warmed, 2 trials/mode)

| Mode | Successes/hr | Cost/success | Avg completion (s) | tok/s |
|---|---:|---:|---:|---:|
| unoptimized | ~9062 | ~4.17e-5 | ~0.301 | ~246 |
| optimized | ~8939 | ~3.97e-5 | ~0.286 | ~245 |

On this CPU host the advisor also selects baseline/fp32, so optimized ≈ unoptimized after warmup — the loop and harness are proven; larger deltas appear when offload/quant/CUDA/batch levers actually differ. Shared warmup avoids cold-cache bias favoring the second mode.

## Trade-offs (documented in every profile)

- **Offload:** lower peak device memory, usually slower when memory already fits.
- **int8/int4:** saves memory; on CPU often slower (see dtype-sweep writeup).
- **Larger batch:** higher throughput if memory allows; higher latency / OOM risk.
- **Distributed map:** only when cost model predicts ≥1.15× after communication.

## Not claimed

- Independent GPUs as one VRAM pool
- Tensor / pipeline parallel
- Fabricated GPU utilization on CPU hosts
