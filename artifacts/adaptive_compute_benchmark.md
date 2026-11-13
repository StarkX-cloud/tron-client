# Adaptive Compute Intelligence benchmark

- model: `synthetic/adaptive`
- real_model: False
- trials/mode: 3
- cuda_available: False

| Mode | Successes/hr | Cost/success ($) | Avg completion (s) | Peak GPU mem | GPU util % | Throughput tok/s | Fail rate | Recovery rate | Comm (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unoptimized | 11246.6 | 1.12506e-05 | 0.0810046 | — | — | — | 0 | — | — |
| optimized | 12857.8 | 1.11678e-05 | 0.0804085 | — | — | — | 0 | — | — |

## Deltas (optimized / unoptimized)
- completion_time_ratio_opt_over_unopt: 0.992641
- cost_per_success_ratio_opt_over_unopt: 0.992641
- successes_per_hour_ratio_opt_over_unopt: 1.14325
- throughput_ratio_opt_over_unopt: —

## Trade-offs
### unoptimized
- maximum simplicity; may OOM on tight memory or under-use GPU
- no quantization — higher memory, often higher CPU quality
### optimized
- —

## Honesty
- Same workload and host for both modes.
- Unoptimized = fixed baseline/fp32/batch=1.
- Optimized = advisor + measured performance history + batch/device levers.
- Real-model runs share a discarded warmup to avoid cold-cache bias.
- Does not claim unified VRAM across independent GPUs.
- GPU util / peak VRAM are None when CUDA/NVML unavailable.
