# Adaptive Compute Intelligence benchmark

- model: `EleutherAI/pythia-70m`
- real_model: True
- trials/mode: 2
- cuda_available: False

| Mode | Successes/hr | Cost/success ($) | Avg completion (s) | Peak GPU mem | GPU util % | Throughput tok/s | Fail rate | Recovery rate | Comm (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unoptimized | 9062.12 | 4.17417e-05 | 0.30054 | — | — | 246.149 | 0 | — | — |
| optimized | 8938.54 | 3.96865e-05 | 0.285743 | — | — | 244.778 | 0 | — | — |

## Deltas (optimized / unoptimized)
- completion_time_ratio_opt_over_unopt: 0.950765
- cost_per_success_ratio_opt_over_unopt: 0.950765
- successes_per_hour_ratio_opt_over_unopt: 0.986363
- throughput_ratio_opt_over_unopt: 0.99443

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
