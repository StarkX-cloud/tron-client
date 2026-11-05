# Benchmark report

Comparing 2 run(s).

## Runs
- `baseline` on EleutherAI/pythia-70m (cpu): OK
  - warning: torch.cuda.is_available() is False — no CUDA GPU detected on this machine. GPU-memory-budget experiments cannot be run here; only the CPU execution path is available.
- `offload` on EleutherAI/pythia-70m (cpu): OK
  - warning: torch.cuda.is_available() is False — no CUDA GPU detected on this machine. GPU-memory-budget experiments cannot be run here; only the CPU execution path is available.

## Metric comparison

| Metric | baseline | offload | Winner |
|---|---|---|---|
| Model load time (s) | 9.959 | 4.005 | offload |
| Time to first token (s) | 0.4735 | 0.3723 | offload |
| End-to-end latency (s) | 1.777 | 10.22 | baseline |
| Throughput (tokens/sec) | 18.01 | 3.131 | baseline |
| Peak CPU memory (bytes) | 4.869e+08 | 4.643e+08 | offload |
| Peak GPU memory allocated (bytes) | - | - | incomparable (missing data) |

## Estimated execution cost
Using a configured rate of $0.5000/hour (this is an input you set, not a measured cloud price):

- `baseline`: $0.000247 estimated for this run's 1.777s of execution (estimate only — not an actual customer saving).
- `offload`: $0.001420 estimated for this run's 10.222s of execution (estimate only — not an actual customer saving).

## Warnings carried from individual runs
- torch.cuda.is_available() is False — no CUDA GPU detected on this machine. GPU-memory-budget experiments cannot be run here; only the CPU execution path is available.

