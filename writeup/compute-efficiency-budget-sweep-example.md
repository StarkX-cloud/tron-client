# Budget sweep: a real tradeoff curve, not two points

The rest of this project's write-ups report one or two budget points at a
time (e.g. "100MB vs. baseline"). `tron_efficiency.cli sweep` runs the same
workload across several budgets in one pass, producing a real curve instead
of guessing at the shape between two measurements.

```bash
python -m tron_efficiency.cli sweep --model EleutherAI/pythia-70m \
    --memory-budget-mb-list 100,150,250 --max-new-tokens 16
```

**Real result** (CPU, this dev machine, `pythia-70m` ≈ 134-268MB depending
on which dtype actually loaded — see the dtype-wiring gap elsewhere in
ROADMAP.md for why that's not always obvious):

| Budget | Load (s) | TTFT (s) | E2E latency (s) | Throughput (tok/s) | Peak CPU mem |
|---|---|---|---|---|---|
| offload @ 250MB | 3.37 | 0.092 | 0.489 | 32.70 | 490MB |
| offload @ 150MB | 3.41 | 0.096 | 0.486 | 32.91 | 490MB |
| offload @ 100MB | 3.55 | 0.697 | 4.204 | 3.81 | 448MB |
| baseline | 8.61 | 0.346 | 0.472 | 33.92 | 485MB |

**What this actually shows:** a real cliff, not a smooth tradeoff. At
150MB and 250MB, offload performs essentially identically to baseline
(~33 tok/s either way) — the model's footprint fits within those budgets
without forcing meaningful disk offload. At 100MB, something crosses a
real threshold and throughput collapses **8.6x** (32.7 → 3.8 tok/s), while
peak memory only drops modestly (490MB → 448MB, ~8.6%). This matches the
pattern from the GPU validation write-up's bigger result (1.4B params on a
real T4): offload doesn't degrade gracefully as the budget tightens, it
holds steady until some point, then falls off sharply. Knowing *where*
that cliff is for a given model+hardware is exactly what a real product
built on this would need to answer automatically — this sweep capability
is the raw tool for finding it, not yet an automated cliff-detector.

**Honest scope:** one model, one machine, three budget points, one run
each (no repeated trials for variance). Good enough to demonstrate the
cliff shape is real and worth characterizing properly, not a statistically
rigorous curve.
