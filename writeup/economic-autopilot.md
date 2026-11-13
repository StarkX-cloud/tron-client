# TRON Economic Autopilot

**Status:** integrated v0.1 — thin orchestration over existing TRON components.  
**Not:** a GPU marketplace, distributed VRAM layer, or cloud provisioner.

## Objective

A user submits an AI workload with budget, deadline, and reliability. Autopilot
determines resources, feasible machines, cost-minimizing configuration, memory
fit, placement, recovery contract, and predicted-vs-actual feedback — by
**connecting** four already-built capabilities:

| Capability | Existing component | Autopilot use |
|---|---|---|
| Intelligent placement | `tron_router`, `tron/spine/matcher`, topology | candidates + pooled handoff |
| GPU / memory optimization | `tron_efficiency.advisor` + policies | resource/config recommendation |
| Execution safety | `queue_server` watchdog, `MAX_JOB_REQUEUE_ATTEMPTS`, `required_memory_gb` | declared safety contract; forward hard memory |
| Real cost measurement | user hourly rate × wall time; optional `tron_costtrace` for API $ | predicted vs actual ledger |

## Architecture

```
WorkloadSpec (budget, deadline, reliability, hourly_rate)
        │
        ▼
┌───────────────────┐     recommend()      ┌────────────────────┐
│ EconomicAutopilot │◄────────────────────►│ tron_efficiency    │
│ plan()            │     query_pool /     │ advisor + hardware │
│                   │     decide_placement │                    │
│                   │◄────────────────────►│ tron_router        │
└─────────┬─────────┘                      └────────────────────┘
          │ candidates + rationale + safety_summary()
          ▼
   WorkloadPlan ──► DecisionLog (JSONL)
          │
          ▼
   execute_plan()
      ├─ local  → run callable in-process
      └─ pooled → tron_router.handoff.submit_to_pool
                    → queue_server / matcher / workers
                    → watchdog requeue on worker death
          │
          ▼
   ExecutionOutcome (actual cost/runtime/memory/success)
          │
          ▼
   RuntimeHistory (measured means for next prediction)
```

### Component interfaces

- **`WorkloadSpec`** — user intent + economic constraints (`tron_autopilot/types.py`)
- **`EconomicAutopilot.plan(spec, strategy=)`** — builds candidates, selects one
- **`execute_plan(plan, func=...)`** — acts on the plan without re-deciding
- **`DecisionLog` / `RuntimeHistory`** — audit trail + measured prediction fuel
- **Strategies** — `fixed` | `cheapest` | `autopilot` share one executor

### What Autopilot deliberately does *not* do

- Rewrite matcher / advisor / watchdog
- Provision `dedicated` machines or call a cloud GPU API
- Claim distributed VRAM aggregation
- Checkpoint arbitrary cloudpickle callables (not technically supported)
- Fabricate runtimes when no history and no `expected_runtime_seconds` exist
  (predicted cost stays `None`; selection falls back to router target among feasible options)

## Validation

```bash
# Unit + e2e (includes live uvicorn master + recovery evidence)
python3 -m pytest tests/test_autopilot_engine.py tests/test_autopilot_e2e.py -q

# Repeatable benchmark (synthetic workload body; real decision path)
python3 -m benchmarks.economic_autopilot --trials 3 --out artifacts/economic_autopilot_benchmark.json

# Same with a real same-host worker pool
python3 -m benchmarks.economic_autopilot --live-pool --trials 2 \
  --out artifacts/economic_autopilot_benchmark_live_pool.json
```

Metrics reported: cost per successful workload, completion time, peak memory
(local), failure rate, recovery requeue counts.

### Preserved results (this environment)

**No live pool** (`artifacts/economic_autopilot_benchmark.json`, 3 trials):
all three strategies succeed locally at ~0.050s; cost/success ≈ $7.0e-6 at
$0.50/hr (pool unreachable → cheapest/autopilot cannot take the cheaper
pooled rate).

**Same-host live pool** (`artifacts/economic_autopilot_benchmark_live_pool.json`,
2 trials, pooled rate $0.20/hr):

| Strategy | Successes | Cost/success | Avg completion |
|---|---:|---:|---:|
| fixed | 2 | ~$7.0e-6 | ~0.051s (local baseline) |
| cheapest | 2 | ~$1.8e-5 | ~0.28s (pooled; queue overhead) |
| autopilot | 2 | ~$7.0e-6 | ~0.050s (local after feedback) |

Cheapest naively routes to the lower hourly rate and pays more in wall-clock
once pool overhead is measured. Autopilot’s `RuntimeHistory` absorbs that
~0.28s pooled sample from the cheapest trials and then prefers local for the
same synthetic job — predicted-vs-actual feedback changing the next decision,
not a hard-coded “always local” rule.

**Failure recovery evidence:** `tests/test_autopilot_e2e.py::test_autopilot_recovery_after_dead_worker`
— dead worker claims a pooled Autopilot job, watchdog requeues (`requeued`
spine event + `requeue_count >= 1`), second worker completes, original future
resolves.

## Remaining limitations

1. Synthetic benchmark body ≠ full LLM inference; wire `tron_efficiency.workloads.runner`
   for GPU runs when deps + hardware allow (env-gated).
2. Same-host `--live-pool` is not a multi-region fleet proof (real WAN tests
   already exist for `tron_router` separately).
3. Hourly rates are **user-configured**, not live cloud list prices.
4. First-run cost predictions need history or `expected_runtime_seconds`.
5. Worker peak memory is not visible to the submitter on pooled runs.
6. No malicious-worker defense on the pool path (`reliability=strict` excludes pool).

## Recommended next engineering milestone

**Autopilot ↔ real efficiency workload loop on CUDA:** execute the advisor’s
chosen `BaselinePolicy` / `OffloadPolicy` through Autopilot end-to-end on a
real GPU, feed measured tok/s and peak VRAM into `RuntimeHistory`, and publish
one fixed/cheapest/autopilot comparison on `pythia-410m`/`1.4b` with the same
honesty standard as `writeup/compute-efficiency-gpu-validation.md`.
