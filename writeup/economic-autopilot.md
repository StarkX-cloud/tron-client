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

1. Real-model Autopilot execution is **local-only** today; pooled real-model
   needs workers that actually ship torch/transformers (not assumed).
2. This development host has **no CUDA** — GPU Autopilot numbers are not
   claimed here; re-run `--real-model` on a CUDA machine for VRAM metrics.
3. Same-host `--live-pool` is not a multi-region fleet proof (real WAN tests
   already exist for `tron_router` separately).
4. Hourly rates are **user-configured**, not live cloud list prices.
5. First-run cost predictions need history or `expected_runtime_seconds`
   (cold Hub/cache loads skew the first trial).
6. No malicious-worker defense on the pool path (`reliability=strict` excludes pool).
7. Checkpoint recovery still unsupported for arbitrary callables.

## Real efficiency bridge (this iteration)

`tron_autopilot/efficiency_exec.py` wires advisor-chosen `baseline` /
`offload` into real `BaselinePolicy` / `OffloadPolicy` + `run_workload`.
CLI/benchmark: `--real-model`. Measured load+e2e, tok/s, and peak CPU/GPU
memory flow into `ExecutionOutcome.metrics` and `RuntimeHistory`.

Preserved CPU run (`artifacts/economic_autopilot_real_model_benchmark.json`,
`EleutherAI/pythia-70m`, 8 tokens, `cuda_available=False` on this host):

```bash
python3 -m benchmarks.economic_autopilot --real-model --trials 2 --max-new-tokens 8
```

First trial of a cold process pays Hub/cache load cost; later trials reflect
steady-state. Autopilot does **not** claim CUDA results on this machine.

Local-only for real models: pool workers are not assumed to ship
torch/transformers (stated in outcomes if a pooled plan is asked to run
efficiency).

## Recommended next engineering milestone

**Same Autopilot path on real CUDA:** re-run `--real-model` on a T4 (or
similar) with `pythia-410m` / `1.4b`, confirm advisor selects `device=cuda`,
capture peak VRAM in `metrics`, and publish fixed/cheapest/autopilot with
the honesty bar of `writeup/compute-efficiency-gpu-validation.md`. Optional
follow-on: advertise efficiency runtime on workers so pooled real-model
execution is honest rather than assumed.
