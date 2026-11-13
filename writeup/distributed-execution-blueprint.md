# TRON — Architecture Audit & Distributed Execution Blueprint

**Status:** audit only (Prompt 1). No systems rewritten. No new capabilities invented.  
**Repo evidence date:** branch `cursor/economic-autopilot-38d1` (Economic Autopilot + dashboard era).  
**Scope:** map reality → matrix → reuse → gaps → first multi-machine milestone.

---

## 1. Current architecture (workload journey)

### Diagram (API → plan → place → execute → recover → cost)

```
Client (curl / dashboard / SDK)
        │
        ▼
┌───────────────────────────┐
│ tron_autopilot.api        │  POST /plan  POST /run  GET /fleet
│ (optional X-TRON-AUTH)    │
└─────────────┬─────────────┘
              │ WorkloadSpec
              ▼
┌───────────────────────────┐
│ EconomicAutopilot.plan()  │
│  • recommend()            │──► tron_efficiency.advisor
│  • decide_placement()     │──► tron_router.decision
│  • query_pool_status()    │──► GET /workers + /topology
│  • candidates + strategy  │    fixed | cheapest | autopilot
│  • DecisionLog append     │
└─────────────┬─────────────┘
              │ WorkloadPlan.chosen  (local | pooled | dedicated*)
              ▼
┌───────────────────────────┐
│ execute_plan()            │
│  local  → efficiency_exec │──► Baseline/Offload + run_workload
│  pooled → submit_to_pool  │──► cloudpickle POST /submit
│  dedicated* → provisioner │──► opt-in Linode → then pooled
└─────────────┬─────────────┘
              │ (pooled only)
              ▼
┌───────────────────────────┐
│ queue_server              │
│  /submit → job_queue      │
│  _match_loop → matcher    │──► memory_fits + Hungarian assign
│  worker /next_job         │
│  _watchdog_loop           │──► requeue ≤ MAX_JOB_REQUEUE_ATTEMPTS
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ worker.py                 │
│  efficiency_runtime flag  │
│  run_efficiency_job OR    │
│  arbitrary cloudpickle    │
└─────────────┬─────────────┘
              │ ExecutionOutcome
              ▼
┌───────────────────────────┐
│ Cost & feedback           │
│  hourly_rate × runtime    │
│  optional costtrace JSONL │
│  RuntimeHistory.record()**│
└───────────────────────────┘

* dedicated is infeasible in plan() unless a provisioner is configured;
  with TRON_DEDICATED_PROVISIONER=linode it provisions then re-routes pooled.
** HTTP /run currently calls record_feedback(outcome) with wrong arity
   (CLI passes plan+outcome). Costtrace still records; history update via API is broken.
```

### Parallel training path (separate from Autopilot job pool)

```
shard_worker(s)  ──HTTP──►  /training/session*
     │                         │
     │  GET shard data         │  ParamServer / LoRAParamServer
     │  GET round init (202)   │  barrier-average adapters/params
     │  POST trained shard     │  spine Artifacts (+ optional encrypt)
     ▼                         ▼
 Local SGD / LoRA Local SGD     GET /training/outcomes
 (data-parallel cooperation)
```

Autopilot **does not** call `/training/session*`. Job pool and training wire are two systems on the same `queue_server` process.

---

## 2. Capability matrix (repository evidence)

| Capability | Status | Evidence |
|---|---|---|
| HTTP Autopilot API (`/plan`, `/run`, `/fleet`, `/pricing`) | **Implemented** | `tron_autopilot/api.py` |
| Dashboard console | **Implemented** | `tron_autopilot/dashboard/index.html` mounted `/dashboard` |
| Plan candidates + strategies | **Implemented** | `tron_autopilot/engine.py` |
| Local real-model efficiency | **Implemented** | `efficiency_exec.py` → `run_workload` |
| Pooled real-model efficiency | **Implemented** (needs workers with `efficiency_runtime`) | `pooled_efficiency.py`, worker advertise |
| Matcher hard memory-fit + Hungarian | **Implemented** | `tron/spine/matcher.py`, `_match_loop` |
| Watchdog requeue + cap | **Implemented** | `queue_server._watchdog_loop`, `MAX_JOB_REQUEUE_ATTEMPTS` |
| Fleet capacity snapshot | **Implemented** | `tron_autopilot/fleet.py` |
| Submit auth + token redaction | **Implemented** | `TRON_SUBMIT_AUTH_TOKEN`, `GET /workers` |
| SKU hourly table | **Implemented** (pinned snapshot, not live feed) | `tron_autopilot/pricing.py` |
| Cost = rate × wall time | **Implemented** | `store.cost_from_runtime` |
| Costtrace compute ledger | **Implemented** when `costtrace_task` set | `execute._record_costtrace` |
| RuntimeHistory feedback | **Partial** | Works via CLI; HTTP `/run` calls `record_feedback(outcome)` but signature is `(plan, outcome)` |
| Dedicated cloud provision | **Partial / experimental** | Opt-in Linode provisioner; plan marks dedicated infeasible by default |
| CUDA Autopilot (single machine) | **Implemented + proven** | Colab T4 artifact `artifacts/economic_autopilot_cuda_benchmark.json` |
| Autopilot on multi-host CUDA pool | **Not proven** | ROADMAP unchecked; no artifact |
| Matcher filter by `efficiency_runtime` | **Partial** | Plan-time count only; matcher does not hard-filter capability |
| Topology latency used in Autopilot choose local vs pool | **Partial** | Measured links exposed; Autopilot does not yet weight them as primary selector |
| Distributed training Local SGD (numpy) over HTTP | **Implemented + tested** | `tron/training/distributed/*`, `tests/test_distributed_training.py` |
| LoRA over wire (adapter average) | **Implemented + tested** | `lora_wire.py`, `tests/test_lora_over_wire.py` |
| Training auth + artifact encryption | **Implemented** | `TRON_TRAINING_AUTH_TOKEN`, `TRON_ARTIFACT_ENCRYPTION_KEY` |
| WAN training (remote master) | **Proven (experimental scale)** | `writeup/distributed-training.md` — client shards → Render master |
| ≥2 physical shard hosts WAN | **Documented path, not claimed as proven** | `examples/distributed_training/README.md` |
| Job-pool two-Linode WAN | **Proven** | `writeup/compute-routing-layer.md` |
| Tensor / pipeline / FSDP model parallel | **Not implemented** | No NCCL/FSDP/tensor-parallel runtime in tree |
| Distributed VRAM aggregation | **Not implemented** (explicitly not claimed) | Autopilot writeup |
| Secure aggregation / malicious-shard defense | **Not implemented** | distributed-training writeup |
| OpenAI GPU gateway → spine `/submit` | **Partial** | `tron/gpu` still aimed at older `/submit_job` API (ROADMAP) |

---

## 3. Reusable components (for multi-machine work)

### Already powering Autopilot (independent jobs)
- `tron_router.decide_placement` + `submit_to_pool`
- `queue_server` matcher, topology, watchdog
- Worker registration (`efficiency_runtime`, memory, SKU/rate)
- `tron_efficiency` advisor + Baseline/Offload
- Autopilot API + dashboard + cost accounting

### Built but not wired into Autopilot (cooperative training)
- `/training/session*` ParamServer / LoRAParamServer
- `shard_worker` + `RequestsTransport` (retries, auth)
- Non-IID shard factories + telecom dataset hook
- Spine artifacts + training outcomes (`capability_per_compute`)
- Bit-exact tests: in-process Local SGD ≡ HTTP path

### Proven infra patterns to copy
- Cross-region job pool (Linode) timeout/retry hardening
- Heartbeat fail-closed auth
- BACKGROUND_STOP for test isolation of match/watchdog threads

---

## 4. What TRON actually parallelizes today

| Approach | Supported now? | What it means in TRON |
|---|---|---|
| **Independent task parallelism** | **Yes (primary Autopilot path)** | Separate cloudpickle / efficiency jobs assigned to workers; no shared weights during the job |
| **Data parallelism** | **Yes (training wire)** | Shards train on data partitions; barrier-average full params or LoRA adapters (Local SGD / DiLoCo-lite) |
| **Model parallelism** (tensor / pipeline) | **No** | Each LoRA shard loads a **full** frozen base locally; only adapters move |

**Conclusion:** Autopilot distributes **independent jobs**. The training subsystem does **real multi-machine cooperation** via data-parallel Local SGD — but Autopilot does not invoke it yet.

---

## 5. Bottlenecks (from code + writeups)

| Area | Bottleneck | Source |
|---|---|---|
| Networking | WAN RTT / TLS / intermittent SSL errors; training barriers poll HTTP 202 | `writeup/distributed-training.md` |
| Networking | Pool status timeout historically too tight (~130 ms RTT) — mitigated (5 s + retry) | `writeup/compute-routing-layer.md` |
| Data movement | Job path ships cloudpickle of entire callable; training ships full vectors or small LoRA adapters | handoff + `lora_wire` |
| Synchronization | Training barrier is master-side “all shards reported”; single-process master = SPOF | distributed-training writeup |
| Memory | Every LoRA shard loads full base (~282 MB for pythia-70m); Autopilot CUDA pool unproven | writeups |
| Scheduling | Matcher ignores `efficiency_runtime`; Autopilot under-uses measured latency for local-vs-pool | matcher + economic-autopilot |
| Execution | Python minor-version cloudpickle incompatibility across heterogeneous fleet | compute-routing-layer |
| Feedback | HTTP `/run` does not correctly call `record_feedback(plan, outcome)` | `api.py:120` vs `engine.py:518` |
| Security | Auth stops strangers; authenticated poison shards / no secure aggregation | distributed-training writeup |

---

## 6. Recommended first distributed execution approach

**Recommend: Independent task parallelism first (Autopilot CUDA worker pool), then deepen data-parallel training hosts.**

### Why not model parallelism first?
- No tensor/pipeline runtime exists; building one is a rewrite-scale project.
- Conflicts with Autopilot’s honesty bar (no fake distributed VRAM).

### Why not jump straight to “N physical training shards” as Autopilot’s first mile?
- Training cooperation is already implemented but **orthogonal** to Autopilot’s `/run` path.
- Product “AI company plugs in heavy inference/batch jobs” maps to the **job pool** they already have.

### Why independent task parallelism is the smallest credible Autopilot milestone?
- Reuses matcher, watchdog, `pooled_efficiency`, fleet flags, API — **no new distributed algorithm**.
- Closes the explicitly unchecked ROADMAP item: Autopilot `--real-model` on a CUDA worker pool with peak VRAM metrics.
- Measurable: same metrics as Colab T4 artifact, but `target=pooled`, ≥1 remote GPU worker, recovery still via watchdog.

**Parallel (second track, not blocking):** prove ≥2 physical `shard_worker` hosts for LoRA Local SGD (code ready; WAN proof today is multi-shard-from-one-client → remote master).

---

## 7. Recommended first milestone (concrete)

### Milestone A — **Autopilot multi-machine inference (independent jobs)**  
**Name:** CUDA pooled Autopilot proof  
**Goal:** `POST /run` or CLI `--real-model --live-pool` completes on a fleet where at least one **remote** worker has CUDA + `efficiency_runtime=true`; artifact shows `target=pooled`, peak GPU memory, cost.

**In scope**
1. Fix HTTP `record_feedback(plan, outcome)` arity (tiny correctness fix).
2. Matcher (or pre-assign filter): refuse efficiency jobs on workers without `efficiency_runtime`.
3. Harness: master + ≥1 GPU worker (Colab worker pointing at public master, or unlocked Linode GPU, or two processes with CUDA).
4. Artifact: `artifacts/economic_autopilot_cuda_pool_benchmark.json`.
5. Optional: weight pool vs local using `/topology` latency when history exists.

**Out of scope for A**
- Tensor/pipeline parallel  
- Distributed VRAM  
- Changing Local SGD math  
- Multi-tenant sandbox  

### Milestone B — **Cooperative training on ≥2 physical shard hosts** (parallel track)
- Same LoRA wire; measure barrier wait + wire bytes across real inter-shard RTT.
- Does not block Autopilot sales path; strengthens “TRON trains across machines” claim.

---

## 8. Implementation plan (Milestone A only — still no code in this prompt)

| Step | Change | Touches | Risk |
|---|---|---|---|
| A1 | Fix `api.py` `record_feedback(plan, outcome)` + test | `tron_autopilot/api.py`, `tests/test_autopilot_api.py` | Low |
| A2 | Capability-aware assign: skip workers with `efficiency_runtime=False` when job metadata requires it (or Autopilot sets a job flag) | `queue_server` match cycle and/or submit payload; tests | Medium — must not break non-efficiency jobs |
| A3 | Benchmark harness `--real-model --live-pool` against remote GPU worker; write artifact | `benchmarks/economic_autopilot.py`, `scripts/` | Medium — needs GPU host |
| A4 | Document results in `writeup/economic-autopilot.md`; update ROADMAP checkbox | writeups | Low |
| A5 | Dashboard: show pooled plan/outcome when fleet has GPU workers | dashboard JS only | Low |

**Success criteria**
- [ ] Pooled Autopilot efficiency job succeeds with `device=cuda` on a non-localhost worker  
- [ ] Peak GPU bytes in outcome metrics  
- [ ] Watchdog recovery still green in existing tests  
- [ ] CI green; no rewrite of matcher algorithm / advisor / Local SGD  

---

## 9. Bottom line

| Question | Answer from repo |
|---|---|
| Can TRON place & run AI work across machines? | **Yes** — independent jobs via pool; **and** cooperative data-parallel training via `/training/session*`. |
| Does Autopilot already do multi-GPU shared-model inference? | **No.** |
| First credible expansion? | **Pooled CUDA Autopilot (independent task parallelism)** reusing existing spine — then physical multi-host LoRA training proof. |
| Rewrite needed? | **No** — connect and harden; do not invent a new fabric. |

---

## Key evidence index

- `tron_autopilot/{api,engine,execute,efficiency_exec,pooled_efficiency,fleet,pricing}.py`
- `tron_router/{decision,handoff}.py`
- `queue_server.py` (submit, match, watchdog, training routes)
- `worker.py`, `tron/spine/matcher.py`
- `tron/training/distributed/*`, `tron/training/local_sgd.py`
- `writeup/{economic-autopilot,compute-routing-layer,distributed-training}.md`
- `artifacts/economic_autopilot_cuda_benchmark.json`
- `ROADMAP.md` (Autopilot CUDA pool checkbox)
