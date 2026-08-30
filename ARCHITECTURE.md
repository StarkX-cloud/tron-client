# TRON Architecture

TRON is being rebuilt around one bet: **an execution substrate that runs
real training and compute workloads across heterogeneous, unreliable nodes
with near-zero communication overhead — with a live, causal 3D replay of
what the cluster is doing.**

This is not a rental-compute marketplace. That was an earlier direction
(billing, royalties, customer accounts) and it has been removed — see
"What was cut" below.

## The spine

Everything else depends on one mechanism: every task, every input, every
result is content-addressed, and every state transition is an event in an
append-only, replayable log.

```
tron/spine/
  model.py   Task, Artifact, Event — pure data + hashing, no I/O
  store.py   ArtifactStore — content-addressed blob storage, dedup for free
  log.py     EventLog — append-only SQLite log; replay(), snapshot(), recovery
```

One mechanism, four consequences:

- **Deduplication.** Identical function + identical inputs hash to the same
  `task_id` (see `Task.compute_id`). Re-submitting completed work is a cache
  hit, not duplicate compute.
- **Lineage-based fault recovery.** A task's recorded `(fn_hash,
  input_hashes)` is enough to re-derive it on another node — no checkpoint
  needed. `EventLog.recover_orphaned_tasks(dead_node_id)` returns exactly
  what a watchdog needs to requeue.
- **Replay and time-travel.** `EventLog.snapshot(up_to_seq=N)` reconstructs
  cluster state as of any point in the log. This is what a debugger — and
  later the 3D Grid — reads from; neither needs a separate data source.
- **A single source of truth.** `queue_server.py` no longer trusts its own
  in-memory job dict for anything durable; the dict is a live-scheduling
  convenience, the log is the record.

## Phases

**Phase 1 — Execution spine.** Done (`tron/spine/`, wired into
`queue_server.py`'s `/submit`, `/next_job`, `/complete`, plus
`/spine/events` and `/spine/task/{id}` for reading the log directly). The
watchdog in `queue_server.py` now recovers dead workers' in-flight tasks via
`recover_orphaned_tasks` instead of trusting in-memory state alone.

**Phase 2 — Heterogeneous fault-tolerant fabric.** Partially built.
`tron/spine/topology.py`'s `TopologyMap` records real, worker-self-reported
heartbeat round-trip latency (EWMA-smoothed, ages out on silence).
`tron_runtime/global_brain.py` is no longer a stub — it scores placement
from that measured latency. The other 11 stub modules that queue_server.py
instantiated but never called (`market_engine.py`, `auto_scaler.py`,
`predictor_engine.py`, etc.) have been deleted rather than filled in — they
had no concrete role.

What's *not* solved: `/next_job` is worker-pull (a worker asks for its best
job; the master answers from that worker's queue view alone), so there is
no point where two workers' scores for the same job are ever compared — a
latency penalty that's constant across every job in one call cannot change
which job wins argmax, full stop. The current mitigation scales the penalty
by the job's `compute_weight` (a slow-link worker is steered toward light
jobs, which is real and tested — see `tests/test_queue_server_topology.py`)
but that is not the same as "the closest worker gets the job." Actually
solving that needs a master-side match step run periodically over all idle
workers and all queued jobs at once — an assignment problem, not N
independent per-worker argmax calls. That's Phase 2b; see ROADMAP.md.

**Phase 3 — Distributed training demo.** Not started. The actual
"get noticed" artifact: DiLoCo-style local SGD (hundreds of local steps
between syncs) plus weight-space merging (TIES/task arithmetic) as the
zero-communication fallback, training or fine-tuning a real model across
heterogeneous nodes. Deliverable is one benchmarked number — communication
bytes and wall-clock vs. naive all-reduce on the same hardware — with a
reproducible repo, not a platform claim.

**Phase 4 — The 3D Grid.** Not started, and deliberately last. A renderer
over `EventLog.replay()` / `snapshot()`: node distance from measured
topology, pipe width from throughput, node size from load. Time-scrubbable.
3D code *authoring* is explicitly out of scope — see rationale below.

## Why 3D observation, not 3D authoring

Text is already a dense, well-understood 2D representation of logic;
floating code in 3D space has failed every time it's been tried, for that
reason. What's genuinely hard to see in 2D is a DAG of thousands of tasks
across dozens of heterogeneous nodes, evolving over time, with data flowing
between them — that's inherently spatial-plus-time, and that's what Phase 4
targets. Interaction (click a node to inspect, drag a stage to re-place it)
comes after the passive replay view works, not before.

## What was cut, and why

The repo previously carried, in parallel with the above:

- A **compute-rental marketplace layer**: per-job billing, a 15%/85%
  royalty split, Stripe/Paystack/Flutterwave/stablecoin payment providers,
  customer accounts and API keys, invoice generation. This was explored
  early as a monetization idea, not the actual goal — it added a second
  trust model (paying strangers to run your code on their machines) that
  directly conflicts with the cloudpickle-based remote execution TRON's SDK
  relies on, and it was never exercised by anything but self-generated test
  traffic. Removed: `tron_billing.py`, `payment_providers.py`,
  `stripe_config.py`, `tron_enterprise_core.py`, `master_scheduler.py` (a
  second, incompatible scheduler with its own $/hr pricing model and its
  own SQLite ledger), the Streamlit royalty dashboard, and all associated
  tests and docs.
- **Three duplicate copies** of the runtime, GPU cluster, and TRON-II
  orchestrator packages (`tkron/` + `tron/_runtime/`; `vgpu/` + `tron/gpu/`;
  `TRON-II/tron_ii/` + `tron/orchestrator/`), left over from an earlier
  "unify everything under `tron/`" pass that copied instead of moved. One
  canonical copy of each remains under `tron/`.
- **~12 stub "engine" modules** (`market_engine.py`, `global_brain.py`,
  `auto_scaler.py`, etc.) at the repo root that returned `None` or a flat
  constant and were not imported by anything — dead weight, not
  architecture. The *active* stub layer, `tron_runtime/`, is kept because
  `queue_server.py` genuinely imports it; it is Phase 2's job to make it
  real rather than delete it out from under the server.
- **~25 status/marketing markdown files** (`PRODUCTION_READY.md`,
  `LAUNCH_READY.md`, `VALIDATION_COMPLETE.md`, `YOUR_GOAL_ACHIEVED.md`, and
  others) claiming a production-ready state the code didn't support —
  replaced by this file, `README.md`, and `ROADMAP.md`.
- Committed secrets and junk that should never have been tracked: a live
  worker auth token, three SQLite databases, a 1.2MB log file, and two
  scratch patch-note text files.

None of this was wrong to have built — the billing layer, in particular, is
a legitimate idea that just isn't the goal. It's removed so the repository
reflects one thesis instead of two, and so a stranger cloning it sees the
actual state of the system rather than four documents asserting it's done.
