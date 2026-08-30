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
`tron/spine/topology.py`'s `TopologyMap` records two real, worker-self-
reported network signals, both EWMA-smoothed and aged out on silence:
heartbeat round-trip **latency**, and **bandwidth** (Phase 2c) — actual
throughput the worker measures by transferring a payload to and from the
master via `/probe/blob` and `/probe/sink`, not a hard-coded constant.
`tron_runtime/global_brain.py` is no longer a stub — it scores placement
from measured latency plus, when a job declares `transfer_bytes`, the
estimated time to ship those bytes over the measured link. The other 11
stub modules that queue_server.py instantiated but never called
(`market_engine.py`, `auto_scaler.py`, `predictor_engine.py`, etc.) have
been deleted rather than filled in — they had no concrete role.

That per-worker-argmax limitation is now actually solved, not just
mitigated: `tron/spine/matcher.py` runs a periodic master-side match step
(`_match_loop` in `queue_server.py`, every `MATCH_INTERVAL_SECONDS`) that
builds the full (job × idle-worker) score matrix and solves it with
`scipy.optimize.linear_sum_assignment` — true optimal assignment, not a
greedy "take the best pair first" heuristic. Greedy was tried first during
development and is provably wrong here: given two workers of very
different latency and two jobs of very different weight, greedy grabs the
single best-looking pair immediately and can strand the other pair far
worse off than the globally optimal pairing (worked numeric example in
`matcher.py`'s docstring; `test_optimal_assignment_beats_the_greedy_trap`
pins the correct behavior down). Results are placed in
`pending_assignment` and `/next_job` serves from there first — the
per-worker argmax loop is now purely a fallback for jobs the match cycle
hasn't reached yet, not the primary placement mechanism.

**Phase 3 — Distributed training demo.** Built at small scale in
`tron/training/`: a hand-written numpy MLP (backprop correctness verified
against numerical gradients — see `tests/test_training_model.py`),
non-IID sharded synthetic data, DiLoCo-style local SGD vs. a
sync-every-step baseline, and weight-space merging (task arithmetic +
TIES). Run `python -m tron.training.benchmark` for the report. On a
deliberately hard (not cherry-picked) non-IID split: local SGD gets 10x
less communication than the baseline for 1.6 points less accuracy;
merging with zero communication during training recovers 81% accuracy
vs. 58.8% for an unmerged solo shard. Every number is pinned as a
regression test.

This is honestly small: a few hundred parameters, synthetic data,
seconds of CPU time, all "shards" as in-process objects on one machine
(communication bytes are counted correctly but nothing is actually sent
over a wire yet). The goal at this stage was to get the algorithms
right and prove it, not to demonstrate scale — training across real
separate machines over an actual network, and scaling to a model size
anyone would call "real," is substantial, distinct future work. See
ROADMAP.md for what's left, including the public writeup this phase was
originally scoped to produce.

`tron/training/spine_integration.py` wires this into Phase 1: each
shard's per-round training is recorded as a real Task in the spine log,
with a parity test proving instrumentation doesn't change the final
model (bit-for-bit identical to the uninstrumented run).
`POST /training/run_demo` triggers it against `queue_server.py`'s own
spine — verified live, including in the Grid (see Phase 4 below).

**Over a real wire.** `tron/training/distributed/` moves the shards out
of the process: each is a separate OS process
(`python -m tron.training.distributed.shard_worker`) that serializes its
parameter vector and `POST`s it to the master over HTTP. The master
(`queue_server.py`'s `/training/session*` endpoints, backed by
`param_server.TrainingSession`) stores each vector as a content-addressed
spine Artifact, barrier-averages once all shards have reported for a
round, and serves the merge back for the next round.
`tests/test_distributed_training.py` drives the full HTTP path through
`fastapi.testclient` and asserts the final model is **bit-for-bit
identical** to the single-process `train_local_sgd` — the transport
changes where the bytes go, not what they are. Running across genuinely
separate machines is then just a different `--master` URL. This is the
numpy MLP; wiring the LoRA path through the same transport is still open
(see ROADMAP.md).

**Phase 3 scale-up — the same story on a real pretrained model.**
`tron/training/lora_demo.py` / `benchmark_lora.py` run the identical
local-SGD and merging comparison against EleutherAI's Pythia-70M via
LoRA, fine-tuned on the tiny-shakespeare corpus, instead of the
hand-rolled numpy MLP. This is what makes the "low-rank delta, not a
full checkpoint, is the unit of communication" idea from the project's
original brief concrete rather than conceptual: the LoRA adapter is
393,216 bytes against the full model's 282,099,712 — 717x smaller — and
local SGD trains on top of that with eval loss genuinely dropping
(4.3296 -> 4.2236) across 4 syncs. Zero-communication merging recovers
most of that gain (merged 4.2403 vs. 4.3226 average for an unmerged solo
shard) with no communication during training at all. Numbers are from
one recorded real run, not simulated — see ROADMAP.md for the full
figures and the honest cost: a run takes ~20 minutes on this project's
CPU-only dev hardware, so the pytest suite covers the pure logic
(shard splitting, byte accounting, adapter averaging) fast and without
a network dependency, rather than re-running the full pipeline on every
test invocation. Optional dependency group: `requirements-training.txt`.

`tron/training/lora_spine.py` wires this into the spine the same way the
numpy demo is wired: `run_local_sgd_lora_with_spine` records each shard's
per-round adapter training as a Task, with the LoRA state dict (~KB) as
the output Artifact, reusing `lora_demo.py`'s helpers so a parity test
holds it tensor-for-tensor against the uninstrumented run.
`benchmark_lora.py --spine-dir DIR` runs the real Pythia local-SGD
portion through it, so an actual LoRA run is Grid-replayable. Still open:
running LoRA across the `tron/training/distributed/` wire (adapter as the
POST body) — the numpy MLP does that, the LoRA path doesn't yet.

**Phase 4 — The 3D Grid (v1: passive replay).** Built: `tron/grid/index.html`,
served by `queue_server.py` at `/grid/`. It's a static page (three.js,
vendored locally — not a CDN reference, so it works offline and isn't at
the mercy of a CDN's availability) that fetches `/workers` and
`/spine/events` itself and renders:

- Worker distance from the master node = the same measured heartbeat
  latency `tron/spine/topology.py` and `GlobalDecisionBrain` use for
  placement — not a layout algorithm's guess. Worker height = reported
  load. Unmeasured workers sit at a fixed mid-distance (same "don't
  assume unknown is best or worst" rule as `TopologyMap.rank_nodes`).
  (Measured bandwidth is now available per worker too — Phase 2c — but
  the Grid does not yet render it as link thickness.)
- Tasks positioned at whichever worker the event log says they're
  actually assigned to, colored by status (queued/running/completed/
  failed/requeued).
- A scrubber that replays the event log to any point — this was verified
  in a real browser during development: scrubbing back to right after 6
  jobs were submitted but before any were assigned showed exactly the 6
  grey "queued" markers clustered at the master with no worker
  assignments, matching what the raw log said at that point. A "Live"
  toggle tails `/spine/events?since_seq=N` for events after that.

Verified end-to-end against a real running server with real workers at
different reported latencies (8ms / 60ms / 260ms) — their rendered
distances from the master matched the position formula exactly (7.2 /
15.0 / 45.0 world units for `BASE_RADIUS=6 + latency_ms * 0.15`).

**What v1 does not do**, deliberately deferred rather than faked:
interaction (click a task to inspect its recorded inputs, drag a worker
to see the scheduler react) — see the "why 3D observation, not 3D
authoring" rationale below for why passive replay came first. Task
motion between states is discrete per scrub step, not smoothly animated
between positions. Worker layout uses only latency (one topology
signal); bandwidth and pipe-width-from-throughput are not implemented —
there's no bandwidth prober yet (see ROADMAP.md).

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
