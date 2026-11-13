# TRON Distributed Execution Engine v1

**Status:** implemented on Autopilot path (independent task parallelism).  
**Audit basis:** `writeup/distributed-execution-blueprint.md` — first multi-machine approach = independent map shards, not tensor/pipeline parallel.

## What it does

When a `WorkloadSpec` sets `parallelizable=True` and supplies `partition_items`, Autopilot may choose `target="distributed"` **only if** the communication-aware cost model predicts ≥1.15× speedup vs serial after latency + transfer overhead.

Flow:

1. **Partition** — contiguous deterministic shards (`tron_autopilot/distributed/partition.py`)
2. **Cost model** — refuse when serial runtime unknown or comm dominates (`cost_model.py`)
3. **Coordinate** — submit each shard via existing `submit_to_pool` / matcher / watchdog (`coordinator.py`)
4. **Idempotent collect** — first success per `idempotency_key` wins; retries cannot double-apply
5. **Reduce** — ordered merge (`ops.reduce_map_results`); reference op `sum_squares` has closed-form check
6. **Feedback** — wall time × hourly rate → `actual_cost_usd`; history via `record_feedback`

Autopilot still chooses among **local | pooled | distributed** by predicted cost. More workers ≠ automatic win.

## Files

| Module | Role |
|---|---|
| `tron_autopilot/distributed/partition.py` | shard split + idempotency keys |
| `tron_autopilot/distributed/cost_model.py` | parallel vs serial with comm |
| `tron_autopilot/distributed/ops.py` | map shard + reduce |
| `tron_autopilot/distributed/coordinator.py` | submit / wait / retry / reduce |
| `tron_autopilot/engine.py` | `_build_distributed_candidate` |
| `tron_autopilot/execute.py` | `_execute_distributed` |
| `tron_autopilot/api.py` | plan/run fields for parallel map |

## Tests (incremental)

1. `tests/test_distributed_partition_cost.py` — partition + cost model (no network)
2. `tests/test_distributed_engine_candidate.py` — Autopilot feasibility rules
3. `tests/test_distributed_coordinator_e2e.py` — live master + **2 worker processes** (same host; not multi-machine proof)

## Multi-machine proof

`scripts/distributed_multi_machine_proof.py` provisions **two Linode CPU Nanodes**, runs master+worker0 on node0 and worker1 on node1, executes Autopilot distributed `sum_squares`, verifies closed-form total, writes `artifacts/distributed_multi_machine_proof.json`, destroys both instances.

Requires `LINODE_TOKEN` (CPU Nanodes work even when GPU plans are account-gated).

## Honesty

- Same-host two-process e2e is **coordination proof**, not multi-machine proof.
- Multi-machine claim requires the Linode artifact (distinct host IPs + `multi_machine: true`).
- No distributed VRAM / tensor parallel claimed.
