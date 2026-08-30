"""Wires Phase 3's local-SGD loop into the Phase 1 execution spine, so a
training run produces real Task/Event/Artifact data — and, through that,
is visible in the Phase 4 Grid — instead of only living inside
benchmark.py's Python objects.

This does not change the training math. run_local_sgd_with_spine reuses
the exact same private helpers (_shared_init_replicas, _sample_batch,
_average_flat_params) as local_sgd.py's uninstrumented
train_local_sgd — it adds recording around the same computation, not a
parallel reimplementation of it. tests/test_spine_integration.py proves
the two produce bit-for-bit identical final models given the same inputs.

Each shard's work in each round becomes one Task: queued -> assigned ->
started -> completed, with the shard's resulting weights stored as a
content-addressed Artifact and referenced by the "completed" event's
output_hash. This is deliberately the same event vocabulary
queue_server.py's job lifecycle uses, so the Grid needs no special case
for "this is a training run" — it's just more events in the same log.
"""
from __future__ import annotations

from typing import Callable

from tron.spine import ArtifactStore, EventLog, Task

from .local_sgd import BYTES_PER_FLOAT64, _average_flat_params, _sample_batch, _shared_init_replicas
from .model import TinyMLP


def run_local_sgd_with_spine(
    shards: list[tuple],
    model_factory: Callable[[], TinyMLP],
    num_rounds: int,
    local_steps: int,
    lr: float,
    event_log: EventLog,
    artifact_store: ArtifactStore,
) -> dict:
    """Same algorithm as local_sgd.train_local_sgd, plus recording every
    shard's per-round work into `event_log` and `artifact_store`. Shards
    are recorded as nodes named "shard-0", "shard-1", ... — register
    those as workers (e.g. via /register_worker) before pointing the
    Grid at this run if you want them to show up with worker metadata;
    the spine log itself doesn't require that.
    """
    models = _shared_init_replicas(model_factory, len(shards))
    num_params = models[0].num_params()
    shard_node_ids = [f"shard-{i}" for i in range(len(shards))]

    fn_artifact = artifact_store.put(b"tron.training.local_sgd.round")
    round_input_hash = artifact_store.put(models[0].get_flat_params().tobytes()).artifact_id

    # A task's identity must depend on which shard's data it trains on,
    # not just the round's shared starting point — otherwise every shard
    # in the same round shares the same (fn_hash, input_hashes, attempt)
    # and Task.compute_id collapses them onto one task id. Hashing each
    # shard's actual data once (it doesn't change across rounds) fixes
    # this and is also the more correct semantics: different data is a
    # different input, full stop, independent of any attempt counter.
    shard_data_hashes = [artifact_store.put(x.tobytes() + y.tobytes()).artifact_id for x, y in shards]

    comm_bytes = 0

    for round_idx in range(num_rounds):
        for shard_idx, (m, shard, node_id) in enumerate(zip(models, shards, shard_node_ids)):
            x, y = shard
            task = Task.new(
                fn_hash=fn_artifact.artifact_id,
                input_hashes=(round_input_hash, shard_data_hashes[shard_idx]),
                attempt=round_idx,
                metadata={"round": round_idx, "shard": shard_idx},
            )
            event_log.append(
                task.task_id, "queued",
                {"fn_hash": fn_artifact.artifact_id, "input_hashes": [round_input_hash, shard_data_hashes[shard_idx]]},
            )
            event_log.append(task.task_id, "assigned", {"node_id": node_id}, node_id=node_id)
            event_log.append(task.task_id, "started", {"node_id": node_id}, node_id=node_id)

            for local_step in range(local_steps):
                global_step = round_idx * local_steps + local_step
                batch_x, batch_y = _sample_batch(x, y, global_step)
                m.train_step(batch_x, batch_y, lr)

            output_artifact = artifact_store.put(m.get_flat_params().tobytes())
            event_log.append(
                task.task_id, "completed",
                {"output_hash": output_artifact.artifact_id}, node_id=node_id,
            )

        averaged = _average_flat_params(models)
        for m in models:
            m.set_flat_params(averaged.copy())
        comm_bytes += len(shards) * num_params * BYTES_PER_FLOAT64
        round_input_hash = artifact_store.put(averaged.tobytes()).artifact_id

    return {
        "model": models[0],
        "comm_bytes": comm_bytes,
        "num_syncs": num_rounds,
        "shard_node_ids": shard_node_ids,
    }
