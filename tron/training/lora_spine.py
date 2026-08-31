"""Wire the LoRA local-SGD run (tron/training/lora_demo.py) into the
Phase 1 execution spine — the adapter equivalent of
spine_integration.run_local_sgd_with_spine, which only covered the numpy
MLP.

Each shard's per-round adapter training becomes one spine Task
(queued -> assigned -> started -> completed), with that shard's LoRA
adapter state dict — a few hundred KB, not the ~280MB full model — stored
as the completed event's output Artifact. So the Grid renders a LoRA run
with the same event vocabulary as everything else, and the "low-rank
delta is the unit of communication" claim is visible in the log:
`transfer_bytes` on each completed event is the adapter size, and the
task metadata carries the full-model size alongside it for contrast.

This module imports torch/peft at module scope (like lora_demo.py) and is
therefore imported lazily by callers that must also run without the
training extras installed (e.g. queue_server.py).

Parity: `run_local_sgd_lora_with_spine` reuses lora_demo's own helpers
(`make_lora_model`, `train_steps`, `average_adapter_states`, ...) in the
identical order as `lora_demo.run_local_sgd_lora`; the recording is pure
side effect. tests/test_lora_spine.py asserts the two produce the same
final adapter tensor-for-tensor.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Optional

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from tron.spine import ArtifactStore, EventLog, Task

from .lora_demo import (
    adapter_state_bytes,
    average_adapter_states,
    evaluate_loss,
    full_model_bytes,
    make_lora_model,
    train_steps,
)

_ROUND_FN_TAG = b"tron.training.lora_spine.round"


def encode_adapter_state(state: dict) -> bytes:
    """A LoRA adapter state dict -> bytes. torch.save into a buffer; the
    inverse is decode_adapter_state. Used for the spine Artifact and,
    later, for sending an adapter over the wire."""
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in state.items()}, buf)
    return buf.getvalue()


def decode_adapter_state(data: bytes) -> dict:
    return torch.load(io.BytesIO(data), map_location="cpu")


def _shard_data_bytes(blocks: torch.Tensor) -> bytes:
    return blocks.detach().cpu().numpy().tobytes()


def run_local_sgd_lora_with_spine(
    base_model,
    shards: list,
    held_out_blocks,
    num_rounds: int,
    local_steps: int,
    lr: float,
    event_log: EventLog,
    artifact_store: ArtifactStore,
    seed: int = 0,
    node_ids: Optional[list] = None,
    outcome_log=None,
    run_name: str = "lora-local-sgd",
) -> dict:
    """Same computation as lora_demo.run_local_sgd_lora, plus recording
    every shard's per-round adapter training into the spine.

    Returns a dict with eval_loss_before/after, comm_bytes (adapter bytes
    summed the same way lora_demo does), num_syncs, adapter_bytes,
    full_model_bytes, final_model, and shard_node_ids.
    """
    torch.manual_seed(seed)
    models = [make_lora_model(base_model, seed=seed) for _ in shards]

    shared_state = get_peft_model_state_dict(models[0])
    for m in models[1:]:
        set_peft_model_state_dict(m, shared_state)

    adapter_bytes = adapter_state_bytes(models[0])
    full_bytes = full_model_bytes(models[0])
    eval_before = evaluate_loss(models[0], held_out_blocks)

    shard_node_ids = node_ids or [f"shard-{i}" for i in range(len(shards))]
    fn_hash = artifact_store.put(_ROUND_FN_TAG).artifact_id
    shard_data_hashes = [
        artifact_store.put(_shard_data_bytes(blocks)).artifact_id for blocks in shards
    ]
    # The vector each shard starts a round from: round 0 = shared init.
    round_init_hash = artifact_store.put(encode_adapter_state(shared_state)).artifact_id

    optimizers = [None] * len(models)
    comm_bytes = 0

    for round_idx in range(num_rounds):
        for i, (m, blocks) in enumerate(zip(models, shards)):
            node_id = shard_node_ids[i]
            task = Task.new(
                fn_hash=fn_hash,
                input_hashes=(round_init_hash, shard_data_hashes[i]),
                attempt=round_idx,
                metadata={
                    "round": round_idx, "shard": i, "kind": "lora-local-sgd",
                    "adapter_bytes": adapter_bytes, "full_model_bytes": full_bytes,
                },
            )
            event_log.append(
                task.task_id, "queued",
                {"fn_hash": fn_hash,
                 "input_hashes": [round_init_hash, shard_data_hashes[i]],
                 "metadata": {
                     "round": round_idx, "shard": i, "kind": "lora-local-sgd",
                     "adapter_bytes": adapter_bytes, "full_model_bytes": full_bytes,
                 }},
            )
            event_log.append(task.task_id, "assigned", {"node_id": node_id}, node_id=node_id)
            event_log.append(task.task_id, "started", {"node_id": node_id}, node_id=node_id)

            optimizers[i] = train_steps(
                m, blocks, local_steps, lr, optimizers[i], step_offset=round_idx * local_steps
            )

            adapter_artifact = artifact_store.put(encode_adapter_state(get_peft_model_state_dict(m)))
            event_log.append(
                task.task_id, "completed",
                {"output_hash": adapter_artifact.artifact_id, "transfer_bytes": adapter_bytes},
                node_id=node_id,
            )

        states = [get_peft_model_state_dict(m) for m in models]
        averaged = average_adapter_states(states)
        for m in models:
            set_peft_model_state_dict(m, averaged)
        comm_bytes += len(shards) * adapter_bytes
        round_init_hash = artifact_store.put(encode_adapter_state(averaged)).artifact_id

    eval_after = evaluate_loss(models[0], held_out_blocks)

    if outcome_log is not None:
        # capability gained = eval-loss reduction; compute spent = total
        # local steps across shards. Feeds tron/orchestrator/outcomes.py.
        gain = float(eval_before - eval_after)
        cost = float(num_rounds * len(shards) * local_steps)
        try:
            from tron.orchestrator.outcomes import TrainingOutcome
            outcome_log.record(TrainingOutcome(
                artifact_id=round_init_hash,  # last merged adapter
                adapter_name=run_name,
                module_id="lora-local-sgd",
                expected_capability_gain=gain,
                actual_capability_gain=gain,
                expected_cost=cost,
                actual_cost=cost,
                success=gain > 0.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
        except Exception:
            pass

    return {
        "eval_loss_before": eval_before,
        "eval_loss_after": eval_after,
        "comm_bytes": comm_bytes,
        "num_syncs": num_rounds,
        "adapter_bytes": adapter_bytes,
        "full_model_bytes": full_bytes,
        "final_model": models[0],
        "shard_node_ids": shard_node_ids,
    }
