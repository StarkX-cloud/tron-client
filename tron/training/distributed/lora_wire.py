"""LoRA adapters over the `tron/training/distributed/` transport.

This is the adapter counterpart of `protocol.py` + `shard_client.run_shard`:
the numpy path sends a flat float64 parameter vector each round; this
sends a LoRA **adapter state dict** — a few hundred KB against the base
model's hundreds of MB — and nothing else crosses the wire. The base
model is assumed already present on every node (a frozen foundation
model), exactly the situation the "low-rank delta is the unit of
communication" thesis is about; only `make_lora_model(base, seed)` is
re-derived locally, which is deterministic.

torch / transformers / peft are imported at module scope, so callers that
must also run without the training extras (queue_server.py) import this
lazily.

Parity: `run_lora_shard` drives the identical
`make_lora_model` -> per-round (`set_peft_model_state_dict` -> `train_steps`)
-> `average_adapter_states` sequence as `lora_demo.run_local_sgd_lora`,
with a long-lived optimizer across rounds. `tests/test_lora_over_wire.py`
asserts the final adapter matches tensor-for-tensor.
"""
from __future__ import annotations

import io
import time
from typing import Callable, Optional

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from ..lora_demo import average_adapter_states, make_lora_model, train_steps


def encode_state_dict(state: dict) -> bytes:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in state.items()}, buf)
    return buf.getvalue()


def decode_state_dict(data: bytes) -> dict:
    return torch.load(io.BytesIO(data), map_location="cpu")


def average_state_dict_blobs(blobs: list[bytes]) -> bytes:
    """Barrier-average a round's per-shard adapter state dicts. Uses
    lora_demo.average_adapter_states — the same elementwise mean the
    in-process run uses — so a distributed round matches it exactly."""
    if not blobs:
        raise ValueError("no adapter states to average")
    return encode_state_dict(average_adapter_states([decode_state_dict(b) for b in blobs]))


def encode_blocks(blocks: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(blocks.detach().cpu().contiguous(), buf)
    return buf.getvalue()


def decode_blocks(data: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(data), map_location="cpu")


def run_lora_shard(
    transport,
    session_id: str,
    shard_idx: int,
    *,
    base_model_factory: Callable[[], object],
    poll_interval: float = 0.05,
    max_wait_seconds: float = 300.0,
    node_id: Optional[str] = None,
) -> dict:
    """One LoRA shard's full run over the wire. `base_model_factory()`
    returns the frozen base model this node already has (for the CLI:
    load Pythia-70M from the HF cache; for tests: a tiny stand-in)."""
    meta = transport.get_json(f"/training/session/{session_id}")
    if meta.get("model_kind") != "lora":
        raise ValueError(f"session {session_id} is not a LoRA session (model_kind={meta.get('model_kind')!r})")
    num_rounds = int(meta["num_rounds"])
    local_steps = int(meta["local_steps"])
    lr = float(meta["lr"])
    seed = int(meta.get("seed", 0))

    blocks = decode_blocks(_await_bytes(
        transport, f"/training/session/{session_id}/shard/{shard_idx}/data",
        poll_interval, max_wait_seconds,
    ))

    torch.manual_seed(seed)
    model = make_lora_model(base_model_factory(), seed=seed)
    optimizer = None

    for round_idx in range(num_rounds):
        init_blob = _await_bytes(
            transport,
            f"/training/session/{session_id}/round/{round_idx}/init?shard={shard_idx}",
            poll_interval, max_wait_seconds,
        )
        set_peft_model_state_dict(model, decode_state_dict(init_blob))
        optimizer = train_steps(
            model, blocks, local_steps, lr, optimizer, step_offset=round_idx * local_steps
        )
        path = f"/training/session/{session_id}/round/{round_idx}/shard/{shard_idx}"
        if node_id:
            path += f"?node_id={node_id}"
        transport.post_bytes(path, encode_state_dict(get_peft_model_state_dict(model)))

    return transport.get_json(f"/training/session/{session_id}")


def _await_bytes(transport, path: str, poll_interval: float, max_wait: float) -> bytes:
    waited = 0.0
    while True:
        blob = transport.get_bytes(path)
        if blob is not None:
            return blob
        if waited >= max_wait:
            raise TimeoutError(f"barrier at {path} did not clear within {max_wait}s")
        time.sleep(poll_interval)
        waited += poll_interval
