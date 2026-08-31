"""Master-side state for an over-the-wire distributed **LoRA** training run.

The adapter counterpart of `param_server.TrainingSession`. Deliberately a
parallel class rather than a refactor of that one: the numpy
`TrainingSession` is pinned bit-for-bit by tests and left untouched. The
barrier + spine-recording shape is the same; what differs is that the
unit crossing the wire is a LoRA adapter state dict (via
`lora_wire.encode_state_dict` / `average_state_dict_blobs`), the shard
data is tokenized text blocks, and "capability gained" is eval-loss
reduction rather than accuracy delta.

torch / transformers / peft are imported (transitively via lora_wire /
lora_demo) at module scope, so queue_server.py imports this lazily.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Callable, Optional

import torch

from tron.spine import ArtifactStore, EventLog, Task

from ..lora_demo import (
    DATA_PATH,
    SEQ_LEN,
    adapter_state_bytes,
    evaluate_loss,
    full_model_bytes,
    make_lora_model,
    make_shards,
)
from .lora_wire import (
    average_state_dict_blobs,
    decode_state_dict,
    encode_blocks,
    encode_state_dict,
)
from peft import get_peft_model_state_dict, set_peft_model_state_dict

_ROUND_FN_TAG = b"tron.training.distributed.lora_round"


class LoraTrainingSession:
    def __init__(
        self,
        session_id: str,
        *,
        base_model_factory: Callable[[], object],
        num_shards: int,
        num_rounds: int,
        local_steps: int,
        lr: float,
        event_log: EventLog,
        artifact_store: ArtifactStore,
        model_name: str = "EleutherAI/pythia-70m",
        seq_len: int = SEQ_LEN,
        corpus_path=None,
        held_out_fraction: float = 0.1,
        seed: int = 0,
        outcome_log=None,
        run_name: Optional[str] = None,
    ):
        if num_shards < 1 or num_rounds < 1 or local_steps < 1:
            raise ValueError("num_shards, num_rounds, local_steps must all be >= 1")

        self.session_id = session_id
        self.model_kind = "lora"
        self.num_shards = num_shards
        self.num_rounds = num_rounds
        self.local_steps = local_steps
        self.lr = float(lr)
        self.seed = int(seed)
        self.model_name = model_name
        self.seq_len = int(seq_len)
        # `model_config` is what a shard reads back to know what base model
        # to load — same field the numpy session exposes.
        self.model_config = {"model_name": model_name, "seq_len": self.seq_len}
        self._make_base = base_model_factory
        self._log = event_log
        self._store = artifact_store
        self._outcome_log = outcome_log
        self._run_name = run_name or f"distributed-lora-{session_id}"
        self._outcome_recorded = False
        self._lock = threading.Lock()

        self._build_problem(corpus_path=corpus_path, held_out_fraction=held_out_fraction)

        self._uploads: dict[int, dict[int, str]] = {r: {} for r in range(num_rounds)}
        self._round_init: dict[int, str] = {0: self._shared_init_hash}
        self._merged: dict[int, str] = {}
        self._task_ids: dict[tuple[int, int], str] = {}
        self._wire_bytes = 0
        # shard_idx -> the node_id it last reported from — see
        # param_server.TrainingSession's identical field for why.
        self._shard_node_ids: dict[int, str] = {}

    # -- problem setup -------------------------------------------------

    def _build_problem(self, *, corpus_path, held_out_fraction: float) -> None:
        base = self._make_base()
        tok_ids = _load_token_ids(corpus_path)
        split_at = int(len(tok_ids) * (1.0 - held_out_fraction))
        train_ids, held_out_ids = tok_ids[:split_at], tok_ids[split_at:]

        shards = make_shards(train_ids, num_shards=self.num_shards, seq_len=self.seq_len)
        self._held_out_blocks = make_shards(held_out_ids, num_shards=1, seq_len=self.seq_len)[0]

        self._shard_data_hashes = [
            self._store.put(encode_blocks(blocks)).artifact_id for blocks in shards
        ]

        torch.manual_seed(self.seed)
        init_model = make_lora_model(base, seed=self.seed)
        shared_state = get_peft_model_state_dict(init_model)
        self._adapter_bytes = adapter_state_bytes(init_model)
        self._full_model_bytes = full_model_bytes(init_model)
        self._shared_init_hash = self._store.put(encode_state_dict(shared_state)).artifact_id
        self._loss_before = evaluate_loss(init_model, self._held_out_blocks)

        self._fn_hash = self._store.put(_ROUND_FN_TAG).artifact_id

    # -- shard-facing operations ------------------------------------------

    def shard_data(self, shard_idx: int) -> bytes:
        self._check_shard(shard_idx)
        return self._store.get(self._shard_data_hashes[shard_idx])

    def round_init(self, round_idx: int, shard_idx: int) -> Optional[bytes]:
        self._check_shard(shard_idx)
        if not 0 <= round_idx < self.num_rounds:
            raise ValueError(f"round {round_idx} out of range")
        with self._lock:
            init_hash = self._round_init.get(round_idx)
            if init_hash is None:
                return None
            key = (round_idx, shard_idx)
            if key not in self._task_ids:
                meta = {
                    "session": self.session_id, "round": round_idx, "shard": shard_idx,
                    "kind": "distributed-lora-local-sgd",
                    "adapter_bytes": self._adapter_bytes,
                    "full_model_bytes": self._full_model_bytes,
                }
                task = Task.new(
                    fn_hash=self._fn_hash,
                    input_hashes=(init_hash, self._shard_data_hashes[shard_idx]),
                    attempt=round_idx,
                    metadata=meta,
                )
                self._task_ids[key] = task.task_id
                nid = f"shard-{shard_idx}"
                self._log.append(
                    task.task_id, "queued",
                    {"fn_hash": self._fn_hash,
                     "input_hashes": [init_hash, self._shard_data_hashes[shard_idx]],
                     "metadata": meta},
                )
                self._log.append(task.task_id, "assigned", {"node_id": nid}, node_id=nid)
                self._log.append(task.task_id, "started", {"node_id": nid}, node_id=nid)
            self._wire_bytes += self._adapter_bytes
            return self._store.get(init_hash)

    def submit_round(self, round_idx: int, shard_idx: int, adapter_bytes_blob: bytes,
                     node_id: Optional[str] = None) -> dict:
        self._check_shard(shard_idx)
        if not 0 <= round_idx < self.num_rounds:
            raise ValueError(f"round {round_idx} out of range")
        # validate it decodes to a state dict
        state = decode_state_dict(adapter_bytes_blob)
        if not isinstance(state, dict) or not state:
            raise ValueError("submitted adapter did not decode to a non-empty state dict")

        with self._lock:
            art = self._store.put(encode_state_dict(state))
            self._uploads[round_idx][shard_idx] = art.artifact_id
            self._wire_bytes += self._adapter_bytes

            key = (round_idx, shard_idx)
            task_id = self._task_ids.get(key)
            nid = node_id or f"shard-{shard_idx}"
            if task_id is None:
                meta = {
                    "session": self.session_id, "round": round_idx, "shard": shard_idx,
                    "kind": "distributed-lora-local-sgd",
                    "adapter_bytes": self._adapter_bytes, "full_model_bytes": self._full_model_bytes,
                }
                task = Task.new(
                    fn_hash=self._fn_hash,
                    input_hashes=(self._round_init[round_idx], self._shard_data_hashes[shard_idx]),
                    attempt=round_idx, metadata=meta,
                )
                task_id = task.task_id
                self._task_ids[key] = task_id
                self._log.append(task_id, "queued", {"fn_hash": self._fn_hash, "metadata": meta})
                self._log.append(task_id, "assigned", {"node_id": nid}, node_id=nid)
                self._log.append(task_id, "started", {"node_id": nid}, node_id=nid)

            self._shard_node_ids[shard_idx] = nid
            self._log.append(
                task_id, "completed",
                {"output_hash": art.artifact_id, "transfer_bytes": self._adapter_bytes},
                node_id=nid,
            )

            round_complete = len(self._uploads[round_idx]) == self.num_shards
            if round_complete and round_idx not in self._merged:
                ordered = [self._uploads[round_idx][k] for k in range(self.num_shards)]
                merged_bytes = average_state_dict_blobs([self._store.get(h) for h in ordered])
                merged_hash = self._store.put(merged_bytes).artifact_id
                self._merged[round_idx] = merged_hash
                if round_idx + 1 < self.num_rounds:
                    self._round_init[round_idx + 1] = merged_hash
                elif round_idx + 1 == self.num_rounds:
                    self._record_outcome(merged_bytes, merged_hash)

        return {
            "session_id": self.session_id, "round": round_idx, "shard": shard_idx,
            "round_complete": round_complete, "rounds_merged": sorted(self._merged),
        }

    def _record_outcome(self, merged_bytes: bytes, merged_hash: str) -> None:
        if self._outcome_log is None or self._outcome_recorded:
            return
        try:
            base = self._make_base()
            model = make_lora_model(base, seed=self.seed)
            set_peft_model_state_dict(model, decode_state_dict(merged_bytes))
            loss_after = evaluate_loss(model, self._held_out_blocks)
            gain = float(self._loss_before - loss_after)   # loss reduction
            cost = float(self.num_rounds * self.num_shards * self.local_steps)
            from tron.orchestrator.outcomes import TrainingOutcome
            expected_gain = self._outcome_log.estimate_capability_gain(self._run_name, "distributed-lora")
            expected_cost = self._outcome_log.estimate_cost(self._run_name, "distributed-lora")
            self._outcome_log.record(TrainingOutcome(
                artifact_id=merged_hash,
                adapter_name=self._run_name,
                module_id="distributed-lora",
                expected_capability_gain=gain if expected_gain is None else expected_gain,
                actual_capability_gain=gain,
                expected_cost=cost if expected_cost is None else expected_cost,
                actual_cost=cost,
                success=gain > 0.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
                node_ids=sorted(set(self._shard_node_ids.values())),
            ))
            self._outcome_recorded = True
            self._loss_after = loss_after
        except Exception:
            pass

    # -- status / result ------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            return {
                "session_id": self.session_id,
                "model_kind": "lora",
                "model_name": self.model_name,
                "num_shards": self.num_shards,
                "num_rounds": self.num_rounds,
                "local_steps": self.local_steps,
                "lr": self.lr,
                "seed": self.seed,
                "adapter_bytes": self._adapter_bytes,
                "full_model_bytes": self._full_model_bytes,
                "size_ratio": self._full_model_bytes / max(1, self._adapter_bytes),
                "uploads_per_round": {r: sorted(v) for r, v in self._uploads.items() if v},
                "rounds_merged": sorted(self._merged),
                "finished": (self.num_rounds - 1) in self._merged,
                "wire_bytes_transferred": self._wire_bytes,
                "eval_loss_before": self._loss_before,
            }

    def final_adapter_bytes(self) -> Optional[bytes]:
        last = self.num_rounds - 1
        with self._lock:
            h = self._merged.get(last)
        return self._store.get(h) if h else None

    def result(self) -> Optional[dict]:
        blob = self.final_adapter_bytes()
        if blob is None:
            return None
        st = self.status()
        base = self._make_base()
        model = make_lora_model(base, seed=self.seed)
        set_peft_model_state_dict(model, decode_state_dict(blob))
        st["eval_loss_after"] = evaluate_loss(model, self._held_out_blocks)
        st["capability_gained"] = st["eval_loss_before"] - st["eval_loss_after"]
        st["final_adapter_hash"] = self._merged[self.num_rounds - 1]
        return st

    def _check_shard(self, shard_idx: int) -> None:
        if not 0 <= shard_idx < self.num_shards:
            raise ValueError(f"shard {shard_idx} out of range [0, {self.num_shards})")


def _load_token_ids(corpus_path):
    """Tokenized corpus as a 1-D LongTensor. Accepts, in order:
      - a 1-D torch.Tensor of token ids (used as-is)
      - a list/tuple of ints (e.g. from JSON) -> tensor
      - a path to a plain-text file -> tokenized with the Pythia tokenizer
      - None -> the vendored tiny-shakespeare corpus, Pythia-tokenized
    The last two need `transformers`; the first two don't."""
    if isinstance(corpus_path, torch.Tensor):
        return corpus_path.reshape(-1).long()
    if isinstance(corpus_path, (list, tuple)):
        return torch.tensor(list(corpus_path), dtype=torch.long)

    from transformers import AutoTokenizer

    path = corpus_path or DATA_PATH
    text = open(path, "r", encoding="utf-8").read()
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    return tok(text, return_tensors="pt")["input_ids"][0]
