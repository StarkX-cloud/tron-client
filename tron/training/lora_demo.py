"""Phase 3 scale-up: the same local-SGD + weight-merging story as
local_sgd.py / merge.py, applied to a real small open-weight model
(EleutherAI's Pythia-70M, via LoRA) instead of a hand-rolled numpy MLP.

Why this exists: the numpy demo proves the training *algorithms* are
implemented correctly at a scale anyone can verify in seconds. This
module proves the same algorithms apply to a real pretrained transformer
the ML community actually recognizes — and because LoRA trains a
low-rank adapter instead of the full model, it demonstrates the other
half of this project's original thesis: the unit of communication
becomes a small adapter (tens of KB), not a full model checkpoint
(hundreds of MB). See benchmark_lora() for the numbers.

Requires torch, transformers, peft — intentionally NOT in the base
requirements.txt. This is the optional, heavier scale-up path, not
something queue_server.py needs to boot. See requirements-training.txt.

A real bug hit during development, not a defensive guess: the
`hf_xet` fast-transfer backend hung indefinitely downloading model
weights on this project's dev machine. HF_HUB_DISABLE_XET=1 below
works around it — see ROADMAP.md.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/pythia-70m"
DATA_PATH = Path(__file__).parent / "data" / "tinyshakespeare.txt"
SEQ_LEN = 64
BYTES_PER_FLOAT32 = 4


def load_base_model_and_tokenizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def make_lora_model(base_model, seed: int = 0):
    """Wrap a fresh copy of base_model in an independently-initialized
    LoRA adapter. Each call returns its own model so multiple "shards"
    never share adapter weights by accident."""
    torch.manual_seed(seed)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query_key_value"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(copy.deepcopy(base_model), config)


def load_corpus_token_ids(tokenizer, path: Path = DATA_PATH) -> torch.Tensor:
    text = path.read_text(encoding="utf-8")
    return tokenizer(text, return_tensors="pt")["input_ids"][0]


def make_shards(token_ids: torch.Tensor, num_shards: int, seq_len: int = SEQ_LEN) -> list[torch.Tensor]:
    """Split the tokenized corpus into `num_shards` contiguous chunks —
    a natural, non-artificial non-IID split (different chunks of
    Shakespeare are literally different scenes), not a synthetic skew
    parameter like the numpy demo's — then each chunk into fixed-length
    blocks for causal LM training. Returns one (num_blocks, seq_len)
    tensor per shard."""
    n = len(token_ids)
    shard_size = n // num_shards
    shards = []
    for i in range(num_shards):
        chunk = token_ids[i * shard_size:(i + 1) * shard_size]
        num_blocks = len(chunk) // seq_len
        if num_blocks == 0:
            raise ValueError(f"shard {i} has fewer than {seq_len} tokens; use fewer shards or a smaller seq_len")
        blocks = chunk[: num_blocks * seq_len].view(num_blocks, seq_len)
        shards.append(blocks)
    return shards


def adapter_state_bytes(model) -> int:
    state = get_peft_model_state_dict(model)
    return sum(t.numel() * BYTES_PER_FLOAT32 for t in state.values())


def full_model_bytes(model) -> int:
    return sum(p.numel() * BYTES_PER_FLOAT32 for p in model.parameters())


def average_adapter_states(states: list[dict]) -> dict:
    averaged = {}
    for key in states[0]:
        stacked = torch.stack([s[key].float() for s in states], dim=0)
        averaged[key] = stacked.mean(dim=0)
    return averaged


@torch.no_grad()
def evaluate_loss(model, blocks: torch.Tensor, max_batches: int = 8) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for i in range(min(max_batches, len(blocks))):
        batch = blocks[i:i + 1]
        out = model(input_ids=batch, labels=batch)
        losses.append(out.loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def train_steps(model, blocks: torch.Tensor, num_steps: int, lr: float, optimizer=None, step_offset: int = 0):
    if optimizer is None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr)
    model.train()
    n = len(blocks)
    for step in range(num_steps):
        idx = (step_offset + step) % n
        batch = blocks[idx:idx + 1]
        optimizer.zero_grad()
        out = model(input_ids=batch, labels=batch)
        out.loss.backward()
        optimizer.step()
    return optimizer


@dataclass
class LoraRunResult:
    eval_loss_before: float
    eval_loss_after: float
    comm_bytes: int
    num_syncs: int
    wall_clock_seconds: float
    final_model: object = field(repr=False)


def run_local_sgd_lora(
    base_model,
    shards: list[torch.Tensor],
    held_out_blocks: torch.Tensor,
    num_rounds: int,
    local_steps: int,
    lr: float,
    seed: int = 0,
) -> LoraRunResult:
    """DiLoCo-style local SGD, applied to LoRA adapters: each shard
    trains its own adapter for `local_steps` steps with zero
    communication, then every replica's adapter is averaged — the same
    shape as local_sgd.train_local_sgd, just with a real transformer's
    LoRA state dict standing in for the numpy demo's flat parameter
    vector.
    """
    torch.manual_seed(seed)
    models = [make_lora_model(base_model, seed=seed) for _ in shards]

    shared_state = get_peft_model_state_dict(models[0])
    for m in models[1:]:
        set_peft_model_state_dict(m, shared_state)

    adapter_bytes = adapter_state_bytes(models[0])
    eval_before = evaluate_loss(models[0], held_out_blocks)

    optimizers = [None] * len(models)
    comm_bytes = 0
    start = time.time()

    for round_idx in range(num_rounds):
        for i, (m, blocks) in enumerate(zip(models, shards)):
            optimizers[i] = train_steps(m, blocks, local_steps, lr, optimizers[i], step_offset=round_idx * local_steps)

        states = [get_peft_model_state_dict(m) for m in models]
        averaged = average_adapter_states(states)
        for m in models:
            set_peft_model_state_dict(m, averaged)
        comm_bytes += len(shards) * adapter_bytes

    eval_after = evaluate_loss(models[0], held_out_blocks)
    return LoraRunResult(eval_before, eval_after, comm_bytes, num_rounds, time.time() - start, models[0])


@dataclass
class MergeRunResult:
    eval_loss_before: float
    solo_eval_losses: list
    merged_eval_loss: float
    wall_clock_seconds: float


def run_independent_lora_and_merge(
    base_model,
    shards: list[torch.Tensor],
    held_out_blocks: torch.Tensor,
    num_steps: int,
    lr: float,
    seed: int = 0,
) -> MergeRunResult:
    """Train one LoRA adapter per shard fully independently — zero
    communication at all until this returns — then merge via simple
    task-arithmetic-style averaging of the adapter state dicts."""
    torch.manual_seed(seed)
    models = [make_lora_model(base_model, seed=seed) for _ in shards]
    shared_state = get_peft_model_state_dict(models[0])
    for m in models[1:]:
        set_peft_model_state_dict(m, shared_state)

    eval_before = evaluate_loss(models[0], held_out_blocks)

    start = time.time()
    for m, blocks in zip(models, shards):
        train_steps(m, blocks, num_steps, lr)

    solo_losses = [evaluate_loss(m, held_out_blocks) for m in models]

    merged_state = average_adapter_states([get_peft_model_state_dict(m) for m in models])
    merged_model = make_lora_model(base_model, seed=seed)
    set_peft_model_state_dict(merged_model, merged_state)
    merged_loss = evaluate_loss(merged_model, held_out_blocks)

    return MergeRunResult(eval_before, solo_losses, merged_loss, time.time() - start)
