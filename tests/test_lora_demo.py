"""Fast, model-download-free tests for tron/training/lora_demo.py's pure
logic (shard splitting, byte accounting, adapter averaging).

What these deliberately do NOT cover: actually loading Pythia-70M and
running the local-SGD / merge training loops. That path was verified
manually via tron/training/benchmark_lora.py — real numbers, real
model, real corpus — but a full run takes ~20 minutes on this project's
CPU-only dev machine, which makes it impractical as a live pytest (it
would make the whole suite that slow on every run). See ROADMAP.md for
the honest tradeoff and the recorded benchmark numbers.

These tests use a tiny synthetic nn.Module instead of downloading any
pretrained weights, so they run in seconds and need no network access.
"""
import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from tron.training.lora_demo import (
    adapter_state_bytes,
    average_adapter_states,
    full_model_bytes,
    make_shards,
)


class _TinyModel(nn.Module):
    """A minimal module with a named Linear layer LoRA can target —
    stands in for a real transformer's attention projection without
    downloading one."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, x):
        return self.proj(x)


def _tiny_lora_model(seed: int = 0):
    torch.manual_seed(seed)
    base = _TinyModel()
    config = LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], lora_dropout=0.0, bias="none")
    return get_peft_model(base, config)


# ---------------------------------------------------------------------------
# make_shards
# ---------------------------------------------------------------------------

def test_make_shards_splits_into_correct_block_shape():
    token_ids = torch.arange(300)
    shards = make_shards(token_ids, num_shards=3, seq_len=10)
    assert len(shards) == 3
    for shard in shards:
        assert shard.shape[1] == 10


def test_make_shards_covers_disjoint_ranges():
    token_ids = torch.arange(60)
    shards = make_shards(token_ids, num_shards=3, seq_len=10)
    # Each shard's flattened tokens should come from a distinct 1/3 of
    # the corpus (no overlap), matching the "contiguous chunks" contract.
    flattened = [shard.flatten().tolist() for shard in shards]
    all_values = [v for shard_vals in flattened for v in shard_vals]
    assert len(all_values) == len(set(all_values))  # no duplicates


def test_make_shards_rejects_shard_smaller_than_seq_len():
    token_ids = torch.arange(10)
    with pytest.raises(ValueError):
        make_shards(token_ids, num_shards=5, seq_len=10)  # 2 tokens/shard < seq_len


# ---------------------------------------------------------------------------
# adapter_state_bytes / full_model_bytes
# ---------------------------------------------------------------------------

def test_adapter_bytes_much_smaller_than_full_model_bytes():
    model = _tiny_lora_model()
    adapter_bytes = adapter_state_bytes(model)
    full_bytes = full_model_bytes(model)
    assert 0 < adapter_bytes < full_bytes


def test_adapter_bytes_matches_hand_computed_lora_param_count():
    # r=2 LoRA on an 8x8 Linear: lora_A is (r, in_features) = (2, 8),
    # lora_B is (out_features, r) = (8, 2) -> 16 + 16 = 32 params, 4 bytes
    # each (float32) = 128 bytes.
    model = _tiny_lora_model()
    assert adapter_state_bytes(model) == 32 * 4


# ---------------------------------------------------------------------------
# average_adapter_states
# ---------------------------------------------------------------------------

def test_average_adapter_states_computes_elementwise_mean():
    states = [
        {"w": torch.tensor([1.0, 2.0])},
        {"w": torch.tensor([3.0, 4.0])},
        {"w": torch.tensor([5.0, 6.0])},
    ]
    averaged = average_adapter_states(states)
    torch.testing.assert_close(averaged["w"], torch.tensor([3.0, 4.0]))


def test_average_adapter_states_identity_when_all_equal():
    tensor = torch.tensor([1.0, -2.0, 3.5])
    states = [{"w": tensor.clone()} for _ in range(4)]
    averaged = average_adapter_states(states)
    torch.testing.assert_close(averaged["w"], tensor)


def test_average_adapter_states_preserves_all_keys():
    states = [
        {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])},
        {"a": torch.tensor([3.0]), "b": torch.tensor([4.0])},
    ]
    averaged = average_adapter_states(states)
    assert set(averaged.keys()) == {"a", "b"}
