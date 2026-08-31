"""Fast tests for tron/training/lora_spine.py — LoRA local-SGD wired into
the execution spine, the adapter counterpart to
tests/test_spine_integration.py's numpy-MLP coverage.

Like test_lora_demo.py, these use a tiny synthetic module with a
`query_key_value` linear (the projection `lora_demo.make_lora_model`
targets) instead of downloading Pythia-70M, so they run in seconds.

The property that matters, same as the numpy wiring: recording must not
change the computation. The instrumented run's final adapter must match
`lora_demo.run_local_sgd_lora`'s tensor-for-tensor.
"""
import types

import pytest
import torch
import torch.nn as nn

from tron.spine import ArtifactStore, EventLog, TaskStatus
from tron.training.lora_demo import run_local_sgd_lora
from tron.training.lora_spine import (
    decode_adapter_state,
    encode_adapter_state,
    run_local_sgd_lora_with_spine,
)
from peft import get_peft_model_state_dict

SEQ_LEN = 8
VOCAB = 64


class _TinyCausalLM(nn.Module):
    """Minimal GPT-NeoX-shaped stand-in: an embedding, a
    `query_key_value` projection (what make_lora_model adapts), and an
    output head. The qkv output feeds the logits so the LoRA adapter is
    actually in the gradient path."""

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(model_type="tiny-causal-lm")
        self.embed = nn.Embedding(VOCAB, 16)
        self.query_key_value = nn.Linear(16, 48)
        self.out = nn.Linear(16, VOCAB)

    def forward(self, input_ids=None, labels=None, **kwargs):
        h = self.embed(input_ids)
        v = self.query_key_value(h)[..., 32:]  # "value" third
        logits = self.out(v)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1))
        return types.SimpleNamespace(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}


def _base_model():
    torch.manual_seed(0)
    model = _TinyCausalLM()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _shards(n=2):
    return [
        torch.randint(0, VOCAB, (6, SEQ_LEN), generator=torch.Generator().manual_seed(100 + i))
        for i in range(n)
    ]


def _held_out():
    return torch.randint(0, VOCAB, (4, SEQ_LEN), generator=torch.Generator().manual_seed(999))


@pytest.fixture
def spine(tmp_path):
    return EventLog(path=tmp_path / "log.db"), ArtifactStore(root=tmp_path / "art")


# ---------------------------------------------------------------------------
# adapter (de)serialization
# ---------------------------------------------------------------------------

def test_adapter_state_round_trips_exactly():
    state = {"a": torch.randn(4, 8), "b": torch.randn(8)}
    back = decode_adapter_state(encode_adapter_state(state))
    assert set(back) == set(state)
    for k in state:
        assert torch.equal(back[k], state[k])


# ---------------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------------

def test_instrumented_lora_run_matches_uninstrumented_tensor_for_tensor(spine):
    log, store = spine
    kwargs = dict(num_rounds=2, local_steps=3, lr=1e-3, seed=5)

    ref = run_local_sgd_lora(_base_model(), _shards(), _held_out(), **kwargs)
    got = run_local_sgd_lora_with_spine(
        _base_model(), _shards(), _held_out(), event_log=log, artifact_store=store, **kwargs
    )

    ref_state = get_peft_model_state_dict(ref.final_model)
    got_state = get_peft_model_state_dict(got["final_model"])
    assert set(ref_state) == set(got_state)
    for k in ref_state:
        assert torch.equal(ref_state[k], got_state[k]), k

    assert got["comm_bytes"] == ref.comm_bytes
    assert got["num_syncs"] == ref.num_syncs
    assert got["eval_loss_after"] == pytest.approx(ref.eval_loss_after, rel=1e-6)


# ---------------------------------------------------------------------------
# spine recording
# ---------------------------------------------------------------------------

def test_one_completed_task_per_shard_per_round(spine):
    log, store = spine
    run_local_sgd_lora_with_spine(
        _base_model(), _shards(n=3), _held_out(),
        num_rounds=2, local_steps=2, lr=1e-3, event_log=log, artifact_store=store, seed=1,
    )
    tasks = log.snapshot()
    assert len(tasks) == 2 * 3
    assert all(t.status == TaskStatus.COMPLETED for t in tasks.values())


def test_completed_events_carry_adapter_bytes_as_transfer_bytes(spine):
    log, store = spine
    result = run_local_sgd_lora_with_spine(
        _base_model(), _shards(), _held_out(),
        num_rounds=1, local_steps=2, lr=1e-3, event_log=log, artifact_store=store, seed=1,
    )
    completed = [e for e in log.replay() if e.type == "completed"]
    assert completed
    for e in completed:
        assert e.data["transfer_bytes"] == result["adapter_bytes"]
        # the adapter, not the full model, is the unit that moved
        assert e.data["transfer_bytes"] < result["full_model_bytes"]


def test_output_artifacts_are_retrievable_adapter_state_dicts(spine):
    log, store = spine
    run_local_sgd_lora_with_spine(
        _base_model(), _shards(), _held_out(),
        num_rounds=1, local_steps=2, lr=1e-3, event_log=log, artifact_store=store, seed=1,
    )
    for task in log.snapshot().values():
        raw = store.get(task.output_hash)
        state = decode_adapter_state(raw)
        assert all(isinstance(v, torch.Tensor) for v in state.values())
        assert any("lora_A" in k for k in state)


def test_records_a_training_outcome_when_given_an_outcome_log(spine):
    from tron.orchestrator.outcomes import OutcomeLog

    log, store = spine
    olog = OutcomeLog()
    result = run_local_sgd_lora_with_spine(
        _base_model(), _shards(), _held_out(),
        num_rounds=2, local_steps=3, lr=1e-3, event_log=log, artifact_store=store, seed=5,
        outcome_log=olog, run_name="unit-lora",
    )
    assert len(olog.outcomes) == 1
    o = olog.outcomes[0]
    assert o.adapter_name == "unit-lora"
    assert o.module_id == "lora-local-sgd"
    assert o.actual_cost == 2 * 2 * 3  # rounds * shards * local_steps
    # gain is the eval-loss reduction
    assert o.actual_capability_gain == pytest.approx(result["eval_loss_before"] - result["eval_loss_after"])


def test_no_outcome_log_means_no_recording(spine):
    log, store = spine
    # just asserting it doesn't raise without an outcome_log
    run_local_sgd_lora_with_spine(
        _base_model(), _shards(), _held_out(),
        num_rounds=1, local_steps=2, lr=1e-3, event_log=log, artifact_store=store, seed=5,
    )


def test_task_metadata_records_round_shard_and_model_sizes(spine):
    log, store = spine
    run_local_sgd_lora_with_spine(
        _base_model(), _shards(), _held_out(),
        num_rounds=2, local_steps=1, lr=1e-3, event_log=log, artifact_store=store, seed=1,
    )
    tasks = log.snapshot()
    # 2 rounds x 2 shards, distinct (round, shard) metadata pairs
    seen = {(t.metadata["round"], t.metadata["shard"]) for t in tasks.values()}
    assert seen == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for t in tasks.values():
        assert t.metadata["kind"] == "lora-local-sgd"
        assert t.metadata["adapter_bytes"] < t.metadata["full_model_bytes"]
