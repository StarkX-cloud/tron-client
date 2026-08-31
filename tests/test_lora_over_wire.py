"""LoRA adapters over the tron/training/distributed/ transport.

Same standard as tests/test_distributed_training.py's numpy parity test:
the wire path (shards POSTing adapter state dicts to a master that
barrier-averages) must produce a final adapter **tensor-for-tensor
identical** to lora_demo.run_local_sgd_lora run in one process — same
seeds, same schedule, same averaging op, a long-lived optimizer across
rounds. The only difference is that a few hundred KB of adapter crosses a
socket each round instead of the model.

Uses a tiny GPT-NeoX-shaped stand-in (a `query_key_value` linear feeding
the logits) so it runs in seconds with no Pythia download.
"""
import importlib
import threading
import types

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from peft import get_peft_model_state_dict

from tron.spine import EventLog, ArtifactStore, TaskStatus
from tron.training.lora_demo import make_shards, run_local_sgd_lora
from tron.training.distributed.lora_param_server import LoraTrainingSession
from tron.training.distributed.lora_wire import decode_state_dict, run_lora_shard

VOCAB = 64
SEQ_LEN = 8
NUM_SHARDS = 2
NUM_ROUNDS = 3
LOCAL_STEPS = 3
LR = 1e-3
SEED = 5
HELD_OUT_FRACTION = 0.1


class _TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(model_type="tiny-causal-lm")
        self.embed = nn.Embedding(VOCAB, 16)
        self.query_key_value = nn.Linear(16, 48)
        self.out = nn.Linear(16, VOCAB)

    def forward(self, input_ids=None, labels=None, **kwargs):
        h = self.embed(input_ids)
        v = self.query_key_value(h)[..., 32:]
        logits = self.out(v)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1))
        return types.SimpleNamespace(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}


def _base_factory():
    torch.manual_seed(0)
    m = _TinyCausalLM()
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def _corpus():
    # deterministic 1-D token stream, long enough to shard
    g = torch.Generator().manual_seed(1234)
    return torch.randint(0, VOCAB, (640,), generator=g)


def _reference_final_state():
    ids = _corpus()
    split_at = int(len(ids) * (1.0 - HELD_OUT_FRACTION))
    train_ids, held_ids = ids[:split_at], ids[split_at:]
    shards = make_shards(train_ids, num_shards=NUM_SHARDS, seq_len=SEQ_LEN)
    held = make_shards(held_ids, num_shards=1, seq_len=SEQ_LEN)[0]
    ref = run_local_sgd_lora(
        _base_factory(), shards, held,
        num_rounds=NUM_ROUNDS, local_steps=LOCAL_STEPS, lr=LR, seed=SEED,
    )
    return get_peft_model_state_dict(ref.final_model), ref


@pytest.fixture
def spine(tmp_path):
    return EventLog(path=tmp_path / "log.db"), ArtifactStore(root=tmp_path / "art")


def _make_session(spine, **overrides):
    log, store = spine
    kwargs = dict(
        base_model_factory=_base_factory,
        num_shards=NUM_SHARDS, num_rounds=NUM_ROUNDS, local_steps=LOCAL_STEPS, lr=LR,
        event_log=log, artifact_store=store, seq_len=SEQ_LEN, seed=SEED,
        corpus_path=_corpus(), held_out_fraction=HELD_OUT_FRACTION,
    )
    kwargs.update(overrides)
    return LoraTrainingSession("lora-wire-unit", **kwargs)


# ---------------------------------------------------------------------------
# barrier + accounting, no HTTP
# ---------------------------------------------------------------------------

def test_round_init_is_none_until_previous_barrier_clears(spine):
    s = _make_session(spine)
    assert s.round_init(0, 0) is not None
    assert s.round_init(1, 0) is None
    s.submit_round(0, 0, s.round_init(0, 0))
    assert s.round_init(1, 0) is None
    s.submit_round(0, 1, s.round_init(0, 1))
    assert s.round_init(1, 0) is not None


def test_status_reports_adapter_much_smaller_than_full_model(spine):
    st = _make_session(spine).status()
    assert 0 < st["adapter_bytes"] < st["full_model_bytes"]
    assert st["size_ratio"] > 1.0
    assert st["model_kind"] == "lora"


# ---------------------------------------------------------------------------
# full stack: run_lora_shard against the real queue_server endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRON_SPINE_DIR", str(tmp_path / "spine"))
    import queue_server
    importlib.reload(queue_server)
    # inject the tiny base model so the endpoint doesn't try to download Pythia
    queue_server._lora_base_cache["tiny-test-model"] = _base_factory()
    return TestClient(queue_server.app), queue_server


class _TestClientTransport:
    def __init__(self, tc):
        self._tc = tc

    def get_bytes(self, path):
        r = self._tc.get(path)
        if r.status_code == 202:
            return None
        r.raise_for_status()
        return r.content

    def get_json(self, path):
        r = self._tc.get(path)
        r.raise_for_status()
        return r.json()

    def post_bytes(self, path, blob):
        r = self._tc.post(path, content=blob)
        r.raise_for_status()
        return r.json()


def _run_all_lora_shards(tc):
    transport = _TestClientTransport(tc)
    errors = []

    def drive(k):
        try:
            run_lora_shard(transport, "lora-wire", k, base_model_factory=_base_factory,
                           poll_interval=0.02, max_wait_seconds=90)
        except Exception as exc:  # noqa: BLE001
            errors.append((k, repr(exc)))

    threads = [threading.Thread(target=drive, args=(k,)) for k in range(NUM_SHARDS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, errors
    assert all(not t.is_alive() for t in threads)


def _create_wire_session(tc):
    # feed the corpus in by monkeypatching the loader for this session:
    # simplest is to pre-tokenize is not exposed over HTTP, so we rely on
    # the endpoint default corpus being deterministic. Instead, build the
    # session object directly and register it, then drive its endpoints.
    import queue_server as qs
    log, store = qs.event_log, qs.artifact_store
    session = LoraTrainingSession(
        "lora-wire",
        base_model_factory=_base_factory,
        num_shards=NUM_SHARDS, num_rounds=NUM_ROUNDS, local_steps=LOCAL_STEPS, lr=LR,
        event_log=log, artifact_store=store, seq_len=SEQ_LEN, seed=SEED,
        corpus_path=_corpus(), held_out_fraction=HELD_OUT_FRACTION,
        outcome_log=qs.training_outcomes,
    )
    qs.training_sessions.register("lora-wire", session)
    return session


def test_wire_lora_final_adapter_matches_single_process_tensor_for_tensor(client):
    tc, qs = client
    session = _create_wire_session(tc)

    _run_all_lora_shards(tc)

    assert tc.get("/training/session/lora-wire/result").json()["finished"] is True

    got = decode_state_dict(session.final_adapter_bytes())
    ref_state, _ref = _reference_final_state()
    assert set(got) == set(ref_state)
    for k in ref_state:
        assert torch.equal(got[k], ref_state[k]), k


def test_wire_lora_run_recorded_in_spine_and_outcomes(client):
    tc, qs = client
    _create_wire_session(tc)
    _run_all_lora_shards(tc)

    tasks = [t for t in qs.event_log.snapshot().values()
             if t.metadata.get("session") == "lora-wire"]
    assert len(tasks) == NUM_ROUNDS * NUM_SHARDS
    assert all(t.status == TaskStatus.COMPLETED for t in tasks)
    for t in tasks:
        assert t.metadata["kind"] == "distributed-lora-local-sgd"
        assert t.metadata["adapter_bytes"] < t.metadata["full_model_bytes"]

    if qs.training_outcomes is not None:
        outs = [o for o in qs.training_outcomes.outcomes if o.module_id == "distributed-lora"]
        assert len(outs) == 1
        assert outs[0].actual_cost == NUM_ROUNDS * NUM_SHARDS * LOCAL_STEPS


def test_create_session_dispatches_on_model_kind_lora(client):
    tc, qs = client
    r = tc.post("/training/session", json={
        "session_id": "lora-endpoint",
        "model_kind": "lora",
        "num_shards": 2, "num_rounds": 2, "local_steps": 2, "lr": LR, "seed": SEED,
        "model_config": {
            "model_name": "tiny-test-model", "seq_len": SEQ_LEN,
            "token_ids": _corpus().tolist(),   # skip the tokenizer download
        },
    })
    assert r.status_code == 200
    body = r.json()
    assert body["model_kind"] == "lora"
    assert body["adapter_bytes"] < body["full_model_bytes"]
    assert qs.training_sessions.get("lora-endpoint").model_kind == "lora"
