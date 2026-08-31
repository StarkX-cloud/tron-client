"""Phase 3 over a real wire: shards as separate callers that serialize
their parameter vectors and POST them to the master over HTTP.

The property that has to hold, exactly like tests/test_spine_integration.py
for the in-process wiring: **the transport must not change the
computation.** Same seeds, same data, same local-SGD schedule, same
averaging op — so the final model from the wire path is bit-for-bit equal
to local_sgd.train_local_sgd run in one process. If it ever isn't, the
serialize/transmit/deserialize path is corrupting numbers and this test
fails loudly.
"""
import importlib
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tron.spine import EventLog, ArtifactStore, TaskStatus
from tron.training.data import make_classification_dataset, make_non_iid_shards, train_test_split
from tron.training.local_sgd import train_local_sgd
from tron.training.model import TinyMLP
from tron.training.distributed import TrainingSession
from tron.training.distributed.protocol import (
    average_vectors,
    decode_vector,
    encode_dataset,
    encode_vector,
    decode_dataset,
)
from tron.training.distributed.shard_client import run_shard
from tron.training.local_sgd import _average_flat_params

# Small, fast config shared by the reference run and the session.
NUM_SHARDS = 2
NUM_ROUNDS = 3
LOCAL_STEPS = 4
LR = 0.3
DATASET = dict(num_samples=400, num_features=4, num_classes=2, seed=0, class_sep=1.0, test_fraction=0.2)
HIDDEN_DIM = 6
SKEW = 0.9
SHARD_SEED = 1


def _reference_final_vector():
    x_all, y_all = make_classification_dataset(
        num_samples=DATASET["num_samples"], num_features=DATASET["num_features"],
        num_classes=DATASET["num_classes"], seed=DATASET["seed"], class_sep=DATASET["class_sep"],
    )
    x_train, y_train, _, _ = train_test_split(x_all, y_all, test_fraction=DATASET["test_fraction"])
    shards = make_non_iid_shards(
        x_train, y_train, num_shards=NUM_SHARDS,
        num_classes=DATASET["num_classes"], skew=SKEW, seed=SHARD_SEED,
    )

    def factory():
        return TinyMLP(
            input_dim=DATASET["num_features"], hidden_dim=HIDDEN_DIM,
            num_classes=DATASET["num_classes"], seed=42,
        )

    ref = train_local_sgd(shards, factory, num_rounds=NUM_ROUNDS, local_steps=LOCAL_STEPS, lr=LR)
    return ref["model"].get_flat_params()


# ---------------------------------------------------------------------------
# protocol.py
# ---------------------------------------------------------------------------

def test_vector_round_trip_is_exact():
    v = np.random.default_rng(0).normal(size=257).astype(np.float64)
    assert np.array_equal(decode_vector(encode_vector(v)), v)


def test_dataset_round_trip_is_exact():
    x = np.random.default_rng(1).normal(size=(20, 4))
    y = np.arange(20) % 3
    dx, dy = decode_dataset(encode_dataset(x, y))
    assert np.array_equal(dx, x) and np.array_equal(dy, y)


def test_average_vectors_matches_in_process_average():
    rng = np.random.default_rng(2)
    models = []
    for _ in range(3):
        m = TinyMLP(input_dim=4, hidden_dim=6, num_classes=2, seed=int(rng.integers(1, 1000)))
        models.append(m)
    blobs = [encode_vector(m.get_flat_params()) for m in models]
    from_wire = decode_vector(average_vectors(blobs))
    in_process = _average_flat_params(models)
    assert np.array_equal(from_wire, in_process)


# ---------------------------------------------------------------------------
# TrainingSession barrier + accounting (no HTTP)
# ---------------------------------------------------------------------------

@pytest.fixture
def spine(tmp_path):
    return EventLog(path=tmp_path / "log.db"), ArtifactStore(root=tmp_path / "art")


def _make_session(spine, **overrides):
    log, store = spine
    kwargs = dict(
        num_shards=NUM_SHARDS, num_rounds=NUM_ROUNDS, local_steps=LOCAL_STEPS, lr=LR,
        event_log=log, artifact_store=store, model_config={"hidden_dim": HIDDEN_DIM},
        dataset_config=dict(DATASET), skew=SKEW, shard_seed=SHARD_SEED,
    )
    kwargs.update(overrides)
    return TrainingSession("sess-unit", **kwargs)


def test_round_init_is_202_until_the_previous_barrier_clears(spine):
    s = _make_session(spine)
    assert s.round_init(0, 0) is not None          # round 0 is always ready
    assert s.round_init(1, 0) is None              # round 1 waits on round 0's merge

    # shard 0 reports round 0, shard 1 hasn't yet -> still not ready
    s.submit_round(0, 0, encode_vector(decode_vector(s.round_init(0, 0))))
    assert s.round_init(1, 0) is None

    s.submit_round(0, 1, encode_vector(decode_vector(s.round_init(0, 1))))
    assert s.round_init(1, 0) is not None          # both in -> merged -> round 1 open


def test_submit_rejects_wrong_length_vector(spine):
    s = _make_session(spine)
    with pytest.raises(ValueError):
        s.submit_round(0, 0, encode_vector(np.zeros(3)))


def test_wire_byte_accounting_counts_upload_and_download_legs(spine):
    s = _make_session(spine)
    for r in range(NUM_ROUNDS):
        for k in range(NUM_SHARDS):
            blob = s.round_init(r, k)
            s.submit_round(r, k, blob)
    st = s.status()
    # per shard per round: 1 download (init) + 1 upload (trained) = 2 legs
    expected = NUM_ROUNDS * NUM_SHARDS * 2 * st["num_params"] * 8
    assert st["wire_bytes_transferred"] == expected
    # the in-process metric only counts the conceptual upload side
    assert st["algorithmic_comm_bytes"] == NUM_ROUNDS * NUM_SHARDS * st["num_params"] * 8


# ---------------------------------------------------------------------------
# Full stack: run_shard against the real queue_server endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRON_SPINE_DIR", str(tmp_path / "spine"))
    import queue_server
    importlib.reload(queue_server)
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


def _run_all_shards(tc, session_id, n_shards=NUM_SHARDS):
    """Every shard is a barrier participant, so they have to run
    concurrently — a shard can't finish round 1 until every other shard
    has posted round 0. One thread per shard, all driving the real
    endpoints through run_shard."""
    transport = _TestClientTransport(tc)
    errors = []

    def drive(k):
        try:
            run_shard(transport, session_id, k, poll_interval=0.02, max_wait_seconds=60)
        except Exception as exc:  # noqa: BLE001 - surfaced via `errors`
            errors.append((k, repr(exc)))

    threads = [threading.Thread(target=drive, args=(k,)) for k in range(n_shards)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)
    assert not errors, errors
    assert all(not t.is_alive() for t in threads), "a shard thread did not finish"


def test_wire_path_final_model_matches_single_process_bit_for_bit(client):
    tc, qs = client
    resp = tc.post("/training/session", json={
        "session_id": "sess-parity",
        "num_shards": NUM_SHARDS, "num_rounds": NUM_ROUNDS, "local_steps": LOCAL_STEPS, "lr": LR,
        "model_config": {"hidden_dim": HIDDEN_DIM},
        "dataset_config": dict(DATASET),
        "skew": SKEW, "shard_seed": SHARD_SEED,
    })
    assert resp.status_code == 200

    _run_all_shards(tc, "sess-parity")

    result = tc.get("/training/session/sess-parity/result")
    assert result.status_code == 200
    assert result.json()["finished"] is True

    final_vec = decode_vector(qs.training_sessions.get("sess-parity").final_vector_bytes())
    np.testing.assert_array_equal(final_vec, _reference_final_vector())


def test_wire_run_is_recorded_in_the_spine_as_completed_tasks(client):
    tc, qs = client
    tc.post("/training/session", json={
        "session_id": "sess-spine",
        "num_shards": NUM_SHARDS, "num_rounds": NUM_ROUNDS, "local_steps": LOCAL_STEPS, "lr": LR,
        "model_config": {"hidden_dim": HIDDEN_DIM}, "dataset_config": dict(DATASET),
        "skew": SKEW, "shard_seed": SHARD_SEED,
    })

    _run_all_shards(tc, "sess-spine")

    tasks = [t for t in qs.event_log.snapshot().values()
             if t.metadata.get("session") == "sess-spine"]
    assert len(tasks) == NUM_ROUNDS * NUM_SHARDS
    assert all(t.status == TaskStatus.COMPLETED for t in tasks)
    # every completed task carries its trained vector as a retrievable artifact
    for t in tasks:
        raw = qs.artifact_store.get(t.output_hash)
        assert decode_vector(raw).shape == (qs.training_sessions.get("sess-spine").status()["num_params"],)


def test_shard_index_out_of_range_is_rejected(client):
    tc, _ = client
    tc.post("/training/session", json={"session_id": "sess-oob", "num_shards": 2, "num_rounds": 2})
    r = tc.get("/training/session/sess-oob/shard/5/data")
    assert r.status_code == 400


def test_finished_run_is_recorded_as_a_training_outcome(client):
    tc, qs = client
    if qs.training_outcomes is None:
        pytest.skip("tron.orchestrator.outcomes unavailable")

    tc.post("/training/session", json={
        "session_id": "sess-outcome",
        "num_shards": NUM_SHARDS, "num_rounds": NUM_ROUNDS, "local_steps": LOCAL_STEPS, "lr": LR,
        "model_config": {"hidden_dim": HIDDEN_DIM}, "dataset_config": dict(DATASET),
        "skew": SKEW, "shard_seed": SHARD_SEED,
    })
    _run_all_shards(tc, "sess-outcome")
    tc.get("/training/session/sess-outcome/result")  # trigger nothing extra; outcome recorded at final merge

    body = tc.get("/training/outcomes").json()
    assert body["count"] == 1
    o = body["outcomes"][0]
    assert o["adapter_name"] == "sess-outcome"
    assert o["module_id"] == "distributed-numpy_mlp"
    # compute spent = total local SGD steps across all shards
    assert o["actual_cost"] == NUM_ROUNDS * NUM_SHARDS * LOCAL_STEPS
    # training a classifier from an untrained init should gain capability
    assert o["actual_capability_gain"] > 0.0
    assert o["success"] is True
    assert o["capability_per_compute"] == pytest.approx(o["actual_capability_gain"] / o["actual_cost"])


def test_outcome_is_recorded_once_not_per_result_poll(client):
    tc, qs = client
    if qs.training_outcomes is None:
        pytest.skip("tron.orchestrator.outcomes unavailable")
    tc.post("/training/session", json={
        "session_id": "sess-once", "num_shards": NUM_SHARDS, "num_rounds": NUM_ROUNDS,
        "local_steps": LOCAL_STEPS, "lr": LR, "model_config": {"hidden_dim": HIDDEN_DIM},
        "dataset_config": dict(DATASET), "skew": SKEW, "shard_seed": SHARD_SEED,
    })
    _run_all_shards(tc, "sess-once")
    for _ in range(3):
        tc.get("/training/session/sess-once/result")
    assert tc.get("/training/outcomes").json()["count"] == 1
