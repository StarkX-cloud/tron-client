"""The concrete telecom demo: N cell towers with distinct (synthetic,
illustrative — see tron/training/telecom_data.py) traffic patterns,
training a shared next-interval congestion-tier predictor over the same
real wire transport and spine machinery every other distributed run in
this repo uses. These tests pin the dataset itself (deterministic,
genuinely non-IID by construction, learnable) and the full wire path
end to end through queue_server.py.
"""
import importlib
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tron.training.telecom_data import (
    NUM_FEATURES,
    NUM_TIERS,
    TOWER_PROFILES,
    build_tower_problem,
    tower_profile_for_shard,
)


# ---------------------------------------------------------------------------
# The dataset itself
# ---------------------------------------------------------------------------

def test_build_tower_problem_shape():
    shards, x_test, y_test, num_features, num_classes = build_tower_problem(
        num_shards=4, dataset_config={"seed": 0, "samples_per_tower": 300}
    )
    assert num_features == NUM_FEATURES
    assert num_classes == NUM_TIERS
    assert len(shards) == 4
    for x, y in shards:
        assert x.shape == (300, NUM_FEATURES)
        assert y.shape == (300,)
        assert set(np.unique(y)).issubset(set(range(NUM_TIERS)))
    assert x_test.shape[1] == NUM_FEATURES
    assert len(x_test) == len(y_test)


def test_build_tower_problem_is_deterministic():
    a = build_tower_problem(num_shards=3, dataset_config={"seed": 7, "samples_per_tower": 100})
    b = build_tower_problem(num_shards=3, dataset_config={"seed": 7, "samples_per_tower": 100})
    for (xa, ya), (xb, yb) in zip(a[0], b[0]):
        assert np.array_equal(xa, xb)
        assert np.array_equal(ya, yb)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


def test_towers_are_genuinely_non_iid():
    """More towers than profiles: cycles, and towers on different
    profiles must have visibly different label distributions — the
    honest hard case, not shuffled to look easier."""
    shards, _, _, _, _ = build_tower_problem(
        num_shards=len(TOWER_PROFILES), dataset_config={"seed": 1, "samples_per_tower": 500}
    )
    label_means = [y.mean() for _, y in shards]
    # not every tower centers on the same average congestion tier
    assert max(label_means) - min(label_means) > 0.15


def test_tower_profile_for_shard_cycles():
    n = len(TOWER_PROFILES)
    assert tower_profile_for_shard(0) == TOWER_PROFILES[0].name
    assert tower_profile_for_shard(n) == TOWER_PROFILES[0].name  # wraps around


def test_held_out_set_covers_every_profile_not_just_the_towers_used():
    _, _, y_test, _, _ = build_tower_problem(
        num_shards=1, dataset_config={"seed": 0, "samples_per_tower": 50, "test_samples_per_profile": 40}
    )
    assert len(y_test) == 40 * len(TOWER_PROFILES)


# ---------------------------------------------------------------------------
# Full stack: the real wire path, through queue_server.py
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRON_SPINE_DIR", str(tmp_path / "spine"))
    monkeypatch.delenv("TRON_TRAINING_AUTH_TOKEN", raising=False)
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


def test_telecom_session_uses_the_tower_dataset_not_the_default_one(client):
    tc, qs = client
    resp = tc.post("/training/session", json={
        "session_id": "sess-towers-shape",
        "problem": "telecom_congestion",
        "num_shards": 4, "num_rounds": 2, "local_steps": 3,
        "dataset_config": {"seed": 0, "samples_per_tower": 100},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_config"]["input_dim"] == NUM_FEATURES
    assert body["model_config"]["num_classes"] == NUM_TIERS


def test_telecom_run_beats_the_naive_majority_baseline(client):
    """The actual claim of the demo: a shared model trained across towers
    that never centralized their raw data learns the real hour/weekend
    congestion pattern well enough to meaningfully beat "always predict
    the most common tier" — not just "the run completes.\""""
    from tron.training.distributed.shard_client import run_shard

    tc, qs = client
    num_shards = 4
    resp = tc.post("/training/session", json={
        "session_id": "sess-towers-accuracy",
        "problem": "telecom_congestion",
        "num_shards": num_shards, "num_rounds": 8, "local_steps": 20, "lr": 0.5,
        "dataset_config": {"seed": 0, "samples_per_tower": 400, "test_samples_per_profile": 200},
    })
    assert resp.status_code == 200

    transport = _TestClientTransport(tc)
    errors = []

    def drive(k):
        try:
            run_shard(transport, "sess-towers-accuracy", k, poll_interval=0.02, max_wait_seconds=60)
        except Exception as exc:  # noqa: BLE001
            errors.append((k, repr(exc)))

    threads = [threading.Thread(target=drive, args=(k,)) for k in range(num_shards)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)
    assert not errors, errors

    result = tc.get("/training/session/sess-towers-accuracy/result").json()
    session = qs.training_sessions.get("sess-towers-accuracy")
    naive_baseline = np.bincount(session._y_test, minlength=NUM_TIERS).max() / len(session._y_test)

    assert result["accuracy"] > naive_baseline + 0.08, (
        f"trained accuracy {result['accuracy']:.3f} did not meaningfully beat "
        f"the naive majority-class baseline {naive_baseline:.3f}"
    )
