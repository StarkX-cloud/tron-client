"""Master-side state for an over-the-wire distributed training run.

One `TrainingSession` owns:

- the problem (generated once on the master: non-IID shards + a held-out
  eval set), each shard's data stored as a content-addressed Artifact so
  the shard fetches real bytes, not a re-derivation;
- the per-round barrier: it collects each shard's trained parameter vector
  as an Artifact, and once all `num_shards` have reported for round r it
  averages them (`protocol.average_vectors`) into round r+1's starting
  point;
- spine recording: every (round, shard) pair is one Task —
  queued -> assigned -> started (logged when the shard pulls that round's
  init) -> completed (logged when it posts its trained vector, with the
  vector as the output Artifact). This is the exact event vocabulary
  `spine_integration.run_local_sgd_with_spine` uses, so a distributed run
  renders in the Grid identically to the in-process one.

The math is deliberately identical to `local_sgd.train_local_sgd`; see
this package's __init__ and `tests/test_distributed_training.py` for the
bit-for-bit parity guarantee.
"""
from __future__ import annotations

import threading
from typing import Optional

from tron.spine import ArtifactStore, EventLog, Task

from ..data import make_classification_dataset, make_non_iid_shards, train_test_split
from ..model import TinyMLP
from .protocol import average_vectors, decode_vector, encode_dataset, encode_vector

_ROUND_FN_TAG = b"tron.training.distributed.round"

# Bytes moved per shard per round on the wire = one upload of the trained
# vector + one download of the merged vector. The in-process metric in
# local_sgd counts only the conceptual "one sync's worth" per shard per
# round (upload side); we report both so the comparison is honest.
_BYTES_PER_FLOAT64 = 8


class TrainingSession:
    def __init__(
        self,
        session_id: str,
        *,
        num_shards: int,
        num_rounds: int,
        local_steps: int,
        lr: float,
        event_log: EventLog,
        artifact_store: ArtifactStore,
        model_kind: str = "numpy_mlp",
        model_config: Optional[dict] = None,
        dataset_config: Optional[dict] = None,
        skew: float = 0.9,
        shard_seed: int = 1,
    ):
        if model_kind != "numpy_mlp":
            raise ValueError(f"unsupported model_kind: {model_kind!r}")
        if num_shards < 1 or num_rounds < 1 or local_steps < 1:
            raise ValueError("num_shards, num_rounds, local_steps must all be >= 1")

        self.session_id = session_id
        self.num_shards = num_shards
        self.num_rounds = num_rounds
        self.local_steps = local_steps
        self.lr = float(lr)
        self.model_kind = model_kind
        self.model_config = dict(model_config or {})
        self.dataset_config = dict(dataset_config or {})
        self._log = event_log
        self._store = artifact_store
        self._lock = threading.Lock()

        self._build_problem(skew=skew, shard_seed=shard_seed)

        # round -> {shard_idx: artifact_id of that shard's trained vector}
        self._uploads: dict[int, dict[int, str]] = {r: {} for r in range(num_rounds)}
        # round -> artifact_id of the vector a shard should START that round from.
        # round 0 is the shared init; round r>0 is the merge of round r-1.
        self._round_init: dict[int, str] = {0: self._shared_init_hash}
        # round -> artifact_id of the merged vector produced at the END of it
        self._merged: dict[int, str] = {}
        # (round, shard) -> task_id, so lifecycle events all target one task
        self._task_ids: dict[tuple[int, int], str] = {}
        self._wire_bytes = 0

    # -- problem setup -------------------------------------------------

    def _build_problem(self, *, skew: float, shard_seed: int) -> None:
        cfg = self.dataset_config
        num_features = int(cfg.get("num_features", 8))
        num_classes = int(cfg.get("num_classes", 4))
        x_all, y_all = make_classification_dataset(
            num_samples=int(cfg.get("num_samples", 2000)),
            num_features=num_features,
            num_classes=num_classes,
            seed=int(cfg.get("seed", 0)),
            class_sep=float(cfg.get("class_sep", 1.0)),
        )
        x_train, y_train, x_test, y_test = train_test_split(
            x_all, y_all, test_fraction=float(cfg.get("test_fraction", 0.2))
        )
        self._x_test, self._y_test = x_test, y_test

        shards = make_non_iid_shards(
            x_train, y_train, num_shards=self.num_shards,
            num_classes=num_classes, skew=skew, seed=shard_seed,
        )

        self.model_config.setdefault("input_dim", num_features)
        self.model_config.setdefault("hidden_dim", 16)
        self.model_config.setdefault("num_classes", num_classes)
        self.model_config.setdefault("seed", 42)

        # Shard data -> Artifacts (the shard downloads these real bytes).
        self._shard_data_hashes: list[str] = []
        for x, y in shards:
            art = self._store.put(encode_dataset(x, y))
            self._shard_data_hashes.append(art.artifact_id)

        # Shared init = a fresh model's flat params, same as
        # local_sgd._shared_init_replicas. Deterministic from model_config,
        # but stored + transmitted so the wire path is real.
        init_vec = self._new_model().get_flat_params()
        self._shared_init_hash = self._store.put(encode_vector(init_vec)).artifact_id
        self._num_params = init_vec.size

        self._fn_hash = self._store.put(_ROUND_FN_TAG).artifact_id

    def _new_model(self) -> TinyMLP:
        return TinyMLP(**self.model_config)

    # -- shard-facing operations ------------------------------------------

    def shard_data(self, shard_idx: int) -> bytes:
        self._check_shard(shard_idx)
        return self._store.get(self._shard_data_hashes[shard_idx])

    def round_init(self, round_idx: int, shard_idx: int) -> Optional[bytes]:
        """The vector shard `shard_idx` should start round `round_idx` from,
        or None if the previous round's barrier hasn't cleared yet (the
        caller turns None into HTTP 202). Logs queued/assigned/started for
        that (round, shard) Task the first time it's pulled.
        """
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
                    "session": self.session_id,
                    "round": round_idx,
                    "shard": shard_idx,
                    "kind": "distributed-local-sgd",
                }
                task = Task.new(
                    fn_hash=self._fn_hash,
                    input_hashes=(init_hash, self._shard_data_hashes[shard_idx]),
                    attempt=round_idx,
                    metadata=meta,
                )
                self._task_ids[key] = task.task_id
                node_id = f"shard-{shard_idx}"
                # metadata goes in the "queued" event body so EventLog._fold
                # (and the Grid, which folds identically in JS) carries it
                # onto the reconstructed Task.
                self._log.append(
                    task.task_id, "queued",
                    {"fn_hash": self._fn_hash,
                     "input_hashes": [init_hash, self._shard_data_hashes[shard_idx]],
                     "metadata": meta},
                )
                self._log.append(task.task_id, "assigned", {"node_id": node_id}, node_id=node_id)
                self._log.append(task.task_id, "started", {"node_id": node_id}, node_id=node_id)
            self._wire_bytes += self._num_params * _BYTES_PER_FLOAT64
            return self._store.get(init_hash)

    def submit_round(
        self, round_idx: int, shard_idx: int, params_bytes: bytes, node_id: Optional[str] = None
    ) -> dict:
        """Record shard `shard_idx`'s trained vector for `round_idx`. When
        the last shard for the round reports, average and open the next
        round. Returns a small status dict.
        """
        self._check_shard(shard_idx)
        if not 0 <= round_idx < self.num_rounds:
            raise ValueError(f"round {round_idx} out of range")

        vec = decode_vector(params_bytes)
        if vec.shape != (self._num_params,):
            raise ValueError(
                f"expected a vector of {self._num_params} params, got shape {vec.shape}"
            )

        with self._lock:
            art = self._store.put(encode_vector(vec))
            self._uploads[round_idx][shard_idx] = art.artifact_id
            self._wire_bytes += vec.size * _BYTES_PER_FLOAT64

            key = (round_idx, shard_idx)
            task_id = self._task_ids.get(key)
            if task_id is None:
                # A shard that posted without first pulling init — still
                # record a complete lifecycle so the log isn't missing it.
                meta = {
                    "session": self.session_id, "round": round_idx,
                    "shard": shard_idx, "kind": "distributed-local-sgd",
                }
                task = Task.new(
                    fn_hash=self._fn_hash,
                    input_hashes=(self._round_init[round_idx], self._shard_data_hashes[shard_idx]),
                    attempt=round_idx,
                    metadata=meta,
                )
                task_id = task.task_id
                self._task_ids[key] = task_id
                nid = node_id or f"shard-{shard_idx}"
                self._log.append(task_id, "queued", {"fn_hash": self._fn_hash, "metadata": meta})
                self._log.append(task_id, "assigned", {"node_id": nid}, node_id=nid)
                self._log.append(task_id, "started", {"node_id": nid}, node_id=nid)

            nid = node_id or f"shard-{shard_idx}"
            self._log.append(
                task_id, "completed",
                {"output_hash": art.artifact_id,
                 "transfer_bytes": vec.size * _BYTES_PER_FLOAT64},
                node_id=nid,
            )

            round_complete = len(self._uploads[round_idx]) == self.num_shards
            if round_complete and round_idx not in self._merged:
                ordered = [self._uploads[round_idx][k] for k in range(self.num_shards)]
                merged_bytes = average_vectors([self._store.get(h) for h in ordered])
                merged_hash = self._store.put(merged_bytes).artifact_id
                self._merged[round_idx] = merged_hash
                if round_idx + 1 < self.num_rounds:
                    self._round_init[round_idx + 1] = merged_hash

        return {
            "session_id": self.session_id,
            "round": round_idx,
            "shard": shard_idx,
            "round_complete": round_complete,
            "rounds_merged": sorted(self._merged),
        }

    # -- status / result ------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            return {
                "session_id": self.session_id,
                "num_shards": self.num_shards,
                "num_rounds": self.num_rounds,
                "local_steps": self.local_steps,
                "lr": self.lr,
                "num_params": self._num_params,
                "uploads_per_round": {r: sorted(v) for r, v in self._uploads.items() if v},
                "rounds_merged": sorted(self._merged),
                "finished": (self.num_rounds - 1) in self._merged,
                "wire_bytes_transferred": self._wire_bytes,
                "algorithmic_comm_bytes": self.num_rounds * self.num_shards
                * self._num_params * _BYTES_PER_FLOAT64,
            }

    def final_vector_bytes(self) -> Optional[bytes]:
        last = self.num_rounds - 1
        with self._lock:
            h = self._merged.get(last)
        return self._store.get(h) if h else None

    def result(self) -> Optional[dict]:
        """Final merged model's held-out accuracy, or None if unfinished."""
        blob = self.final_vector_bytes()
        if blob is None:
            return None
        model = self._new_model()
        model.set_flat_params(decode_vector(blob))
        st = self.status()
        st["accuracy"] = model.accuracy(self._x_test, self._y_test)
        st["final_vector_hash"] = self._merged[self.num_rounds - 1]
        return st

    def _check_shard(self, shard_idx: int) -> None:
        if not 0 <= shard_idx < self.num_shards:
            raise ValueError(f"shard {shard_idx} out of range [0, {self.num_shards})")


class TrainingSessionRegistry:
    """Thread-safe map of session_id -> TrainingSession. queue_server owns
    one of these; the shard-facing endpoints look sessions up here."""

    def __init__(self):
        self._sessions: dict[str, TrainingSession] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str, **kwargs) -> TrainingSession:
        session = TrainingSession(session_id, **kwargs)
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[TrainingSession]:
        with self._lock:
            return self._sessions.get(session_id)
