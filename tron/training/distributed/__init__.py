"""Phase 3, over a real wire.

The in-process Phase 3 demo (`tron/training/local_sgd.py`) counts the
communication bytes a distributed local-SGD run *would* send, but the
"shards" are Python objects in one process and nothing crosses a socket.
This package closes that gap: shards are separate OS processes that
serialize their parameters and `POST` them to the master over HTTP, the
master stores each as a content-addressed spine Artifact, barrier-averages
once every shard for a round has reported, and serves the merged vector
back for the next round.

The transport is the only thing that changes. The math is identical — same
seeds, same `_sample_batch`/`train_step` sequence, same averaging
operation — so `tests/test_distributed_training.py` asserts the wire path
produces a final model **bit-for-bit equal** to `local_sgd.train_local_sgd`
run in one process. If that ever diverges, the transport is corrupting the
computation and the test says so.

"Genuinely separate physical machines" is then a deployment topology, not
a code change: point `shard_worker` at a different `--master` URL and the
same bytes go over the real internet instead of loopback. Commit 3d6939b
already exercised that path (a real worker against a deployed Render
instance); this package makes training itself run across it.

- `protocol.py`     bytes <-> numpy vectors / datasets; the averaging op
- `param_server.py` master-side session state + spine recording
- `shard_client.py` `run_shard()` — one shard's full round loop
- `shard_worker.py` `python -m tron.training.distributed.shard_worker ...`
"""
from .protocol import (
    average_vectors,
    decode_dataset,
    decode_vector,
    encode_dataset,
    encode_vector,
)
from .param_server import TrainingSession, TrainingSessionRegistry
from .shard_client import run_shard

__all__ = [
    "average_vectors",
    "decode_dataset",
    "decode_vector",
    "encode_dataset",
    "encode_vector",
    "TrainingSession",
    "TrainingSessionRegistry",
    "run_shard",
]
