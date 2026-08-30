"""Serialization for the over-the-wire training path.

Everything that crosses the socket is bytes, and every decode is the exact
inverse of its encode — `np.save`/`np.load` is lossless for the float64
parameter vectors and the float64/int dataset arrays this uses, so a
round trip through here changes nothing. `average_vectors` performs the
*identical* `np.stack(...).mean(axis=0)` that
`local_sgd._average_flat_params` does in-process, which is what lets the
distributed run match the single-process run bit-for-bit.
"""
from __future__ import annotations

import io

import numpy as np


def encode_vector(v: np.ndarray) -> bytes:
    """A 1-D parameter vector -> bytes (npy format, no pickle)."""
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(v), allow_pickle=False)
    return buf.getvalue()


def decode_vector(data: bytes) -> np.ndarray:
    arr = np.load(io.BytesIO(data), allow_pickle=False)
    return np.ascontiguousarray(arr)


def encode_dataset(x: np.ndarray, y: np.ndarray) -> bytes:
    """A shard's (x, y) training data -> bytes (npz, no pickle)."""
    buf = io.BytesIO()
    np.savez(buf, x=x, y=y)
    return buf.getvalue()


def decode_dataset(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    loaded = np.load(io.BytesIO(data), allow_pickle=False)
    return np.ascontiguousarray(loaded["x"]), np.ascontiguousarray(loaded["y"])


def average_vectors(blobs: list[bytes]) -> bytes:
    """Barrier-average a round's per-shard parameter vectors. Byte-for-byte
    the same result as `local_sgd._average_flat_params` given the same
    inputs in the same order — same stack, same mean, same dtype."""
    if not blobs:
        raise ValueError("no vectors to average")
    arrs = [decode_vector(b) for b in blobs]
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(f"cannot average vectors of differing shapes: {shapes}")
    return encode_vector(np.stack(arrs, axis=0).mean(axis=0))
