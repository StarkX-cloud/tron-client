"""Deterministic partitioning of independent work items into shards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ShardSpec:
    shard_id: int
    items: tuple[Any, ...]
    idempotency_key: str
    index_start: int
    index_end: int  # exclusive

    def to_dict(self) -> dict:
        return asdict(self)


def _key_for(shard_id: int, items: list[Any], workload_id: str) -> str:
    payload = json.dumps(
        {"workload_id": workload_id, "shard_id": shard_id, "items": items},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def partition_items(
    items: list[Any],
    n_shards: int,
    *,
    workload_id: str = "default",
    min_shard_size: int = 1,
) -> list[ShardSpec]:
    """Split ``items`` into up to ``n_shards`` contiguous shards.

    Contiguous slices keep ordering stable for reproducibility. Empty input
    yields no shards. ``n_shards`` is clamped so each shard has at least
    ``min_shard_size`` items when possible.
    """
    if not items:
        return []
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    if min_shard_size < 1:
        raise ValueError("min_shard_size must be >= 1")

    max_by_size = max(1, len(items) // min_shard_size)
    n = min(n_shards, len(items), max_by_size)

    # Even split with remainder distributed to the first shards.
    base = len(items) // n
    rem = len(items) % n
    shards: list[ShardSpec] = []
    cursor = 0
    for i in range(n):
        take = base + (1 if i < rem else 0)
        chunk = items[cursor : cursor + take]
        shards.append(
            ShardSpec(
                shard_id=i,
                items=tuple(chunk),
                idempotency_key=_key_for(i, list(chunk), workload_id),
                index_start=cursor,
                index_end=cursor + take,
            )
        )
        cursor += take
    return shards
