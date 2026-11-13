"""Shard ops for Distributed Execution Engine v1.

``run_map_shard`` is the cloudpickle-friendly worker entrypoint.
``reduce_map_results`` merges ordered shard outputs into one result.
"""

from __future__ import annotations

from typing import Any


def run_map_shard(
    shard_id: int,
    items: list[Any],
    op: str = "sum_squares",
    idempotency_key: str | None = None,
) -> dict:
    """Execute one independent shard. Pure function of (op, items)."""
    if op == "sum_squares":
        partial = 0
        for x in items:
            xi = int(x)
            partial += xi * xi
        return {
            "shard_id": shard_id,
            "op": op,
            "n": len(items),
            "partial": partial,
            "idempotency_key": idempotency_key,
        }
    if op == "sum":
        return {
            "shard_id": shard_id,
            "op": op,
            "n": len(items),
            "partial": sum(int(x) for x in items),
            "idempotency_key": idempotency_key,
        }
    raise ValueError(f"unsupported map op: {op!r}")


def reduce_map_results(shard_results: list[dict], op: str = "sum_squares") -> dict:
    """Combine shard outputs. Requires complete, ordered coverage."""
    if not shard_results:
        return {"op": op, "total": 0, "n": 0, "shards": 0}
    ordered = sorted(shard_results, key=lambda r: int(r["shard_id"]))
    # Detect gaps / duplicates
    seen = set()
    for r in ordered:
        sid = int(r["shard_id"])
        if sid in seen:
            raise ValueError(f"duplicate shard_id {sid} in results")
        seen.add(sid)
    expected = set(range(len(ordered)))
    if seen != expected:
        raise ValueError(f"incomplete shard set: got {sorted(seen)} expected {sorted(expected)}")

    total = sum(int(r["partial"]) for r in ordered)
    n = sum(int(r["n"]) for r in ordered)
    return {
        "op": op,
        "total": total,
        "n": n,
        "shards": len(ordered),
        "shard_partials": [int(r["partial"]) for r in ordered],
    }


def closed_form_sum_squares(n_items: int) -> int:
    """Sum of squares 0^2 + 1^2 + ... + (n-1)^2."""
    n = int(n_items)
    return (n - 1) * n * (2 * n - 1) // 6 if n > 0 else 0
