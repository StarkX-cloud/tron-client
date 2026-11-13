"""When is independent-task parallelism worth the communication cost?

Does **not** assume more workers ⇒ faster. Compares serial runtime to
parallel wall time including measured (or conservatively estimated)
network overhead and transfer time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ParallelCostEstimate:
    worth_it: bool
    n_shards: int
    predicted_serial_seconds: float | None
    predicted_parallel_compute_seconds: float | None
    predicted_comm_seconds: float
    predicted_parallel_wall_seconds: float | None
    predicted_serial_cost_usd: float | None
    predicted_parallel_cost_usd: float | None
    transfer_bytes: float
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_parallel_benefit(
    *,
    n_items: int,
    n_workers: int,
    serial_seconds: float | None,
    hourly_rate_usd: float,
    avg_latency_ms: float | None = None,
    bandwidth_mbps: float | None = None,
    bytes_per_item: float = 64.0,
    rounds: int = 2,
    # Fixed per-shard pool/queue overhead when latency unknown.
    default_per_shard_overhead_seconds: float = 0.15,
    # Refuse parallel if predicted speedup is below this factor.
    min_speedup: float = 1.15,
) -> ParallelCostEstimate:
    """Estimate whether sharding across ``n_workers`` beats serial local/pool.

    Communication model (honest, conservative):
    - ``rounds`` round-trips per shard (submit + result poll) × latency
    - transfer time from ``n_items * bytes_per_item`` at measured bandwidth
    - if latency/bandwidth unknown, use ``default_per_shard_overhead_seconds``
      per shard (documented as estimate, not fabricated zero)
    """
    rationale: list[str] = []
    n_workers = max(0, int(n_workers))
    n_items = max(0, int(n_items))

    if n_workers < 2:
        return ParallelCostEstimate(
            worth_it=False,
            n_shards=0,
            predicted_serial_seconds=serial_seconds,
            predicted_parallel_compute_seconds=None,
            predicted_comm_seconds=0.0,
            predicted_parallel_wall_seconds=None,
            predicted_serial_cost_usd=_cost(hourly_rate_usd, serial_seconds),
            predicted_parallel_cost_usd=None,
            transfer_bytes=0.0,
            rationale=["need >= 2 alive workers for independent-task parallelism"],
        )
    if n_items < 2:
        return ParallelCostEstimate(
            worth_it=False,
            n_shards=0,
            predicted_serial_seconds=serial_seconds,
            predicted_parallel_compute_seconds=None,
            predicted_comm_seconds=0.0,
            predicted_parallel_wall_seconds=None,
            predicted_serial_cost_usd=_cost(hourly_rate_usd, serial_seconds),
            predicted_parallel_cost_usd=None,
            transfer_bytes=0.0,
            rationale=["workload has < 2 items — nothing to partition"],
        )

    n_shards = min(n_workers, n_items)
    transfer_bytes = float(n_items) * float(bytes_per_item)

    # Communication seconds
    if avg_latency_ms is not None and avg_latency_ms >= 0:
        # Each shard: submit RTT + at least one status RTT (rounds).
        per_shard_lat = (avg_latency_ms / 1000.0) * max(1, rounds)
        # Shards submit concurrently → latency roughly one shard's worth,
        # not n_shards × latency (coordinator pipelines submits).
        lat_component = per_shard_lat
        rationale.append(
            f"using measured avg latency {avg_latency_ms:.1f}ms × {rounds} rounds "
            f"≈ {lat_component:.3f}s coordination (concurrent submits)"
        )
    else:
        lat_component = default_per_shard_overhead_seconds
        rationale.append(
            f"latency unknown — using conservative {default_per_shard_overhead_seconds:.2f}s "
            "per-wave pool overhead (not assuming free coordination)"
        )

    if bandwidth_mbps is not None and bandwidth_mbps > 0:
        # megabits/s → bytes/s
        bytes_per_sec = (bandwidth_mbps * 1_000_000.0) / 8.0
        xfer = transfer_bytes / bytes_per_sec
        rationale.append(
            f"transfer ~{transfer_bytes:.0f}B at {bandwidth_mbps:.2f}Mbps ≈ {xfer:.3f}s"
        )
    else:
        # ~1ms per KB conservative floor when bandwidth unknown
        xfer = max(0.001, transfer_bytes / 1_000_000.0)
        rationale.append(
            f"bandwidth unknown — floor transfer estimate {xfer:.3f}s for {transfer_bytes:.0f}B"
        )

    comm = lat_component + xfer

    if serial_seconds is None or serial_seconds <= 0:
        rationale.append(
            "serial runtime unknown — cannot prove speedup; refuse distributed "
            "(set expected_runtime_seconds or build RuntimeHistory first)"
        )
        return ParallelCostEstimate(
            worth_it=False,
            n_shards=n_shards,
            predicted_serial_seconds=serial_seconds,
            predicted_parallel_compute_seconds=None,
            predicted_comm_seconds=comm,
            predicted_parallel_wall_seconds=None,
            predicted_serial_cost_usd=None,
            predicted_parallel_cost_usd=None,
            transfer_bytes=transfer_bytes,
            rationale=rationale,
        )

    parallel_compute = serial_seconds / float(n_shards)
    parallel_wall = parallel_compute + comm
    speedup = serial_seconds / parallel_wall if parallel_wall > 0 else 0.0
    worth = speedup >= min_speedup and parallel_wall < serial_seconds

    rationale.append(
        f"serial {serial_seconds:.3f}s → ~{parallel_compute:.3f}s compute/shard "
        f"+ {comm:.3f}s comm = {parallel_wall:.3f}s wall "
        f"(speedup {speedup:.2f}x, threshold {min_speedup:.2f}x)"
    )
    if worth:
        rationale.append("parallelization predicted beneficial after communication cost")
    else:
        rationale.append(
            "communication overhead makes distributed inefficient vs serial — preferring local/pooled"
        )

    return ParallelCostEstimate(
        worth_it=worth,
        n_shards=n_shards,
        predicted_serial_seconds=serial_seconds,
        predicted_parallel_compute_seconds=parallel_compute,
        predicted_comm_seconds=comm,
        predicted_parallel_wall_seconds=parallel_wall,
        predicted_serial_cost_usd=_cost(hourly_rate_usd, serial_seconds),
        predicted_parallel_cost_usd=_cost(hourly_rate_usd, parallel_wall),
        transfer_bytes=transfer_bytes,
        rationale=rationale,
    )


def _cost(hourly: float, seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return (hourly / 3600.0) * float(seconds)
