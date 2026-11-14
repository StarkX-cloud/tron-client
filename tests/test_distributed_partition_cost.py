"""Unit tests for partition + communication-aware cost model."""

from tron_autopilot.distributed.cost_model import estimate_parallel_benefit
from tron_autopilot.distributed.ops import closed_form_sum_squares, reduce_map_results, run_map_shard
from tron_autopilot.distributed.partition import partition_items


def test_partition_contiguous_and_complete():
    items = list(range(10))
    shards = partition_items(items, 3, workload_id="t")
    assert len(shards) == 3
    flat = [x for s in shards for x in s.items]
    assert flat == items
    keys = {s.idempotency_key for s in shards}
    assert len(keys) == 3


def test_partition_clamps_to_items():
    shards = partition_items([1, 2], 8, workload_id="t")
    assert len(shards) == 2


def test_sum_squares_op_and_reduce():
    shards = partition_items(list(range(20)), 4, workload_id="sq")
    results = [
        run_map_shard(s.shard_id, list(s.items), op="sum_squares", idempotency_key=s.idempotency_key)
        for s in shards
    ]
    reduced = reduce_map_results(results, op="sum_squares")
    assert reduced["total"] == closed_form_sum_squares(20)


def test_cost_model_refuses_without_serial():
    est = estimate_parallel_benefit(
        n_items=100,
        n_workers=4,
        serial_seconds=None,
        hourly_rate_usd=0.5,
    )
    assert est.worth_it is False


def test_cost_model_refuses_when_comm_dominates():
    # Tiny serial work, huge latency → not worth it
    est = estimate_parallel_benefit(
        n_items=10,
        n_workers=4,
        serial_seconds=0.05,
        hourly_rate_usd=0.5,
        avg_latency_ms=500.0,
        bandwidth_mbps=1.0,
        min_speedup=1.15,
    )
    assert est.worth_it is False


def test_cost_model_accepts_clear_speedup():
    est = estimate_parallel_benefit(
        n_items=1000,
        n_workers=4,
        serial_seconds=40.0,
        hourly_rate_usd=0.5,
        avg_latency_ms=20.0,
        bandwidth_mbps=100.0,
        bytes_per_item=8.0,
    )
    assert est.worth_it is True
    assert est.predicted_parallel_wall_seconds < est.predicted_serial_seconds
