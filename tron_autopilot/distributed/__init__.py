"""Distributed Execution Engine v1 — independent task parallelism.

Partitions a parallelizable workload into shards, submits them through the
existing pool (`submit_to_pool` / matcher / watchdog), collects ordered
results, and only prefers this path when the cost model says communication
overhead will not erase the speedup.

Not model/tensor parallelism. Not simulated multi-machine.
"""

from tron_autopilot.distributed.coordinator import DistributedCoordinator, run_distributed_map
from tron_autopilot.distributed.cost_model import ParallelCostEstimate, estimate_parallel_benefit
from tron_autopilot.distributed.partition import ShardSpec, partition_items
from tron_autopilot.distributed.ops import run_map_shard, reduce_map_results

__all__ = [
    "ShardSpec",
    "partition_items",
    "ParallelCostEstimate",
    "estimate_parallel_benefit",
    "DistributedCoordinator",
    "run_distributed_map",
    "run_map_shard",
    "reduce_map_results",
]
