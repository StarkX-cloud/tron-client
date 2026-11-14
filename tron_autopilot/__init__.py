"""TRON Economic Autopilot — unified decision + execution for AI workloads.

Thin orchestration over existing TRON components. Does **not** rewrite
placement, memory policies, the watchdog, or costtrace — it connects them:

- ``tron_efficiency.advisor`` — what resources / config a workload needs
- ``tron_router`` — local vs pooled vs dedicated, pool handoff
- ``tron/spine/matcher`` + ``queue_server`` watchdog — where / how to recover
- ``tron_costtrace`` + user hourly rates — cost measurement & feedback

See ``writeup/economic-autopilot.md`` for architecture and honest limits.
"""

from __future__ import annotations

from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.fleet import FleetCapacity, query_fleet_capacity
from tron_autopilot.pricing import GPU_SKU_HOURLY_USD, hourly_usd_for_sku
from tron_autopilot.types import ExecutionOutcome, WorkloadPlan, WorkloadSpec

__all__ = [
    "EconomicAutopilot",
    "WorkloadSpec",
    "WorkloadPlan",
    "ExecutionOutcome",
    "FleetCapacity",
    "query_fleet_capacity",
    "GPU_SKU_HOURLY_USD",
    "hourly_usd_for_sku",
]
