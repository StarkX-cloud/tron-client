"""Shared data shapes for the Economic Autopilot.

Every monetary field that is not measured from a real run is labeled as
an estimate derived from a **user-supplied** hourly rate (same contract
as ``tron_efficiency.results.report.generate_report``). Missing values
are ``None``, never fabricated zeros.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WorkloadSpec:
    """What the user asks TRON to run, plus economic constraints.

    ``hourly_rate_usd`` is required for cost accounting: it is the caller's
    configured $/hour for compute time on this machine (local), not a live
    cloud price feed. ``pooled_hourly_rate_usd`` optionally prices pool
    compute differently; when omitted, the same rate is used for both.
    """

    model_name: str
    hourly_rate_usd: float
    budget_usd: float | None = None
    deadline_seconds: float | None = None
    reliability: str = "tolerant"  # "tolerant" | "strict"
    pooled_hourly_rate_usd: float | None = None
    expected_runtime_seconds: float | None = None
    requires_multi_node: bool = False
    gpu_budget_bytes_override: int | None = None
    master_url: str | None = None
    # Optional API-call cost ledger task name (tron_costtrace). When set,
    # execute() records a costtrace CallRecord for the run's wall time /
    # success; token costs stay None unless the caller fills them via the
    # returned handle — compute $ comes from hourly_rate × seconds.
    costtrace_task: str | None = None
    # Efficiency-workload knobs (used when executing via tron_efficiency).
    prompt: str = "The quick brown fox jumps over the lazy dog, and"
    max_new_tokens: int = 32
    # When True, pooled candidates require workers that advertised
    # efficiency_runtime (torch/transformers/accelerate).
    requires_efficiency_runtime: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CandidateOption:
    """One feasible (or explicitly infeasible) way to run the workload."""

    option_id: str
    target: str  # "local" | "pooled" | "dedicated" | "rejected"
    policy: str | None  # "baseline" | "offload" | None
    device: str | None
    dtype: str | None
    memory_budget_bytes: int | None
    required_memory_gb: float | None
    hourly_rate_usd: float
    predicted_runtime_seconds: float | None
    predicted_cost_usd: float | None
    predicted_peak_memory_bytes: int | None
    feasible: bool
    rejection_reasons: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    source_signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorkloadPlan:
    """The autopilot's decision: chosen option + full candidate set + why."""

    plan_id: str
    spec: WorkloadSpec
    chosen: CandidateOption | None
    candidates: list[CandidateOption]
    placement_decision: dict | None  # tron_router.PlacementDecision.to_dict()
    local_recommendation: dict | None
    pool_status: dict | None
    safety: dict
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "spec": self.spec.to_dict(),
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "placement_decision": self.placement_decision,
            "local_recommendation": self.local_recommendation,
            "pool_status": self.pool_status,
            "safety": self.safety,
            "rationale": self.rationale,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
        }


@dataclass
class ExecutionOutcome:
    """Measured result of executing a plan, plus predicted-vs-actual feedback."""

    plan_id: str
    success: bool
    target: str
    strategy: str
    actual_runtime_seconds: float
    actual_cost_usd: float | None
    actual_peak_memory_bytes: int | None
    predicted_runtime_seconds: float | None
    predicted_cost_usd: float | None
    predicted_peak_memory_bytes: int | None
    result: Any = None
    error: str | None = None
    recovery: dict = field(default_factory=dict)
    decision_rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    job_id: str | None = None
    timestamp: float = 0.0
    # Optional measured efficiency metrics (tok/s, GPU/CPU peaks, etc.).
    metrics: dict = field(default_factory=dict)

    def feedback_delta(self) -> dict:
        """Predicted vs actual deltas. None where either side is unknown."""

        def _delta(pred, actual):
            if pred is None or actual is None:
                return None
            return actual - pred

        return {
            "runtime_seconds_delta": _delta(
                self.predicted_runtime_seconds, self.actual_runtime_seconds
            ),
            "cost_usd_delta": _delta(self.predicted_cost_usd, self.actual_cost_usd),
            "peak_memory_bytes_delta": _delta(
                self.predicted_peak_memory_bytes, self.actual_peak_memory_bytes
            ),
            "success": self.success,
        }

    def to_dict(self) -> dict:
        data = asdict(self)
        # result may be non-JSON; keep a safe summary
        if data.get("result") is not None and not isinstance(
            data["result"], (str, int, float, bool, dict, list, type(None))
        ):
            data["result"] = repr(data["result"])
        data["feedback"] = self.feedback_delta()
        return data
