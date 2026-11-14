"""Standardized workload fingerprint for comparing computational tasks.

Never stores prompts, source code, model inputs, or outputs — only
structural characteristics safe for learning under governance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WorkloadFingerprint:
    model_name: str
    target: str
    policy: str | None
    device: str | None
    dtype: str | None
    batch_size: int = 1
    map_op: str | None = None
    parallel_workers: int | None = None
    reliability: str = "tolerant"
    requires_efficiency_runtime: bool = False
    customer_id: str | None = None
    tenant_id: str | None = None
    # Hardware / capacity signals (coarse — no secrets)
    worker_sku: str | None = None
    gpu: bool | None = None
    region: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def history_key(self) -> str:
        """Align with RuntimeHistory.key (model|target|policy|device|dtype)."""
        return "|".join(
            [
                self.model_name or "unknown",
                self.target or "unknown",
                self.policy or "none",
                self.device or "none",
                self.dtype or "none",
            ]
        )

    def class_id(self) -> str:
        """Stable hash of structural fields (excludes customer/tenant)."""
        payload = {
            "model_name": self.model_name,
            "target": self.target,
            "policy": self.policy,
            "device": self.device,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "map_op": self.map_op,
            "parallel_workers": self.parallel_workers,
            "reliability": self.reliability,
            "requires_efficiency_runtime": self.requires_efficiency_runtime,
            "worker_sku": self.worker_sku,
            "gpu": self.gpu,
            "region": self.region,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return "wf_" + hashlib.sha256(blob).hexdigest()[:16]

    def feature_vector(self) -> dict[str, float | str | int | bool | None]:
        """Sparse feature map for the learner (categorical + numeric)."""
        return {
            "model_name": self.model_name,
            "target": self.target,
            "policy": self.policy or "none",
            "device": self.device or "none",
            "dtype": self.dtype or "none",
            "batch_size": int(self.batch_size or 1),
            "map_op": self.map_op or "none",
            "parallel_workers": int(self.parallel_workers or 1),
            "reliability": self.reliability,
            "requires_efficiency_runtime": bool(self.requires_efficiency_runtime),
            "worker_sku": self.worker_sku or "none",
            "gpu": bool(self.gpu) if self.gpu is not None else False,
            "region": self.region or "none",
        }


def fingerprint_from_plan(plan: dict | Any, *, outcome: dict | None = None) -> WorkloadFingerprint:
    """Build fingerprint from a DecisionLog plan row (or WorkloadPlan.to_dict())."""
    if hasattr(plan, "to_dict"):
        plan = plan.to_dict()
    spec = plan.get("spec") or {}
    chosen = plan.get("chosen") or {}
    if outcome and not chosen:
        chosen = {
            "target": outcome.get("target"),
            "policy": None,
            "device": None,
            "dtype": None,
        }
    # Prefer chosen option; fall back to outcome target.
    target = (chosen.get("target") if chosen else None) or (
        (outcome or {}).get("target") if outcome else None
    ) or "local"
    batch = int(spec.get("batch_size") or 1)
    max_workers = spec.get("max_parallel_workers")
    pool = plan.get("pool_status") or {}
    workers = pool.get("workers") if isinstance(pool, dict) else None
    sku = None
    gpu = None
    region = None
    if isinstance(workers, list) and workers:
        w0 = workers[0] if isinstance(workers[0], dict) else {}
        sku = w0.get("sku")
        gpu = w0.get("gpu")
        region = w0.get("location") or w0.get("region")
    return WorkloadFingerprint(
        model_name=str(spec.get("model_name") or "unknown"),
        target=str(target),
        policy=chosen.get("policy") if chosen else None,
        device=chosen.get("device") if chosen else None,
        dtype=chosen.get("dtype") if chosen else None,
        batch_size=batch,
        map_op=spec.get("map_op"),
        parallel_workers=int(max_workers) if max_workers else None,
        reliability=str(spec.get("reliability") or "tolerant"),
        requires_efficiency_runtime=bool(spec.get("requires_efficiency_runtime")),
        customer_id=_customer_from_spec(spec),
        tenant_id=spec.get("tenant_id"),
        worker_sku=sku,
        gpu=gpu,
        region=region,
    )


def _customer_from_spec(spec: dict) -> str | None:
    task = spec.get("costtrace_task")
    if isinstance(task, str) and task.startswith("customer:"):
        return task.split(":", 1)[1] or None
    return spec.get("customer_id")
