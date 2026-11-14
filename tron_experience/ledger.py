"""Workload Intelligence Ledger — structured experience from real executions.

Built by joining DecisionLog plan + outcome rows. Never fabricates records.
Does not store prompts, code, or model I/O.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tron_experience.fingerprint import WorkloadFingerprint, fingerprint_from_plan
from tron_experience.governance import GovernancePolicy


@dataclass
class ExperienceRecord:
    plan_id: str
    fingerprint: WorkloadFingerprint
    predicted_runtime_seconds: float | None
    actual_runtime_seconds: float
    predicted_cost_usd: float | None
    actual_cost_usd: float | None
    predicted_peak_memory_bytes: int | None = None
    actual_peak_memory_bytes: int | None = None
    prediction_source: str | None = None
    success: bool = True
    strategy: str | None = None
    target: str | None = None
    requeue_or_recovery: dict = field(default_factory=dict)
    optimizations: dict = field(default_factory=dict)
    model_version_id: str | None = None
    source_log: str | None = None
    timestamp: float = field(default_factory=time.time)
    # Capacity / network snapshot (coarse)
    fleet_alive_count: int | None = None
    capacity_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint.to_dict()
        d["class_id"] = self.fingerprint.class_id()
        return d


class ExperienceLedger:
    def __init__(
        self,
        path: str | os.PathLike | None = None,
        governance: GovernancePolicy | None = None,
    ):
        default = os.environ.get("TRON_EXPERIENCE_LEDGER", "tron_experience_ledger.jsonl")
        self.path = Path(path or default)
        self.governance = governance or GovernancePolicy()

    def append(self, record: ExperienceRecord) -> None:
        if not self.governance.allows_record(record):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = record.to_dict()
        # Strip anything that looks like prompt/code content.
        row.pop("prompt", None)
        row.pop("result", None)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def read_all(
        self,
        *,
        customer_id: str | None = None,
        tenant_id: str | None = None,
        allow_shared: bool | None = None,
    ) -> list[ExperienceRecord]:
        if not self.path.exists():
            return []
        shared = (
            self.governance.allow_shared_learning
            if allow_shared is None
            else allow_shared
        )
        out: list[ExperienceRecord] = []
        cutoff = self.governance.retention_cutoff()
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ts = float(row.get("timestamp") or 0)
                if cutoff is not None and ts < cutoff:
                    continue
                fp_raw = row.get("fingerprint") or {}
                fp = WorkloadFingerprint(**{
                    k: fp_raw.get(k)
                    for k in WorkloadFingerprint.__dataclass_fields__
                })
                if customer_id and fp.customer_id and fp.customer_id != customer_id:
                    if not shared:
                        continue
                    if not self.governance.customer_opted_into_shared(fp.customer_id):
                        continue
                if tenant_id and fp.tenant_id and fp.tenant_id != tenant_id:
                    if not shared:
                        continue
                out.append(
                    ExperienceRecord(
                        plan_id=str(row.get("plan_id") or ""),
                        fingerprint=fp,
                        predicted_runtime_seconds=row.get("predicted_runtime_seconds"),
                        actual_runtime_seconds=float(row.get("actual_runtime_seconds") or 0),
                        predicted_cost_usd=row.get("predicted_cost_usd"),
                        actual_cost_usd=row.get("actual_cost_usd"),
                        predicted_peak_memory_bytes=row.get("predicted_peak_memory_bytes"),
                        actual_peak_memory_bytes=row.get("actual_peak_memory_bytes"),
                        prediction_source=row.get("prediction_source"),
                        success=bool(row.get("success", True)),
                        strategy=row.get("strategy"),
                        target=row.get("target"),
                        requeue_or_recovery=dict(row.get("requeue_or_recovery") or {}),
                        optimizations=dict(row.get("optimizations") or {}),
                        model_version_id=row.get("model_version_id"),
                        source_log=row.get("source_log"),
                        timestamp=ts,
                        fleet_alive_count=row.get("fleet_alive_count"),
                        capacity_notes=list(row.get("capacity_notes") or []),
                    )
                )
        return out

    def ingest_decision_log(
        self,
        log_path: str | os.PathLike,
        *,
        customer_id: str | None = None,
    ) -> int:
        """Join plan+outcome rows from a DecisionLog JSONL into the ledger."""
        path = Path(log_path)
        if not path.exists():
            return 0
        plans: dict[str, dict] = {}
        outcomes: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("kind")
            if kind == "plan":
                plans[str(row.get("plan_id"))] = row
            elif kind == "outcome":
                outcomes.append(row)
            elif "actual_runtime_seconds" in row and "plan_id" in row:
                # Already an outcome-shaped row without kind.
                outcomes.append(row)
            elif "chosen" in row and "plan_id" in row:
                plans[str(row.get("plan_id"))] = row

        n = 0
        for oc in outcomes:
            pid = str(oc.get("plan_id") or "")
            plan = plans.get(pid) or {"plan_id": pid, "spec": {}, "chosen": {"target": oc.get("target")}}
            fp = fingerprint_from_plan(plan, outcome=oc)
            if customer_id and not fp.customer_id:
                fp = WorkloadFingerprint(**{**fp.to_dict(), "customer_id": customer_id})
            # Skip rows without measured actual runtime.
            if oc.get("actual_runtime_seconds") is None:
                continue
            # Never ingest raw result / prompts.
            pred_src = None
            chosen = (plan.get("chosen") or {})
            src_signals = chosen.get("source_signals") or {}
            if isinstance(src_signals, dict):
                pred_src = src_signals.get("prediction_source") or src_signals.get("pred_src")
            pool = plan.get("pool_status") or {}
            alive = pool.get("alive_count") if isinstance(pool, dict) else None
            rec = ExperienceRecord(
                plan_id=pid,
                fingerprint=fp,
                predicted_runtime_seconds=oc.get("predicted_runtime_seconds"),
                actual_runtime_seconds=float(oc["actual_runtime_seconds"]),
                predicted_cost_usd=oc.get("predicted_cost_usd"),
                actual_cost_usd=oc.get("actual_cost_usd"),
                predicted_peak_memory_bytes=oc.get("predicted_peak_memory_bytes"),
                actual_peak_memory_bytes=oc.get("actual_peak_memory_bytes"),
                prediction_source=pred_src,
                success=bool(oc.get("success", True)),
                strategy=oc.get("strategy"),
                target=oc.get("target") or fp.target,
                requeue_or_recovery=dict(oc.get("recovery") or {}),
                optimizations={
                    "policy": fp.policy,
                    "dtype": fp.dtype,
                    "device": fp.device,
                    "batch_size": fp.batch_size,
                },
                source_log=str(path),
                timestamp=float(oc.get("timestamp") or time.time()),
                fleet_alive_count=int(alive) if isinstance(alive, (int, float)) else None,
            )
            self.append(rec)
            n += 1
        return n

    def summary(self, records: Iterable[ExperienceRecord] | None = None) -> dict:
        rows = list(records) if records is not None else self.read_all()
        classes = {r.fingerprint.class_id() for r in rows}
        successes = sum(1 for r in rows if r.success)
        return {
            "workloads_observed": len(rows),
            "workload_classes": len(classes),
            "successes": successes,
            "failures": len(rows) - successes,
            "targets": sorted({r.target or r.fingerprint.target for r in rows}),
        }
