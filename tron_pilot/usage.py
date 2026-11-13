"""Customer usage accounting hooks — not billing custody.

Emits ``UsageEvent`` rows that an external billing system can ingest.
No wallets, invoices, payments, or regulated financial products.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class UsageEvent:
    customer_id: str
    event_type: str  # "workload_complete" | "workload_failed"
    plan_id: str | None
    model_name: str | None
    target: str | None
    success: bool
    runtime_seconds: float
    cost_usd: float | None
    strategy: str | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    provider_id: str | None = None
    worker_name: str | None = None
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class UsageLedger:
    def __init__(
        self,
        path: str | os.PathLike | None = None,
        webhook: Callable[[UsageEvent], None] | None = None,
    ):
        default = os.environ.get("TRON_USAGE_LEDGER", "tron_usage.jsonl")
        self.path = Path(path or default)
        self.webhook = webhook

    def record(self, event: UsageEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), default=str) + "\n")
        if self.webhook is not None:
            try:
                self.webhook(event)
            except Exception:  # noqa: BLE001 — never break execute path
                pass

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows


def usage_report(
    ledger: UsageLedger,
    *,
    customer_id: str | None = None,
    since: float | None = None,
    until: float | None = None,
) -> dict:
    rows = ledger.read_all()
    if customer_id:
        rows = [r for r in rows if r.get("customer_id") == customer_id]
    if since is not None:
        rows = [r for r in rows if float(r.get("timestamp") or 0) >= since]
    if until is not None:
        rows = [r for r in rows if float(r.get("timestamp") or 0) <= until]

    successes = [r for r in rows if r.get("success")]
    failures = [r for r in rows if not r.get("success")]
    cost_known = [
        float(r["cost_usd"]) for r in successes if r.get("cost_usd") is not None
    ]
    runtime = [float(r["runtime_seconds"]) for r in rows if r.get("runtime_seconds") is not None]
    return {
        "customer_id": customer_id,
        "since": since,
        "until": until,
        "events": len(rows),
        "successes": len(successes),
        "failures": len(failures),
        "total_cost_usd_estimate": sum(cost_known) if cost_known else None,
        "total_runtime_seconds": sum(runtime) if runtime else 0.0,
        "note": (
            "cost_usd is hourly_rate × wall time — an estimate for external "
            "billing systems, not a TRON-issued invoice or custody transfer"
        ),
        "rows": rows,
    }
