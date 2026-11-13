"""Service-level measurement from usage + decision history.

Measurement only — not an SLA contract or credits system.
"""

from __future__ import annotations

import math
from typing import Any

from tron_pilot.usage import UsageLedger


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def sla_report(
    ledger: UsageLedger | None = None,
    *,
    customer_id: str | None = None,
    deadline_seconds: float | None = None,
    since: float | None = None,
    until: float | None = None,
) -> dict[str, Any]:
    """Aggregate availability / latency / deadline-miss from UsageLedger."""
    ledger = ledger or UsageLedger()
    rows = ledger.read_all()
    if customer_id:
        rows = [r for r in rows if r.get("customer_id") == customer_id]
    if since is not None:
        rows = [r for r in rows if float(r.get("timestamp") or 0) >= since]
    if until is not None:
        rows = [r for r in rows if float(r.get("timestamp") or 0) <= until]

    successes = [r for r in rows if r.get("success")]
    failures = [r for r in rows if not r.get("success")]
    runtimes = sorted(
        float(r["runtime_seconds"])
        for r in rows
        if r.get("runtime_seconds") is not None
    )
    deadline_misses = 0
    if deadline_seconds is not None:
        deadline_misses = sum(
            1
            for r in successes
            if float(r.get("runtime_seconds") or 0) > float(deadline_seconds)
        )

    n = len(rows)
    return {
        "customer_id": customer_id,
        "events": n,
        "successes": len(successes),
        "failures": len(failures),
        "availability": (len(successes) / n) if n else None,
        "runtime_p50_seconds": _percentile(runtimes, 50),
        "runtime_p95_seconds": _percentile(runtimes, 95),
        "runtime_p99_seconds": _percentile(runtimes, 99),
        "deadline_seconds": deadline_seconds,
        "deadline_misses": deadline_misses if deadline_seconds is not None else None,
        "deadline_miss_rate": (
            (deadline_misses / len(successes)) if successes and deadline_seconds is not None else None
        ),
        "note": (
            "Measurement only — not an SLA credit system or contractual guarantee"
        ),
    }
