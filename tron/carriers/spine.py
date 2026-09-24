"""Spine integration for carrier ops.

Records each carrier operation into the TRON execution spine's append-only
event log, so carrier work shows up in the same replay/time-travel substrate
the 3D Grid renders from. Targets that identify people or devices (phone
numbers, ICCIDs) are **redacted to a short content hash** before they touch the
log — the event stream stays useful for tracing without storing PII or secrets.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .base import CarrierOp, CarrierResult

try:  # spine is core, but stay import-safe.
    from tron.spine import content_hash
    _HAVE_SPINE = True
except Exception:  # pragma: no cover - spine always present in practice
    import hashlib

    def content_hash(data: bytes) -> str:  # type: ignore
        return hashlib.sha256(data).hexdigest()

    _HAVE_SPINE = False


# Params whose values identify a person or device: redact to a hash prefix.
_SENSITIVE_KEYS = {"to", "from", "iccid", "device_ids", "msisdn", "imei"}


def _redact_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_redact_value(v) for v in value]
    digest = content_hash(str(value).encode("utf-8"))
    return f"sha256:{digest[:12]}"


def redact_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a log-safe copy of op params: sensitive targets hashed, text
    replaced by its length, everything else kept as-is."""
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if k in _SENSITIVE_KEYS:
            out[k] = _redact_value(v)
        elif k == "text":
            out["text_len"] = len(str(v))
        else:
            out[k] = v
    return out


def op_task_id(op: CarrierOp) -> str:
    """A stable content-address for an op, safe to use as an EventLog task_id."""
    redacted = redact_params(op.params)
    canonical = "|".join([
        "carrier", op.carrier,
        "kind", op.kind.value,
        "params", repr(sorted(redacted.items())),
        "ref", op.ref or "",
    ])
    return content_hash(canonical.encode("utf-8"))


def log_submitted(event_log, op: CarrierOp, *, node_id: Optional[str] = None,
                  distributed: bool = False) -> str:
    task_id = op_task_id(op)
    if event_log is not None:
        event_log.append(
            task_id, "carrier_op_submitted",
            data={
                "carrier": op.carrier,
                "kind": op.kind.value,
                "params": redact_params(op.params),
                "ref": op.ref,
                "distributed": distributed,
            },
            node_id=node_id,
            timestamp=time.time(),
        )
    return task_id


def log_result(event_log, op: CarrierOp, result: CarrierResult,
               *, task_id: Optional[str] = None,
               node_id: Optional[str] = None) -> None:
    if event_log is None:
        return
    tid = task_id or op_task_id(op)
    event_log.append(
        tid,
        "carrier_op_completed" if result.ok else "carrier_op_failed",
        data={
            "carrier": op.carrier,
            "kind": op.kind.value,
            "ok": result.ok,
            "status": result.status,
            "error": result.error,
            "ref": op.ref,
        },
        node_id=node_id,
        timestamp=time.time(),
    )
