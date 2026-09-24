"""Normalize carrier responses into a small, stable shape across carriers.

Verizon, AT&T/Jasper and T-Mobile each return device and usage data in their
own JSON layouts. These helpers map the common fields (iccid, status, rate
plan, data usage) into one vocabulary so callers don't special-case per
carrier. The untouched carrier payload is always kept under ``_raw`` so nothing
is lost.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Per-carrier field aliases -> normalized key.
_DEVICE_FIELDS = {
    "iccid": ("iccid", "deviceIds", "iccId"),
    "status": ("status", "state", "connectionStatus"),
    "rate_plan": ("ratePlan", "servicePlan", "carrierRatePlan"),
    "imsi": ("imsi",),
    "msisdn": ("msisdn", "mdn"),
}

_USAGE_FIELDS = {
    "iccid": ("iccid", "iccId"),
    "data_bytes": ("ctdDataUsage", "dataUsage", "bytesUsed", "dataUsageBytes"),
    "sms_count": ("ctdSMSUsage", "smsUsage", "smsCount"),
    "voice_minutes": ("ctdVoiceUsage", "voiceUsage", "voiceMinutes"),
}


def _first(record: Dict[str, Any], keys) -> Any:
    for k in keys:
        if isinstance(record, dict) and record.get(k) not in (None, ""):
            return record[k]
    return None


def normalize_device(record: Dict[str, Any]) -> Dict[str, Any]:
    """Map one carrier device record to normalized fields (keeps ``_raw``)."""
    out: Dict[str, Any] = {norm: _first(record, aliases)
                           for norm, aliases in _DEVICE_FIELDS.items()}
    out["_raw"] = record
    return out


def normalize_usage(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {norm: _first(record, aliases)
                           for norm, aliases in _USAGE_FIELDS.items()}
    out["_raw"] = record
    return out


def extract_devices(payload: Any) -> List[Dict[str, Any]]:
    """Pull a list of device records out of a carrier's list response.

    Handles the common envelopes: a bare list, ``{"devices": [...]}``,
    Verizon's ``{"devices": [...], "hasMoreData": bool}``, and Jasper's
    ``{"devices": [...], "lastPage": bool}``. A single-device object becomes a
    one-item list.
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return [d for d in payload if isinstance(d, dict)]
    if isinstance(payload, dict):
        for key in ("devices", "deviceList", "results", "items"):
            val = payload.get(key)
            if isinstance(val, list):
                return [d for d in val if isinstance(d, dict)]
        # Looks like a single device record.
        if any(k in payload for k in ("iccid", "iccId", "status", "state")):
            return [payload]
    return []
