"""Pinned GPU / cloud SKU hourly prices for Autopilot economics.

Snapshot list prices — not a live feed. Unknown SKUs return ``None``
(never guessed as $0). Workers may advertise ``sku`` + ``hourly_rate_usd``;
Autopilot prefers measured/advertised rates over free-form CLI rates when
present.
"""

from __future__ import annotations

# (sku, hourly_usd) — Akamai/Linode GPU + common cloud reference SKUs
GPU_SKU_HOURLY_USD: dict[str, float] = {
    # Linode / Akamai
    "g2-gpu-rtx4000a1-s": 0.52,
    "g2-gpu-rtx4000a1-m": 1.04,
    "g1-gpu-rtx6000-1": 1.50,
    "g6-nanode-1": 0.0075,
    "g6-standard-2": 0.018,
    "g6-dedicated-2": 0.036,
    # Colab / reference (not billable via API — for comparison only)
    "colab-t4": 0.0,
    # AWS-ish reference labels workers may set via TRON_WORKER_SKU
    "aws-g4dn.xlarge": 0.526,
    "aws-p3.2xlarge": 3.06,
    "gcp-n1-standard-4-t4": 0.35,
}


def hourly_usd_for_sku(sku: str | None) -> float | None:
    if not sku:
        return None
    return GPU_SKU_HOURLY_USD.get(sku.strip())


def resolve_hourly_rate(
    *,
    explicit: float | None = None,
    sku: str | None = None,
    advertised: float | None = None,
    fallback: float | None = None,
) -> float | None:
    """Priority: explicit caller rate → worker-advertised → SKU table → fallback."""
    if explicit is not None:
        return float(explicit)
    if advertised is not None:
        return float(advertised)
    sku_rate = hourly_usd_for_sku(sku)
    if sku_rate is not None:
        return float(sku_rate)
    if fallback is not None:
        return float(fallback)
    return None
