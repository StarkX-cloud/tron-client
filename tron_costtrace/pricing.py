"""Pinned per-model $/1K-token pricing.

These are published list prices snapshotted at the time this table was
written, not a live price feed — they will drift out of date and must be
updated by hand. A model missing from this table returns ``None`` from
``cost_usd`` rather than guessing, and the tracer surfaces that as an
explicit "cost unknown" record instead of silently reporting $0.

(input_usd_per_1k, output_usd_per_1k)
"""

from __future__ import annotations

PRICING_USD_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "o1": (0.015, 0.06),
    "o1-mini": (0.0011, 0.0044),
    "claude-opus-4": (0.015, 0.075),
    "claude-sonnet-4": (0.003, 0.015),
    "claude-haiku-3-5": (0.0008, 0.004),
}


def cost_usd(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """Returns None (unknown), not 0.0, when the model or token counts
    aren't available — a missing cost must never be reported as a free call.
    """
    if input_tokens is None or output_tokens is None:
        return None
    rates = PRICING_USD_PER_1K.get(model)
    if rates is None:
        return None
    input_rate, output_rate = rates
    return (input_tokens / 1000.0) * input_rate + (output_tokens / 1000.0) * output_rate


def is_known_model(model: str) -> bool:
    return model in PRICING_USD_PER_1K
