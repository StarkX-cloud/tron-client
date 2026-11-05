"""Data shape for one recorded LLM call.

One ``CallRecord`` per call, written as one JSON line. Nothing here is
aggregated or estimated at record time — that happens in report.py, from the
raw records, so the raw log always stays the ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class CallRecord:
    task: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    latency_seconds: float
    retry_count: int
    success: bool | None
    error: str | None
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CallRecord":
        return cls(**data)
