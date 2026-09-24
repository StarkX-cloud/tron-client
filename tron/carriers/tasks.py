"""Worker-side execution of carrier operations.

The functions here are what actually run on a TRON worker. They are top-level
(so cloudpickle can ship them) and they reconstruct an adapter from a
**secret-free** :class:`AdapterConfig`, resolving credentials from the
worker's own environment via ``credentials_from_env``. Credentials therefore
never travel in a task payload or over the wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base import CarrierOp, CarrierResult, OpKind, CarrierError
from .att import ATTAdapter
from .tmobile import TMobileAdapter
from .verizon import VerizonAdapter


# Registry of carrier name -> adapter class.
ADAPTER_CLASSES = {
    "verizon": VerizonAdapter,
    "att": ATTAdapter,
    "tmobile": TMobileAdapter,
}


@dataclass
class AdapterConfig:
    """Secret-free description of how to build an adapter on a worker."""

    carrier: str
    base_url: Optional[str] = None
    timeout: float = 30.0
    endpoints: Dict[str, str] = field(default_factory=dict)
    messaging_url: Optional[str] = None

    def build(self):
        try:
            cls = ADAPTER_CLASSES[self.carrier]
        except KeyError as exc:
            raise CarrierError(f"unknown carrier {self.carrier!r}") from exc
        kwargs: Dict[str, Any] = {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "endpoints": self.endpoints or None,
        }
        # Adapters that support messaging accept a messaging_url kwarg.
        if self.messaging_url is not None:
            kwargs["messaging_url"] = self.messaging_url
        return cls(**kwargs)


def run_op(op: CarrierOp, config: AdapterConfig) -> CarrierResult:
    """Execute a single carrier op on this machine.

    Returns a :class:`CarrierResult`; raises only for consent violations, which
    are a programming/authorization error rather than a transient failure.
    """
    adapter = config.build()
    return adapter.execute(op)
