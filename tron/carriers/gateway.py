"""CarrierGateway — the distributed carrier automation front door.

The gateway is the single object an application talks to. It:

* holds one or more carrier :class:`AdapterConfig` s (secret-free),
* routes a :class:`CarrierOp` to the right carrier,
* enforces the shared rate-limit budget and retry policy,
* runs each op **locally** or distributes it across the TRON worker fabric via
  ``@tron.remote`` (workers resolve their own credentials), and
* fans a batch of ops out across the cluster and gathers the results.

Distribution here scales *authorized* traffic and gives fault-tolerance; it is
not a way to exceed a carrier's rate limits. The rate limiter is one shared
budget, applied before dispatch, regardless of how many workers run the ops.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .base import (
    CarrierError,
    CarrierOp,
    CarrierResult,
    ConsentError,
    OpKind,
    RateLimiter,
    retry,
)
from .http import is_retryable
from .base import CarrierAPIError
from .tasks import AdapterConfig, run_op


class CarrierGateway:
    def __init__(self, *, rate_per_sec: float = 5.0, burst: Optional[int] = None,
                 max_attempts: int = 3) -> None:
        self._configs: Dict[str, AdapterConfig] = {}
        self._limiter = RateLimiter(rate_per_sec, burst=burst)
        self.max_attempts = max_attempts

    # -- registration ------------------------------------------------------- #
    def register(self, config: AdapterConfig) -> "CarrierGateway":
        self._configs[config.carrier] = config
        return self

    def register_verizon(self, **kw: Any) -> "CarrierGateway":
        return self.register(AdapterConfig(carrier="verizon", **kw))

    def register_att(self, **kw: Any) -> "CarrierGateway":
        return self.register(AdapterConfig(carrier="att", **kw))

    def carriers(self) -> List[str]:
        return sorted(self._configs)

    def _config_for(self, carrier: str) -> AdapterConfig:
        try:
            return self._configs[carrier]
        except KeyError as exc:
            raise CarrierError(
                f"carrier {carrier!r} not registered (have: {self.carriers()})"
            ) from exc

    # -- validation --------------------------------------------------------- #
    @staticmethod
    def _validate(op: CarrierOp) -> None:
        if op.requires_consent() and not op.consent:
            raise ConsentError(
                f"{op.kind.value} to {op.params.get('to')!r} requires consent=True; "
                "only send messages to recipients who have opted in")

    # -- execution ---------------------------------------------------------- #
    def execute(self, op: CarrierOp, *, distributed: bool = False,
                wait: bool = True) -> CarrierResult:
        """Run one op. Rate-limited and retried; local or distributed."""
        self._validate(op)
        config = self._config_for(op.carrier)
        self._limiter.acquire(1.0, wait=wait)

        if not distributed:
            return retry(lambda: run_op(op, config),
                         attempts=self.max_attempts,
                         retry_on=(CarrierAPIError,),
                         should_retry=is_retryable)

        # Distribute across the TRON fabric. Import lazily to avoid a hard
        # dependency at import time and any circular import.
        import tron

        remote_run = tron.remote(run_op, remote_only=True)
        future = remote_run(op, config)
        return future  # a MagicFuture; caller decides when to .get()

    def send_message(self, carrier: str, to: Any, text: str, *,
                     sender: Optional[str] = None, consent: bool = False,
                     distributed: bool = False, ref: Optional[str] = None,
                     **params: Any) -> CarrierResult:
        op = CarrierOp(
            kind=OpKind.SEND_MESSAGE, carrier=carrier, consent=consent, ref=ref,
            params={"to": to, "text": text, "from": sender, **params},
        )
        return self.execute(op, distributed=distributed)

    def manage_sim(self, carrier: str, action: str, *, distributed: bool = False,
                   ref: Optional[str] = None, **params: Any) -> CarrierResult:
        kind = {
            "list": OpKind.SIM_LIST,
            "status": OpKind.SIM_STATUS,
            "activate": OpKind.SIM_ACTIVATE,
            "deactivate": OpKind.SIM_DEACTIVATE,
        }.get(action)
        if kind is None:
            raise CarrierError(f"unknown SIM action {action!r}")
        return self.execute(CarrierOp(kind=kind, carrier=carrier, ref=ref,
                                      params=params), distributed=distributed)

    def query(self, carrier: str, what: str, *, distributed: bool = False,
              ref: Optional[str] = None, **params: Any) -> CarrierResult:
        kind = {
            "usage": OpKind.USAGE_QUERY,
            "network": OpKind.NETWORK_QUERY,
            "account": OpKind.ACCOUNT_QUERY,
        }.get(what)
        if kind is None:
            raise CarrierError(f"unknown query {what!r}")
        return self.execute(CarrierOp(kind=kind, carrier=carrier, ref=ref,
                                      params=params), distributed=distributed)

    # -- batch fan-out ------------------------------------------------------ #
    def map(self, ops: Iterable[CarrierOp], *, distributed: bool = True,
            wait: bool = True) -> List[CarrierResult]:
        """Run many ops. When distributed, fan out then gather futures."""
        ops = list(ops)
        for op in ops:
            self._validate(op)

        if not distributed:
            return [self.execute(op, distributed=False) for op in ops]

        futures = [self.execute(op, distributed=True) for op in ops]
        if not wait:
            return futures  # type: ignore[return-value]
        results: List[CarrierResult] = []
        for op, fut in zip(ops, futures):
            try:
                results.append(fut.get())
            except Exception as exc:  # gather failures as results, don't abort
                results.append(CarrierResult.failure(op, f"dispatch failed: {exc}"))
        return results
