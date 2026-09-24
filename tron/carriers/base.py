"""Carrier integration primitives for TRON.

This package lets TRON tasks talk to telecom carriers' **official developer
APIs** (Verizon ThingSpace, AT&T IoT / Control Center, and configurable
messaging endpoints) and run those calls across the distributed worker fabric.

Design rules that hold across every adapter in this package:

* **Authorized use only.** An adapter never invents credentials or endpoints.
  It talks to the tenant/API you configure with credentials you supply
  (env vars or explicit ``CarrierCredentials``). With no credentials it
  refuses to run rather than guessing.
* **Honor carrier limits, never evade them.** Distribution across TRON workers
  is for throughput and fault-tolerance of *your own* authorized traffic, not
  for spreading requests to dodge a carrier's rate limits or detection. The
  shared ``RateLimiter`` is a single global budget, not a per-worker
  multiplier.
* **Messaging is consent-gated.** Outbound messages require an explicit
  ``consent`` marker on the request; the gateway refuses to send without it.
* **No secret leakage.** Credentials are never placed in logs, task payloads,
  or ``repr``/``str`` output.
"""

from __future__ import annotations

import abc
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# Module logger. Never log credential material; adapters log op kind, carrier,
# HTTP status and redacted identifiers only.
logger = logging.getLogger("tron.carriers")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class CarrierError(Exception):
    """Base class for all carrier-integration errors."""


class CarrierAuthError(CarrierError):
    """Missing or rejected credentials for a carrier API."""


class CarrierConfigError(CarrierError):
    """The adapter is misconfigured (missing base URL, endpoint, etc.)."""


class CarrierAPIError(CarrierError):
    """The carrier API returned a non-success response."""

    def __init__(self, message: str, *, status: Optional[int] = None,
                 body: Optional[str] = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class ConsentError(CarrierError):
    """An outbound message was submitted without the required consent marker."""


class RateLimitExceeded(CarrierError):
    """A local rate-limit budget was exhausted and waiting was disabled."""


# --------------------------------------------------------------------------- #
# Operations & results
# --------------------------------------------------------------------------- #
class OpKind(str, Enum):
    """The families of task an adapter may support."""

    SEND_MESSAGE = "send_message"
    SIM_LIST = "sim_list"
    SIM_ACTIVATE = "sim_activate"
    SIM_DEACTIVATE = "sim_deactivate"
    SIM_STATUS = "sim_status"
    USAGE_QUERY = "usage_query"
    NETWORK_QUERY = "network_query"
    ACCOUNT_QUERY = "account_query"


@dataclass
class CarrierOp:
    """A single, serializable unit of carrier work.

    ``CarrierOp`` is intentionally plain data so it can be shipped to a TRON
    worker via cloudpickle and executed there. It carries no credentials —
    the worker resolves those from its own environment / configuration.
    """

    kind: OpKind
    carrier: str
    params: Dict[str, Any] = field(default_factory=dict)
    # Consent marker required for SEND_MESSAGE. Must be explicitly True.
    consent: bool = False
    # Free-form correlation id for tracing an op through the spine event log.
    ref: Optional[str] = None

    def requires_consent(self) -> bool:
        return self.kind == OpKind.SEND_MESSAGE


@dataclass
class CarrierResult:
    """Normalized result of a ``CarrierOp`` across carriers."""

    ok: bool
    kind: OpKind
    carrier: str
    data: Any = None
    error: Optional[str] = None
    ref: Optional[str] = None
    status: Optional[int] = None

    @classmethod
    def success(cls, op: "CarrierOp", data: Any, status: Optional[int] = None) -> "CarrierResult":
        return cls(ok=True, kind=op.kind, carrier=op.carrier, data=data,
                   ref=op.ref, status=status)

    @classmethod
    def failure(cls, op: "CarrierOp", error: str, status: Optional[int] = None) -> "CarrierResult":
        return cls(ok=False, kind=op.kind, carrier=op.carrier, error=error,
                   ref=op.ref, status=status)


@dataclass
class PreflightResult:
    """Outcome of a read-only credential/connectivity check for one carrier."""

    carrier: str
    ok: bool
    detail: str = ""
    base_url: Optional[str] = None
    status: Optional[int] = None


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
@dataclass
class CarrierCredentials:
    """Credentials for a carrier API.

    Load order for :meth:`from_env` is documented per adapter. The object never
    reveals secret material in ``repr``/``str``.
    """

    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    account_name: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)

    def require(self, *names: str) -> None:
        missing = [n for n in names if not getattr(self, n, None) and n not in self.extra]
        if missing:
            raise CarrierAuthError(
                "Missing required credential(s): " + ", ".join(missing)
            )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        present = [n for n in ("client_id", "client_secret", "api_key",
                               "username", "password")
                   if getattr(self, n)]
        return f"CarrierCredentials(set={present}, account={self.account_name!r})"

    __str__ = __repr__


def _env(*names: str) -> Optional[str]:
    for n in names:
        val = os.getenv(n)
        if val:
            return val
    return None


# --------------------------------------------------------------------------- #
# Rate limiting (single shared token bucket)
# --------------------------------------------------------------------------- #
class RateLimiter:
    """A thread-safe token bucket for honoring a carrier's rate limit.

    This is a *single* budget shared by all callers in a process. It exists to
    keep authorized traffic under the carrier's published ceiling — not to be
    replicated per worker to exceed it.
    """

    def __init__(self, rate_per_sec: float, burst: Optional[int] = None) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        self.rate = float(rate_per_sec)
        self.capacity = float(burst if burst is not None else max(1.0, rate_per_sec))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    def acquire(self, tokens: float = 1.0, *, wait: bool = True,
                timeout: Optional[float] = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                if not wait:
                    raise RateLimitExceeded("rate limit budget exhausted")
                needed = (tokens - self._tokens) / self.rate
            if deadline is not None and time.monotonic() + needed > deadline:
                raise RateLimitExceeded("timed out waiting for rate limit budget")
            time.sleep(min(needed, 0.25))


def retry(fn: Callable[[], Any], *, attempts: int = 3, base_delay: float = 0.5,
          retry_on: tuple = (CarrierAPIError,),
          should_retry: Optional[Callable[[Exception], bool]] = None) -> Any:
    """Call ``fn`` with exponential backoff on transient failures."""
    last: Optional[Exception] = None
    for i in range(max(1, attempts)):
        try:
            return fn()
        except retry_on as exc:  # type: ignore[misc]
            last = exc
            if should_retry is not None and not should_retry(exc):
                raise
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))
    assert last is not None
    raise last


# --------------------------------------------------------------------------- #
# Adapter interface
# --------------------------------------------------------------------------- #
class CarrierAdapter(abc.ABC):
    """Base class for a single carrier's official-API integration."""

    name: str = "carrier"

    def __init__(self, credentials: Optional[CarrierCredentials] = None,
                 *, base_url: Optional[str] = None,
                 timeout: float = 30.0,
                 endpoints: Optional[Dict[str, str]] = None) -> None:
        self.credentials = credentials or self.credentials_from_env()
        self.base_url = (base_url or self.default_base_url()).rstrip("/")
        self.timeout = timeout
        self.endpoints = {**self.default_endpoints(), **(endpoints or {})}

    # -- configuration hooks (override per carrier) ------------------------- #
    @classmethod
    @abc.abstractmethod
    def credentials_from_env(cls) -> CarrierCredentials: ...

    @abc.abstractmethod
    def default_base_url(self) -> str: ...

    def default_endpoints(self) -> Dict[str, str]:
        return {}

    @abc.abstractmethod
    def supported(self) -> List[OpKind]: ...

    # -- execution ---------------------------------------------------------- #
    def supports(self, kind: OpKind) -> bool:
        return kind in self.supported()

    def execute(self, op: CarrierOp) -> CarrierResult:
        """Validate and dispatch a single op to the concrete handler."""
        if not self.supports(op.kind):
            return CarrierResult.failure(
                op, f"{self.name} does not support operation {op.kind.value}")
        if op.requires_consent() and not op.consent:
            raise ConsentError(
                f"{op.kind.value} requires an explicit consent=True marker")
        logger.info("carrier op start carrier=%s kind=%s ref=%s",
                    self.name, op.kind.value, op.ref)
        try:
            data, status = self._dispatch(op)
            result = CarrierResult.success(op, data, status=status)
            logger.info("carrier op ok carrier=%s kind=%s status=%s ref=%s",
                        self.name, op.kind.value, status, op.ref)
            return result
        except CarrierError as exc:
            status = getattr(exc, "status", None)
            logger.warning("carrier op fail carrier=%s kind=%s status=%s ref=%s: %s",
                           self.name, op.kind.value, status, op.ref, exc)
            return CarrierResult.failure(op, str(exc), status=status)

    # -- preflight ---------------------------------------------------------- #
    def preflight(self) -> PreflightResult:
        """Read-only credential + connectivity check for this carrier.

        Runs the carrier's cheapest authenticated read (token exchange and/or a
        single-record list) with **no side effects**, so operators can verify
        real credentials before going live. Never activates a SIM or sends a
        message. Override ``_preflight`` per carrier.
        """
        try:
            detail, status = self._preflight()
            logger.info("preflight ok carrier=%s status=%s", self.name, status)
            return PreflightResult(self.name, True, detail=detail,
                                   base_url=self.base_url, status=status)
        except CarrierError as exc:
            status = getattr(exc, "status", None)
            logger.warning("preflight fail carrier=%s: %s", self.name, exc)
            return PreflightResult(self.name, False, detail=str(exc),
                                   base_url=self.base_url, status=status)

    def _preflight(self) -> tuple[str, Optional[int]]:
        """Return ``(detail, http_status)`` for a successful read-only check."""
        raise CarrierConfigError(f"{self.name} has no preflight check")

    @abc.abstractmethod
    def _dispatch(self, op: CarrierOp) -> tuple[Any, Optional[int]]:
        """Return ``(data, http_status)`` for a supported, validated op."""
        ...

    def endpoint(self, key: str, **fmt: Any) -> str:
        try:
            tmpl = self.endpoints[key]
        except KeyError as exc:
            raise CarrierConfigError(
                f"{self.name}: no endpoint configured for {key!r}") from exc
        path = tmpl.format(**fmt)
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"
