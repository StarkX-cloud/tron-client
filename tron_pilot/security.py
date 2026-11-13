"""Trust boundaries and auth helpers for the global compute surface.

Honest model (v1):
- Workers authenticate with per-registration opaque tokens.
- Submit / Autopilot APIs optionally require shared secrets
  (``TRON_SUBMIT_AUTH_TOKEN``, ``TRON_AUTOPILOT_API_TOKEN``).
- Provider ``verified`` is an operator attestation, not hardware root-of-trust.
- Cloudpickle jobs run with OS resource limits only — **not** multi-tenant
  sandbox isolation. Do not admit untrusted code onto shared hosts.

No financial custody. Secrets stay in env / files — never logged.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TrustModel:
    submit_auth_required: bool
    autopilot_auth_required: bool
    sandbox: str
    notes: tuple[str, ...]


def current_trust_model() -> TrustModel:
    return TrustModel(
        submit_auth_required=bool(os.environ.get("TRON_SUBMIT_AUTH_TOKEN")),
        autopilot_auth_required=bool(os.environ.get("TRON_AUTOPILOT_API_TOKEN")),
        sandbox="process_timeout_and_rlimit_only",
        notes=(
            "Worker auth_token required for heartbeat/claim/complete",
            "Provider verified flag is operator attestation",
            "No container/seccomp multi-tenant isolation in v1",
            "No payment rails or custody — usage hooks only",
        ),
    )


def auth_headers(
    *,
    autopilot: bool = False,
    submit: bool = False,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if autopilot:
        tok = os.environ.get("TRON_AUTOPILOT_API_TOKEN")
        if tok:
            headers["X-TRON-AUTH"] = tok
    if submit:
        tok = os.environ.get("TRON_SUBMIT_AUTH_TOKEN")
        if tok:
            headers["X-TRON-AUTH"] = tok
    return headers


def constant_time_equals(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return hmac.compare_digest(a, b)


def redact_secrets(row: dict) -> dict:
    """Drop known secret fields before returning public snapshots."""
    out = dict(row)
    for k in ("auth_token", "token", "secret", "password", "api_key"):
        if k in out:
            out[k] = "***"
    return out
