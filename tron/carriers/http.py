"""Thin HTTP helper shared by carrier adapters.

Wraps ``requests`` with a per-adapter session, a default timeout, and
normalized error handling that raises :class:`CarrierAPIError` (retryable on
5xx / 429) without ever logging credentials.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from .base import CarrierAPIError


# HTTP status codes worth retrying with backoff.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class HttpSession:
    def __init__(self, timeout: float = 30.0) -> None:
        self._session = requests.Session()
        self.timeout = timeout

    def request(self, method: str, url: str, *,
                headers: Optional[Dict[str, str]] = None,
                json: Any = None,
                params: Optional[Dict[str, Any]] = None,
                data: Any = None,
                expect_json: bool = True) -> tuple[Any, int]:
        try:
            resp = self._session.request(
                method.upper(), url,
                headers=headers, json=json, params=params, data=data,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            # Network-level failure: retryable.
            raise CarrierAPIError(f"request to {url} failed: {exc}", status=None)

        if resp.status_code >= 400:
            body = _safe_body(resp)
            raise CarrierAPIError(
                f"{method.upper()} {url} -> HTTP {resp.status_code}",
                status=resp.status_code,
                body=body,
            )

        if not expect_json:
            return resp.text, resp.status_code
        if not resp.content:
            return None, resp.status_code
        try:
            return resp.json(), resp.status_code
        except ValueError:
            return resp.text, resp.status_code


def is_retryable(exc: CarrierAPIError) -> bool:
    return exc.status is None or exc.status in RETRYABLE_STATUS


def _safe_body(resp: "requests.Response") -> str:
    try:
        text = resp.text or ""
    except Exception:  # pragma: no cover - defensive
        return ""
    return text[:2000]
