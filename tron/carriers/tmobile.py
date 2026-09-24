"""T-Mobile adapter — official developer APIs.

Integrates **T-Mobile IoT / Control Center** for SIM/device management, status
and usage, plus a configurable enterprise messaging endpoint. Authentication is
OAuth2 *client_credentials* (HTTP Basic on the token endpoint -> bearer token),
which T-Mobile's developer platform uses for machine-to-machine access. Base
URL and endpoints are overridable so you can point the adapter at the exact
host/tenant T-Mobile provisioned for you.

Credentials (``credentials_from_env``) read, in order:

* ``TMOBILE_CLIENT_ID``      -> client_id     (OAuth2)
* ``TMOBILE_CLIENT_SECRET``  -> client_secret (OAuth2)
* ``TMOBILE_ACCOUNT_NAME``   -> account_name   (billing / account id)
* ``TMOBILE_BASE_URL``       -> base_url override
* ``TMOBILE_MESSAGING_URL``  -> messaging endpoint override
"""

from __future__ import annotations

import base64
import os
import threading
import time
from typing import Any, Dict, List, Optional

from .base import (
    CarrierAdapter,
    CarrierAuthError,
    CarrierConfigError,
    CarrierCredentials,
    CarrierOp,
    OpKind,
    _env,
)
from .http import HttpSession


# T-Mobile IoT platform host. Override for your assigned tenant.
DEFAULT_BASE = "https://iot.api.t-mobile.com"


class TMobileAdapter(CarrierAdapter):
    name = "tmobile"

    def __init__(self, credentials: Optional[CarrierCredentials] = None,
                 *, base_url: Optional[str] = None, timeout: float = 30.0,
                 endpoints: Optional[Dict[str, str]] = None,
                 messaging_url: Optional[str] = None) -> None:
        super().__init__(credentials, base_url=base_url, timeout=timeout,
                         endpoints=endpoints)
        self.messaging_url = messaging_url or _env("TMOBILE_MESSAGING_URL")
        self._http = HttpSession(timeout=self.timeout)
        self._lock = threading.Lock()
        self._access_token: Optional[str] = None
        self._access_expiry: float = 0.0

    # -- configuration ------------------------------------------------------ #
    @classmethod
    def credentials_from_env(cls) -> CarrierCredentials:
        return CarrierCredentials(
            client_id=_env("TMOBILE_CLIENT_ID"),
            client_secret=_env("TMOBILE_CLIENT_SECRET"),
            account_name=_env("TMOBILE_ACCOUNT_NAME"),
        )

    def default_base_url(self) -> str:
        return os.getenv("TMOBILE_BASE_URL", DEFAULT_BASE)

    def default_endpoints(self) -> Dict[str, str]:
        return {
            "oauth_token": "oauth2/v1/token",
            "device_list": "iot/v1/devices",
            "device_get": "iot/v1/devices/{iccid}",
            "device_update": "iot/v1/devices/{iccid}",
            "usage_get": "iot/v1/devices/{iccid}/usage",
        }

    def supported(self) -> List[OpKind]:
        base = [
            OpKind.SIM_LIST, OpKind.SIM_ACTIVATE, OpKind.SIM_DEACTIVATE,
            OpKind.SIM_STATUS, OpKind.USAGE_QUERY, OpKind.ACCOUNT_QUERY,
        ]
        if self.messaging_url:
            base.append(OpKind.SEND_MESSAGE)
        return base

    # -- auth --------------------------------------------------------------- #
    def _ensure_access_token(self) -> str:
        with self._lock:
            if self._access_token and time.time() < self._access_expiry - 30:
                return self._access_token
            self.credentials.require("client_id", "client_secret")
            basic = base64.b64encode(
                f"{self.credentials.client_id}:{self.credentials.client_secret}".encode()
            ).decode()
            data, _ = self._http.request(
                "POST", self.endpoint("oauth_token"),
                headers={"Authorization": f"Basic {basic}",
                         "Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "client_credentials"},
            )
            if not isinstance(data, dict) or "access_token" not in data:
                raise CarrierAuthError("T-Mobile OAuth2 returned no access_token")
            self._access_token = data["access_token"]
            self._access_expiry = time.time() + float(data.get("expires_in", 3600))
            return self._access_token

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        token = self._ensure_access_token()
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        if extra:
            headers.update(extra)
        return headers

    def _account_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        acct = params.get("account_name") or self.credentials.account_name
        return {"accountId": acct} if acct else {}

    def _iccid(self, params: Dict[str, Any]) -> str:
        iccid = params.get("iccid")
        if not iccid:
            raise CarrierConfigError("this T-Mobile operation needs an 'iccid'")
        return str(iccid)

    # -- dispatch ----------------------------------------------------------- #
    def _dispatch(self, op: CarrierOp) -> tuple[Any, Optional[int]]:
        if op.kind == OpKind.SEND_MESSAGE:
            return self._send_message(op.params)
        if op.kind in (OpKind.SIM_LIST, OpKind.ACCOUNT_QUERY):
            return self._device_list(op.params)
        if op.kind == OpKind.SIM_STATUS:
            return self._device_get(op.params)
        if op.kind == OpKind.SIM_ACTIVATE:
            return self._set_status(op.params, "active")
        if op.kind == OpKind.SIM_DEACTIVATE:
            return self._set_status(op.params, "inactive")
        if op.kind == OpKind.USAGE_QUERY:
            return self._usage(op.params)
        raise CarrierConfigError(f"unhandled op {op.kind}")

    def _device_list(self, params: Dict[str, Any]):
        query = self._account_params(params)
        for k in ("pageSize", "pageNumber"):
            if params.get(k) is not None:
                query[k] = params[k]
        return self._http.request("GET", self.endpoint("device_list"),
                                  headers=self._headers(), params=query)

    def _device_get(self, params: Dict[str, Any]):
        url = self.endpoint("device_get", iccid=self._iccid(params))
        return self._http.request("GET", url, headers=self._headers(),
                                  params=self._account_params(params))

    def _set_status(self, params: Dict[str, Any], state: str):
        url = self.endpoint("device_update", iccid=self._iccid(params))
        body: Dict[str, Any] = {"state": state}
        if params.get("service_plan"):
            body["ratePlan"] = params["service_plan"]
        return self._http.request(
            "PATCH", url,
            headers=self._headers({"Content-Type": "application/json"}),
            params=self._account_params(params), json=body)

    def _usage(self, params: Dict[str, Any]):
        url = self.endpoint("usage_get", iccid=self._iccid(params))
        query = self._account_params(params)
        for k in ("from", "to", "cycle"):
            if params.get(k):
                query[k] = params[k]
        return self._http.request("GET", url, headers=self._headers(), params=query)

    def _send_message(self, params: Dict[str, Any]):
        if not self.messaging_url:
            raise CarrierConfigError(
                "T-Mobile messaging is not configured (set TMOBILE_MESSAGING_URL "
                "or pass messaging_url=)")
        to = params.get("to")
        text = params.get("text")
        if not to or text is None:
            raise CarrierConfigError("send_message needs 'to' and 'text'")
        addresses = to if isinstance(to, list) else [to]
        body: Dict[str, Any] = {"to": addresses, "message": text}
        if params.get("from"):
            body["from"] = params["from"]
        return self._http.request(
            "POST", self.messaging_url,
            headers=self._headers({"Content-Type": "application/json"}),
            json=body)
