"""Verizon adapter — official developer APIs.

Primary integration is **Verizon ThingSpace / Connectivity Management** (the
Verizon IoT M2M platform) for SIM/device management, status and usage. A
configurable messaging endpoint supports Verizon's enterprise messaging where
your account is provisioned for it.

Authentication (ThingSpace) is two-legged, exactly as Verizon documents
(https://thingspace.verizon.com/documentation/apis/connectivity-management):

1. OAuth2 *client_credentials* against ``POST /api/ts/v1/oauth2/token`` using
   HTTP Basic auth with your ``client_id`` / ``client_secret`` -> a bearer
   ``access_token`` (valid ~1 hour).
2. A ThingSpace *session login* — ``POST /api/m2m/v1/session/login`` with the
   bearer token in the header and your UWS ``username`` / ``password`` in the
   body -> a ``sessionToken`` sent as the ``VZ-M2M-Token`` header on subsequent
   M2M calls (idles out after ~20 minutes).

Endpoints and the base URL are overridable so you can point the adapter at the
exact host/tenant Verizon provisioned for you. Defaults reflect Verizon's
public ThingSpace host.

Credentials (``credentials_from_env``) read, in order:

* ``VERIZON_CLIENT_ID``      -> client_id     (OAuth2 app key)
* ``VERIZON_CLIENT_SECRET``  -> client_secret (OAuth2 app secret)
* ``VERIZON_USERNAME``       -> username       (UWS account username)
* ``VERIZON_PASSWORD``       -> password       (UWS account password)
* ``VERIZON_ACCOUNT_NAME``   -> account_name   (M2M billing account, e.g. 0000123456-00001)
* ``VERIZON_BASE_URL``       -> base_url override
* ``VERIZON_MESSAGING_URL``  -> messaging endpoint override
"""

from __future__ import annotations

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
from .normalize import extract_devices, normalize_device


DEFAULT_BASE = "https://thingspace.verizon.com/api"


class VerizonAdapter(CarrierAdapter):
    name = "verizon"

    def __init__(self, credentials: Optional[CarrierCredentials] = None,
                 *, base_url: Optional[str] = None, timeout: float = 30.0,
                 endpoints: Optional[Dict[str, str]] = None,
                 messaging_url: Optional[str] = None) -> None:
        super().__init__(credentials, base_url=base_url, timeout=timeout,
                         endpoints=endpoints)
        self.messaging_url = messaging_url or _env("VERIZON_MESSAGING_URL")
        self._http = HttpSession(timeout=self.timeout)
        self._lock = threading.Lock()
        self._access_token: Optional[str] = None
        self._access_expiry: float = 0.0
        self._session_token: Optional[str] = None
        self._session_expiry: float = 0.0

    # -- configuration ------------------------------------------------------ #
    @classmethod
    def credentials_from_env(cls) -> CarrierCredentials:
        return CarrierCredentials(
            client_id=_env("VERIZON_CLIENT_ID"),
            client_secret=_env("VERIZON_CLIENT_SECRET"),
            # ThingSpace session login uses the UWS account username/password.
            username=_env("VERIZON_USERNAME"),
            password=_env("VERIZON_PASSWORD"),
            account_name=_env("VERIZON_ACCOUNT_NAME"),
        )

    def default_base_url(self) -> str:
        return os.getenv("VERIZON_BASE_URL", DEFAULT_BASE)

    def default_endpoints(self) -> Dict[str, str]:
        return {
            "oauth_token": "ts/v1/oauth2/token",
            "session_login": "m2m/v1/session/login",
            "device_list": "m2m/v1/devices/actions/list",
            "device_activate": "m2m/v1/devices/actions/activate",
            "device_deactivate": "m2m/v1/devices/actions/deactivate",
            "device_info": "m2m/v1/devices/actions/list",
            "usage_list": "m2m/v1/devices/usage/actions/list/aggregate",
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
            import base64
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
                raise CarrierAuthError("Verizon OAuth2 returned no access_token")
            self._access_token = data["access_token"]
            self._access_expiry = time.time() + float(data.get("expires_in", 3600))
            return self._access_token

    def _ensure_session_token(self) -> str:
        with self._lock:
            if self._session_token and time.time() < self._session_expiry - 30:
                return self._session_token
        access = self._ensure_access_token()
        with self._lock:
            # Per ThingSpace docs: POST the UWS username/password in the body
            # with the OAuth bearer token in the header; the response carries a
            # sessionToken used as VZ-M2M-Token on subsequent M2M calls.
            self.credentials.require("username", "password")
            data, _ = self._http.request(
                "POST", self.endpoint("session_login"),
                headers={"Authorization": f"Bearer {access}",
                         "Content-Type": "application/json"},
                json={"username": self.credentials.username,
                      "password": self.credentials.password},
            )
            token = data.get("sessionToken") if isinstance(data, dict) else None
            if not token:
                raise CarrierAuthError("Verizon session login returned no sessionToken")
            self._session_token = token
            # Verizon sessions idle out after ~20 minutes.
            self._session_expiry = time.time() + 1200
            return token

    def _m2m_headers(self) -> Dict[str, str]:
        access = self._ensure_access_token()
        session = self._ensure_session_token()
        return {
            "Authorization": f"Bearer {access}",
            "VZ-M2M-Token": session,
            "Content-Type": "application/json",
        }

    # -- preflight ---------------------------------------------------------- #
    def _preflight(self):
        # Exercises OAuth2 + ThingSpace session login only; no device I/O.
        self._ensure_access_token()
        self._ensure_session_token()
        return ("OAuth2 + session login OK", 200)

    # -- dispatch ----------------------------------------------------------- #
    def _dispatch(self, op: CarrierOp) -> tuple[Any, Optional[int]]:
        if op.kind == OpKind.SEND_MESSAGE:
            return self._send_message(op.params)
        if op.kind in (OpKind.SIM_LIST, OpKind.ACCOUNT_QUERY):
            return self._device_list(op.params)
        if op.kind == OpKind.SIM_STATUS:
            return self._device_status(op.params)
        if op.kind == OpKind.SIM_ACTIVATE:
            return self._device_state(op.params, "device_activate")
        if op.kind == OpKind.SIM_DEACTIVATE:
            return self._device_state(op.params, "device_deactivate")
        if op.kind == OpKind.USAGE_QUERY:
            return self._usage(op.params)
        raise CarrierConfigError(f"unhandled op {op.kind}")

    def _account(self, params: Dict[str, Any]) -> str:
        acct = params.get("account_name") or self.credentials.account_name
        if not acct:
            raise CarrierConfigError(
                "Verizon M2M calls need an account_name (param or VERIZON_ACCOUNT_NAME)")
        return acct

    def _device_ids(self, params: Dict[str, Any]) -> List[Dict[str, str]]:
        ids = params.get("device_ids")
        if ids:
            return [{"id": str(i), "kind": params.get("id_kind", "iccid")} for i in ids]
        iccid = params.get("iccid")
        if iccid:
            return [{"id": str(iccid), "kind": "iccid"}]
        return []

    def _device_list(self, params: Dict[str, Any]):
        body: Dict[str, Any] = {"accountName": self._account(params)}
        devices = self._device_ids(params)
        if devices:
            body["deviceId"] = devices[0] if len(devices) == 1 else devices
        return self._http.request("POST", self.endpoint("device_list"),
                                  headers=self._m2m_headers(), json=body)

    def paginate_devices(self, params: Dict[str, Any], *, max_pages: int = 1000):
        """Yield every device across pages (normalized). Uses ThingSpace's
        largestDeviceIdSeen cursor and stops when hasMoreData is false."""
        seen = int(params.get("largestDeviceIdSeen", 0))
        for _ in range(max_pages):
            body = {"accountName": self._account(params),
                    "largestDeviceIdSeen": seen}
            data, _status = self._http.request(
                "POST", self.endpoint("device_list"),
                headers=self._m2m_headers(), json=body)
            batch = extract_devices(data)
            for rec in batch:
                yield normalize_device(rec)
            more = isinstance(data, dict) and data.get("hasMoreData")
            if not more or not batch:
                return
            seen += len(batch)

    def _device_status(self, params: Dict[str, Any]):
        if not (params.get("iccid") or params.get("device_ids")):
            raise CarrierConfigError("SIM_STATUS needs 'iccid' or 'device_ids'")
        return self._device_list(params)

    def _device_state(self, params: Dict[str, Any], endpoint_key: str):
        devices = self._device_ids(params)
        if not devices:
            raise CarrierConfigError("device activate/deactivate needs 'iccid' or 'device_ids'")
        body: Dict[str, Any] = {
            "accountName": self._account(params),
            "devices": [{"deviceIds": [d]} for d in devices],
        }
        if params.get("service_plan"):
            body["servicePlan"] = params["service_plan"]
        return self._http.request("POST", self.endpoint(endpoint_key),
                                  headers=self._m2m_headers(), json=body)

    def _usage(self, params: Dict[str, Any]):
        body: Dict[str, Any] = {"accountName": self._account(params)}
        for k_src, k_dst in (("start", "startTime"), ("end", "endTime")):
            if params.get(k_src):
                body[k_dst] = params[k_src]
        devices = self._device_ids(params)
        if devices:
            body["deviceId"] = devices if len(devices) > 1 else devices[0]
        return self._http.request("POST", self.endpoint("usage_list"),
                                  headers=self._m2m_headers(), json=body)

    def _send_message(self, params: Dict[str, Any]):
        if not self.messaging_url:
            raise CarrierConfigError(
                "Verizon messaging is not configured (set VERIZON_MESSAGING_URL "
                "or pass messaging_url=)")
        to = params.get("to")
        text = params.get("text")
        if not to or text is None:
            raise CarrierConfigError("send_message needs 'to' and 'text'")
        access = self._ensure_access_token()
        body = {"to": to, "from": params.get("from"), "text": text}
        body = {k: v for k, v in body.items() if v is not None}
        return self._http.request(
            "POST", self.messaging_url,
            headers={"Authorization": f"Bearer {access}",
                     "Content-Type": "application/json"},
            json=body)
