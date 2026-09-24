"""AT&T adapter — official developer APIs.

Primary integration is **AT&T IoT / Control Center** (the Cisco Jasper-based
REST API AT&T provisions for connected-device fleets) for SIM/device
management, status and usage. A configurable messaging endpoint supports AT&T
enterprise messaging where your account is provisioned for it.

Authentication is HTTP Basic using your Control Center ``username`` and
``api_key`` (the account's API credential), exactly as AT&T/Jasper documents.
The base URL is overridable so you can point at the exact Control Center host
AT&T assigned to your account.

Credentials (``credentials_from_env``) read, in order:

* ``ATT_USERNAME``       -> username     (Control Center API user)
* ``ATT_API_KEY``        -> api_key       (Control Center API key)
* ``ATT_ACCOUNT_NAME``   -> account_name  (billing / account id, when required)
* ``ATT_BASE_URL``       -> base_url override
* ``ATT_MESSAGING_URL``  -> messaging endpoint override
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional

from .base import (
    CarrierAdapter,
    CarrierConfigError,
    CarrierCredentials,
    CarrierOp,
    OpKind,
    _env,
)
from .http import HttpSession


# AT&T Control Center (Jasper) REST host. Override for your assigned tenant.
DEFAULT_BASE = "https://api.att-iot.com"


class ATTAdapter(CarrierAdapter):
    name = "att"

    def __init__(self, credentials: Optional[CarrierCredentials] = None,
                 *, base_url: Optional[str] = None, timeout: float = 30.0,
                 endpoints: Optional[Dict[str, str]] = None,
                 messaging_url: Optional[str] = None) -> None:
        super().__init__(credentials, base_url=base_url, timeout=timeout,
                         endpoints=endpoints)
        self.messaging_url = messaging_url or _env("ATT_MESSAGING_URL")
        self._http = HttpSession(timeout=self.timeout)

    # -- configuration ------------------------------------------------------ #
    @classmethod
    def credentials_from_env(cls) -> CarrierCredentials:
        return CarrierCredentials(
            username=_env("ATT_USERNAME"),
            api_key=_env("ATT_API_KEY"),
            account_name=_env("ATT_ACCOUNT_NAME"),
        )

    def default_base_url(self) -> str:
        return os.getenv("ATT_BASE_URL", DEFAULT_BASE)

    def default_endpoints(self) -> Dict[str, str]:
        return {
            "device_list": "rest/v1/devices",
            "device_get": "rest/v1/devices/{iccid}",
            "device_update": "rest/v1/devices/{iccid}",
            "usage_get": "rest/v1/devices/{iccid}/ctdUsages",
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
    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        self.credentials.require("username", "api_key")
        token = base64.b64encode(
            f"{self.credentials.username}:{self.credentials.api_key}".encode()
        ).decode()
        headers = {"Authorization": f"Basic {token}",
                   "Accept": "application/json"}
        if extra:
            headers.update(extra)
        return headers

    def _account_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        acct = params.get("account_name") or self.credentials.account_name
        return {"licenseKey": acct} if acct else {}

    def _iccid(self, params: Dict[str, Any]) -> str:
        iccid = params.get("iccid")
        if not iccid:
            raise CarrierConfigError("this AT&T operation needs an 'iccid'")
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
            return self._set_status(op.params, "ACTIVATED")
        if op.kind == OpKind.SIM_DEACTIVATE:
            return self._set_status(op.params, "DEACTIVATED")
        if op.kind == OpKind.USAGE_QUERY:
            return self._usage(op.params)
        raise CarrierConfigError(f"unhandled op {op.kind}")

    def _device_list(self, params: Dict[str, Any]):
        query = self._account_params(params)
        for k in ("pageSize", "pageNumber", "modifiedSince"):
            if params.get(k) is not None:
                query[k] = params[k]
        return self._http.request("GET", self.endpoint("device_list"),
                                  headers=self._headers(), params=query)

    def _device_get(self, params: Dict[str, Any]):
        url = self.endpoint("device_get", iccid=self._iccid(params))
        return self._http.request("GET", url, headers=self._headers(),
                                  params=self._account_params(params))

    def _set_status(self, params: Dict[str, Any], status: str):
        url = self.endpoint("device_update", iccid=self._iccid(params))
        body: Dict[str, Any] = {"status": status}
        if params.get("service_plan"):
            body["ratePlan"] = params["service_plan"]
        return self._http.request(
            "PUT", url,
            headers=self._headers({"Content-Type": "application/json"}),
            params=self._account_params(params), json=body)

    def _usage(self, params: Dict[str, Any]):
        url = self.endpoint("usage_get", iccid=self._iccid(params))
        query = self._account_params(params)
        for k_src, k_dst in (("cycle", "cycle"), ("from", "from"), ("to", "to")):
            if params.get(k_src):
                query[k_dst] = params[k_src]
        return self._http.request("GET", url, headers=self._headers(), params=query)

    def _send_message(self, params: Dict[str, Any]):
        if not self.messaging_url:
            raise CarrierConfigError(
                "AT&T messaging is not configured (set ATT_MESSAGING_URL "
                "or pass messaging_url=)")
        to = params.get("to")
        text = params.get("text")
        if not to or text is None:
            raise CarrierConfigError("send_message needs 'to' and 'text'")
        addresses = to if isinstance(to, list) else [to]
        body = {
            "messageRequest": {
                "address": addresses,
                "message": text,
            }
        }
        if params.get("from"):
            body["messageRequest"]["from"] = params["from"]
        return self._http.request(
            "POST", self.messaging_url,
            headers=self._headers({"Content-Type": "application/json"}),
            json=body)
