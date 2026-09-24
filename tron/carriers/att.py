"""AT&T adapter — official developer APIs.

Primary integration is **AT&T IoT / Control Center** — Cisco IoT Control Center
(formerly Jasper), the REST API AT&T provisions for connected-device fleets —
for SIM/device management, status and usage. The REST v1 API lives under
``/rws/api/v1`` (see developer.cisco.com/docs/control-center):

* ``GET  /rws/api/v1/devices``                       list devices (accountId, pageNumber, pageSize)
* ``GET  /rws/api/v1/devices/{iccid}``               device details
* ``PUT  /rws/api/v1/devices/{iccid}``               update status / ratePlan
* ``GET  /rws/api/v1/devices/{iccid}/ctdUsages``     cycle-to-date usage
* ``POST /rws/api/v1/devices/{iccid}/smsMessages``   mobile-terminated SMS to the SIM

Authentication is HTTP Basic using your Control Center ``username`` and
``api_key`` (the account's API key from Cisco DevNet). The base URL is
overridable so you can point at the exact ``rws`` host AT&T assigned to your
account (sandbox is ``https://rws-jpotest.jasper.com``).

Credentials (``credentials_from_env``) read, in order:

* ``ATT_USERNAME``       -> username     (Control Center API user)
* ``ATT_API_KEY``        -> api_key       (Control Center API key)
* ``ATT_ACCOUNT_NAME``   -> account_name  (accountId, when required)
* ``ATT_BASE_URL``       -> base_url override (per-tenant rws host)
* ``ATT_MESSAGING_URL``  -> A2P messaging gateway override (addressed by 'to')
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
from .normalize import extract_devices, normalize_device


# AT&T Control Center (Jasper) REST host. Override for your assigned tenant.
# AT&T Control Center is Cisco IoT Control Center (Jasper). The REST v1 API
# lives under /rws/api/v1 on a Jasper "rws" host. Verizon-style hostnames vary
# per tenant (sandbox is https://rws-jpotest.jasper.com); override per account.
DEFAULT_BASE = "https://rws.jasper.com"


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
            "device_list": "rws/api/v1/devices",
            "device_get": "rws/api/v1/devices/{iccid}",
            "device_update": "rws/api/v1/devices/{iccid}",
            "usage_get": "rws/api/v1/devices/{iccid}/ctdUsages",
            "device_sms": "rws/api/v1/devices/{iccid}/smsMessages",
        }

    def supported(self) -> List[OpKind]:
        # Control Center has a native mobile-terminated SMS-to-SIM endpoint, so
        # messaging is always supported (to a device you own by ICCID). A
        # messaging_url override additionally enables an A2P gateway.
        return [
            OpKind.SIM_LIST, OpKind.SIM_ACTIVATE, OpKind.SIM_DEACTIVATE,
            OpKind.SIM_STATUS, OpKind.USAGE_QUERY, OpKind.ACCOUNT_QUERY,
            OpKind.SEND_MESSAGE,
        ]

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
        return {"accountId": acct} if acct else {}

    def _iccid(self, params: Dict[str, Any]) -> str:
        iccid = params.get("iccid")
        if not iccid:
            raise CarrierConfigError("this AT&T operation needs an 'iccid'")
        return str(iccid)

    # -- preflight ---------------------------------------------------------- #
    def _preflight(self):
        # Single-record device list: authenticates without side effects.
        query = self._account_params({})
        query["pageSize"] = 1
        _data, status = self._http.request(
            "GET", self.endpoint("device_list"),
            headers=self._headers(), params=query)
        return ("device list (pageSize=1) OK", status)

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

    def paginate_devices(self, params: Dict[str, Any], *, max_pages: int = 1000):
        """Yield every device across pages (normalized). Stops on lastPage."""
        page = int(params.get("pageNumber", 1))
        for _ in range(max_pages):
            query = self._account_params(params)
            query["pageSize"] = int(params.get("pageSize", 100))
            query["pageNumber"] = page
            data, _status = self._http.request(
                "GET", self.endpoint("device_list"),
                headers=self._headers(), params=query)
            batch = extract_devices(data)
            for rec in batch:
                yield normalize_device(rec)
            last = isinstance(data, dict) and data.get("lastPage")
            if last or not batch:
                return
            page += 1

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
        text = params.get("text")
        if text is None:
            raise CarrierConfigError("send_message needs 'text'")

        # A2P gateway override (recipient MSISDNs).
        if self.messaging_url:
            to = params.get("to")
            if not to:
                raise CarrierConfigError("send_message needs 'to' for the A2P gateway")
            addresses = to if isinstance(to, list) else [to]
            body: Dict[str, Any] = {"messageRequest": {"address": addresses,
                                                        "message": text}}
            if params.get("from"):
                body["messageRequest"]["from"] = params["from"]
            return self._http.request(
                "POST", self.messaging_url,
                headers=self._headers({"Content-Type": "application/json"}),
                json=body)

        # Native Control Center mobile-terminated SMS to a SIM you own. The
        # recipient is the device ICCID, not a phone number.
        iccid = params.get("iccid") or params.get("to")
        if not iccid:
            raise CarrierConfigError(
                "Control Center SMS is sent to a device by 'iccid' (or set "
                "ATT_MESSAGING_URL for an A2P gateway addressed by 'to')")
        if isinstance(iccid, list):
            iccid = iccid[0]
        url = self.endpoint("device_sms", iccid=str(iccid))
        body = {"messageText": text}
        return self._http.request(
            "POST", url,
            headers=self._headers({"Content-Type": "application/json"}),
            params=self._account_params(params), json=body)
