"""Tests for the carrier integration layer (tron.carriers).

No network: adapters' HTTP layer is monkeypatched to a recorder so we can
assert the exact request each carrier op produces, plus the cross-cutting
guarantees (consent gating, single shared rate budget, no secret leakage,
secret-free worker payloads).
"""
from __future__ import annotations

import pytest

from tron.carriers import (
    AdapterConfig,
    ATTAdapter,
    CarrierCredentials,
    CarrierGateway,
    CarrierOp,
    ConsentError,
    OpKind,
    RateLimiter,
    RateLimitExceeded,
    VerizonAdapter,
    run_op,
)
from tron.carriers.base import CarrierAPIError


class RecordingHttp:
    """Stand-in for HttpSession that records calls and returns canned data."""

    def __init__(self, responses=None):
        self.calls = []
        self._responses = list(responses or [])

    def request(self, method, url, *, headers=None, json=None, params=None,
                data=None, expect_json=True):
        self.calls.append({
            "method": method, "url": url, "headers": headers or {},
            "json": json, "params": params, "data": data,
        })
        if self._responses:
            return self._responses.pop(0)
        return ({"ok": True}, 200)


def verizon_with_recorder(**kw):
    creds = CarrierCredentials(client_id="id", client_secret="sec",
                               api_key="tok", account_name="ACME-0001")
    adapter = VerizonAdapter(creds, messaging_url=kw.pop("messaging_url", None))
    # Pretend we already hold valid tokens so we don't exercise the auth dance
    # unless a test wants to.
    adapter._access_token = "access"
    adapter._access_expiry = 10 ** 12
    adapter._session_token = "session"
    adapter._session_expiry = 10 ** 12
    rec = RecordingHttp(**kw)
    adapter._http = rec
    return adapter, rec


def att_with_recorder(**kw):
    creds = CarrierCredentials(username="user", api_key="key",
                               account_name="LIC-1")
    adapter = ATTAdapter(creds, messaging_url=kw.pop("messaging_url", None))
    rec = RecordingHttp(**kw)
    adapter._http = rec
    return adapter, rec


# --------------------------------------------------------------------------- #
# Consent gating
# --------------------------------------------------------------------------- #
def test_send_message_requires_consent():
    adapter, _ = att_with_recorder(messaging_url="https://msg.example/att")
    op = CarrierOp(kind=OpKind.SEND_MESSAGE, carrier="att",
                   params={"to": ["+15551234567"], "text": "hi"}, consent=False)
    with pytest.raises(ConsentError):
        adapter.execute(op)


def test_gateway_send_message_without_consent_raises():
    gw = CarrierGateway().register_att(messaging_url="https://msg.example/att")
    with pytest.raises(ConsentError):
        gw.send_message("att", to=["+15551234567"], text="hi")  # consent defaults False


def test_send_message_with_consent_goes_through():
    adapter, rec = att_with_recorder(messaging_url="https://msg.example/att")
    op = CarrierOp(kind=OpKind.SEND_MESSAGE, carrier="att",
                   params={"to": ["+15551234567"], "text": "hi"}, consent=True)
    res = adapter.execute(op)
    assert res.ok
    assert rec.calls[0]["url"] == "https://msg.example/att"
    assert rec.calls[0]["json"]["messageRequest"]["address"] == ["+15551234567"]


# --------------------------------------------------------------------------- #
# Rate limiter is a single shared budget
# --------------------------------------------------------------------------- #
def test_rate_limiter_single_budget_blocks_when_empty():
    rl = RateLimiter(rate_per_sec=1000, burst=2)
    rl.acquire(wait=False)
    rl.acquire(wait=False)
    with pytest.raises(RateLimitExceeded):
        rl.acquire(wait=False)


def test_rate_limiter_refills_over_time():
    rl = RateLimiter(rate_per_sec=1000, burst=1)
    rl.acquire(wait=False)
    # Waiting path should succeed once tokens refill.
    rl.acquire(wait=True, timeout=1.0)


# --------------------------------------------------------------------------- #
# Verizon request shaping
# --------------------------------------------------------------------------- #
def test_verizon_sim_status_builds_m2m_list_call():
    adapter, rec = verizon_with_recorder()
    res = adapter.execute(CarrierOp(kind=OpKind.SIM_STATUS, carrier="verizon",
                                    params={"iccid": "8914800012345"}))
    assert res.ok
    call = rec.calls[0]
    assert call["url"].endswith("/m2m/v1/devices/actions/list")
    assert call["json"]["accountName"] == "ACME-0001"
    assert call["json"]["deviceId"]["id"] == "8914800012345"
    assert call["headers"]["VZ-M2M-Token"] == "session"


def test_verizon_activate_uses_activate_endpoint():
    adapter, rec = verizon_with_recorder()
    adapter.execute(CarrierOp(kind=OpKind.SIM_ACTIVATE, carrier="verizon",
                              params={"iccid": "89148", "service_plan": "PLAN_A"}))
    call = rec.calls[0]
    assert call["url"].endswith("/m2m/v1/devices/actions/activate")
    assert call["json"]["servicePlan"] == "PLAN_A"


def test_verizon_without_messaging_url_does_not_support_send():
    adapter, _ = verizon_with_recorder()
    assert not adapter.supports(OpKind.SEND_MESSAGE)


# --------------------------------------------------------------------------- #
# AT&T request shaping
# --------------------------------------------------------------------------- #
def test_att_sim_status_gets_device_by_iccid():
    adapter, rec = att_with_recorder()
    res = adapter.execute(CarrierOp(kind=OpKind.SIM_STATUS, carrier="att",
                                    params={"iccid": "8901"}))
    assert res.ok
    call = rec.calls[0]
    assert call["method"] == "GET"
    assert call["url"].endswith("/rest/v1/devices/8901")
    assert call["params"]["licenseKey"] == "LIC-1"


def test_att_deactivate_puts_status():
    adapter, rec = att_with_recorder()
    adapter.execute(CarrierOp(kind=OpKind.SIM_DEACTIVATE, carrier="att",
                              params={"iccid": "8901"}))
    call = rec.calls[0]
    assert call["method"] == "PUT"
    assert call["json"]["status"] == "DEACTIVATED"


# --------------------------------------------------------------------------- #
# Errors surface as failed results, not exceptions
# --------------------------------------------------------------------------- #
def test_api_error_becomes_failed_result():
    def boom(*a, **k):
        raise CarrierAPIError("HTTP 503", status=503)
    adapter, rec = att_with_recorder()
    rec.request = boom  # type: ignore
    res = adapter.execute(CarrierOp(kind=OpKind.SIM_STATUS, carrier="att",
                                    params={"iccid": "8901"}))
    assert not res.ok
    assert res.status == 503


def test_unsupported_op_is_failed_result():
    adapter, _ = att_with_recorder()
    res = adapter.execute(CarrierOp(kind=OpKind.NETWORK_QUERY, carrier="att",
                                    params={}))
    assert not res.ok
    assert "does not support" in res.error


# --------------------------------------------------------------------------- #
# Secret hygiene + secret-free worker payload
# --------------------------------------------------------------------------- #
def test_credentials_never_leak_in_repr():
    creds = CarrierCredentials(client_id="id", client_secret="TOPSECRET",
                               api_key="ALSO_SECRET")
    text = f"{creds!r} {creds}"
    assert "TOPSECRET" not in text
    assert "ALSO_SECRET" not in text


def test_adapter_config_carries_no_secrets():
    cfg = AdapterConfig(carrier="verizon", base_url="https://x", messaging_url=None)
    blob = repr(cfg)
    assert "secret" not in blob.lower()
    # AdapterConfig has no credential fields at all.
    assert not any(f in cfg.__dict__ for f in
                   ("client_secret", "api_key", "password"))


def test_run_op_builds_adapter_and_resolves_env_creds(monkeypatch):
    # Worker resolves credentials from its own environment.
    monkeypatch.setenv("ATT_USERNAME", "wuser")
    monkeypatch.setenv("ATT_API_KEY", "wkey")
    monkeypatch.setenv("ATT_ACCOUNT_NAME", "WLIC")

    captured = {}

    class FakeHttp:
        def request(self, method, url, **kw):
            captured["url"] = url
            captured["params"] = kw.get("params")
            return ({"status": "ACTIVATED"}, 200)

    import tron.carriers.att as attmod
    monkeypatch.setattr(attmod, "HttpSession", lambda timeout=30.0: FakeHttp())

    cfg = AdapterConfig(carrier="att")
    res = run_op(CarrierOp(kind=OpKind.SIM_STATUS, carrier="att",
                           params={"iccid": "555"}), cfg)
    assert res.ok
    assert captured["url"].endswith("/rest/v1/devices/555")
    assert captured["params"]["licenseKey"] == "WLIC"


# --------------------------------------------------------------------------- #
# Gateway routing + local batch fan-out
# --------------------------------------------------------------------------- #
def test_gateway_routes_and_maps_locally(monkeypatch):
    gw = CarrierGateway(rate_per_sec=1000).register_verizon().register_att()

    # Patch run_op used inside the gateway to avoid real HTTP.
    import tron.carriers.gateway as gwmod

    def fake_run_op(op, config):
        from tron.carriers import CarrierResult
        return CarrierResult.success(op, {"routed_to": config.carrier})

    monkeypatch.setattr(gwmod, "run_op", fake_run_op)

    ops = [
        CarrierOp(kind=OpKind.SIM_STATUS, carrier="verizon", params={"iccid": "1"}),
        CarrierOp(kind=OpKind.SIM_STATUS, carrier="att", params={"iccid": "2"}),
    ]
    results = gw.map(ops, distributed=False)
    assert [r.data["routed_to"] for r in results] == ["verizon", "att"]


def test_gateway_unregistered_carrier_errors():
    gw = CarrierGateway()
    with pytest.raises(Exception):
        gw.manage_sim("verizon", "status", iccid="1")
