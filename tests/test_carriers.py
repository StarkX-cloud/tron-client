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
    assert call["url"].endswith("/rws/api/v1/devices/8901")
    assert call["params"]["accountId"] == "LIC-1"


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
    assert captured["url"].endswith("/rws/api/v1/devices/555")
    assert captured["params"]["accountId"] == "WLIC"


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


# --------------------------------------------------------------------------- #
# T-Mobile adapter
# --------------------------------------------------------------------------- #
def tmobile_with_recorder(**kw):
    from tron.carriers import TMobileAdapter
    creds = CarrierCredentials(client_id="id", client_secret="sec",
                               account_name="ACC-9")
    adapter = TMobileAdapter(creds, messaging_url=kw.pop("messaging_url", None))
    adapter._access_token = "tok"
    adapter._access_expiry = 10 ** 12
    rec = RecordingHttp(**kw)
    adapter._http = rec
    return adapter, rec


def test_tmobile_registered_in_registry():
    from tron.carriers.tasks import ADAPTER_CLASSES
    assert "tmobile" in ADAPTER_CLASSES


def test_tmobile_status_and_deactivate():
    adapter, rec = tmobile_with_recorder()
    adapter.execute(CarrierOp(kind=OpKind.SIM_STATUS, carrier="tmobile",
                              params={"iccid": "8901"}))
    assert rec.calls[0]["method"] == "GET"
    assert rec.calls[0]["url"].endswith("/iot/v1/devices/8901")

    adapter.execute(CarrierOp(kind=OpKind.SIM_DEACTIVATE, carrier="tmobile",
                              params={"iccid": "8901"}))
    assert rec.calls[1]["method"] == "PATCH"
    assert rec.calls[1]["json"]["state"] == "inactive"


def test_gateway_registers_three_carriers():
    gw = CarrierGateway().register_verizon().register_att().register_tmobile()
    assert gw.carriers() == ["att", "tmobile", "verizon"]


# --------------------------------------------------------------------------- #
# Spine integration
# --------------------------------------------------------------------------- #
class FakeLog:
    def __init__(self):
        self.events = []

    def append(self, task_id, type, data=None, node_id=None, timestamp=None):
        self.events.append({"task_id": task_id, "type": type, "data": data or {},
                            "node_id": node_id})
        return self.events[-1]


def test_spine_logs_submit_and_complete_locally(monkeypatch):
    import tron.carriers.gateway as gwmod
    from tron.carriers import CarrierResult

    monkeypatch.setattr(gwmod, "run_op",
                        lambda op, cfg: CarrierResult.success(op, {"ok": 1}))

    log = FakeLog()
    gw = CarrierGateway(rate_per_sec=1000, event_log=log,
                        node_id="node-A").register_att()
    gw.manage_sim("att", "status", iccid="8901")

    types = [e["type"] for e in log.events]
    assert types == ["carrier_op_submitted", "carrier_op_completed"]
    # Same task_id for the pair -> submit->complete is one lineage.
    assert log.events[0]["task_id"] == log.events[1]["task_id"]
    assert all(e["node_id"] == "node-A" for e in log.events)


def test_spine_redacts_sensitive_targets(monkeypatch):
    import tron.carriers.gateway as gwmod
    from tron.carriers import CarrierResult
    monkeypatch.setattr(gwmod, "run_op",
                        lambda op, cfg: CarrierResult.success(op, None))

    log = FakeLog()
    gw = CarrierGateway(rate_per_sec=1000, event_log=log).register_att(
        messaging_url="https://msg.example/att")
    gw.send_message("att", to=["+15551234567"], text="hello there",
                    consent=True)

    submitted = log.events[0]["data"]["params"]
    # Phone number never stored in the clear; text stored only as a length.
    assert submitted["to"] == ["sha256:" + submitted["to"][0].split(":")[1]]
    assert "+15551234567" not in json_dumps(log.events)
    assert submitted["text_len"] == len("hello there")
    assert "text" not in submitted


def json_dumps(obj):
    import json
    return json.dumps(obj, default=str)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_carriers_lists(capsys):
    from tron.carriers.cli import main
    rc = main(["carriers"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "verizon" in out and "att" in out and "tmobile" in out


def test_cli_sim_status(monkeypatch, capsys):
    import tron.carriers.gateway as gwmod
    from tron.carriers import CarrierResult
    monkeypatch.setattr(gwmod, "run_op",
                        lambda op, cfg: CarrierResult.success(op, {"state": "ACTIVE"},
                                                              status=200))
    from tron.carriers.cli import main
    rc = main(["sim", "--carrier", "att", "--action", "status", "--iccid", "8901"])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"ok": true' in out
    assert "ACTIVE" in out


def test_cli_send_requires_consent(monkeypatch, capsys):
    from tron.carriers.cli import main
    # No --consent -> refused with exit code 3, before any network.
    rc = main(["send", "--carrier", "att", "--to", "+15551234567", "--text", "hi"])
    assert rc == 3
    assert "refused" in capsys.readouterr().err


def test_cli_sim_status_missing_iccid():
    from tron.carriers.cli import main
    rc = main(["sim", "--carrier", "verizon", "--action", "status"])
    assert rc == 2


# --------------------------------------------------------------------------- #
# Verizon real auth flow (username/password session login)
# --------------------------------------------------------------------------- #
def test_verizon_session_login_posts_username_password():
    creds = CarrierCredentials(client_id="cid", client_secret="csec",
                               username="uws-user", password="uws-pass",
                               account_name="0000123456-00001")
    adapter = VerizonAdapter(creds)
    rec = RecordingHttp(responses=[
        ({"access_token": "acc", "expires_in": 3600}, 200),  # oauth
        ({"sessionToken": "sess"}, 200),                      # session login
    ])
    adapter._http = rec
    token = adapter._ensure_session_token()
    assert token == "sess"
    login_call = rec.calls[1]
    assert login_call["url"].endswith("/m2m/v1/session/login")
    assert login_call["json"] == {"username": "uws-user", "password": "uws-pass"}
    assert login_call["headers"]["Authorization"] == "Bearer acc"


# --------------------------------------------------------------------------- #
# Preflight (read-only, no side effects)
# --------------------------------------------------------------------------- #
def test_preflight_ok_for_tmobile():
    from tron.carriers import TMobileAdapter
    creds = CarrierCredentials(client_id="id", client_secret="sec")
    adapter = TMobileAdapter(creds)
    adapter._http = RecordingHttp(responses=[
        ({"access_token": "t", "expires_in": 3600}, 200),
    ])
    res = adapter.preflight()
    assert res.ok
    assert res.carrier == "tmobile"


def test_preflight_fails_without_credentials():
    from tron.carriers import ATTAdapter
    adapter = ATTAdapter(CarrierCredentials())  # nothing set
    res = adapter.preflight()
    assert not res.ok
    assert "credential" in res.detail.lower()


def test_att_preflight_uses_single_record_list():
    adapter, rec = att_with_recorder()
    res = adapter.preflight()
    assert res.ok
    call = rec.calls[0]
    assert call["method"] == "GET"
    assert call["params"]["pageSize"] == 1  # read-only, minimal


def test_gateway_preflight_all_carriers():
    gw = CarrierGateway().register_att()
    results = gw.preflight()
    assert len(results) == 1 and results[0].carrier == "att"


# --------------------------------------------------------------------------- #
# .env loading
# --------------------------------------------------------------------------- #
def test_load_env_reads_file_without_override(tmp_path, monkeypatch):
    from tron.carriers import load_env
    envfile = tmp_path / ".env"
    envfile.write_text(
        "# comment\n"
        "VERIZON_CLIENT_ID=abc123\n"
        'export ATT_API_KEY="quoted-key"\n'
        "\n"
        "TMOBILE_CLIENT_ID=zzz\n"
    )
    monkeypatch.delenv("VERIZON_CLIENT_ID", raising=False)
    monkeypatch.setenv("TMOBILE_CLIENT_ID", "already-set")
    applied = load_env(str(envfile))
    import os
    assert os.environ["VERIZON_CLIENT_ID"] == "abc123"
    assert os.environ["ATT_API_KEY"] == "quoted-key"
    # existing env wins (no override)
    assert os.environ["TMOBILE_CLIENT_ID"] == "already-set"
    assert "TMOBILE_CLIENT_ID" not in applied


def test_load_env_missing_file_is_noop(tmp_path):
    from tron.carriers import load_env
    assert load_env(str(tmp_path / "nope.env")) == {}


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def test_normalize_device_maps_aliases():
    from tron.carriers import normalize_device
    v = normalize_device({"iccid": "89", "state": "ACTIVATED", "servicePlan": "P"})
    assert v["iccid"] == "89"
    assert v["status"] == "ACTIVATED"
    assert v["rate_plan"] == "P"
    assert v["_raw"]["servicePlan"] == "P"


def test_extract_devices_handles_envelopes():
    from tron.carriers import extract_devices
    assert len(extract_devices({"devices": [{"iccid": "1"}, {"iccid": "2"}]})) == 2
    assert len(extract_devices([{"iccid": "1"}])) == 1
    assert len(extract_devices({"iccid": "solo", "status": "X"})) == 1
    assert extract_devices(None) == []


# --------------------------------------------------------------------------- #
# Pagination
# --------------------------------------------------------------------------- #
def test_att_paginate_devices_follows_pages():
    adapter, rec = att_with_recorder(responses=[
        ({"devices": [{"iccid": "1"}, {"iccid": "2"}], "lastPage": False}, 200),
        ({"devices": [{"iccid": "3"}], "lastPage": True}, 200),
    ])
    devices = list(adapter.paginate_devices({"pageSize": 2}))
    assert [d["iccid"] for d in devices] == ["1", "2", "3"]
    # Second request advanced the page number.
    assert rec.calls[1]["params"]["pageNumber"] == 2


def test_verizon_paginate_devices_uses_cursor():
    adapter, rec = verizon_with_recorder(responses=[
        ({"devices": [{"iccid": "a"}], "hasMoreData": True}, 200),
        ({"devices": [{"iccid": "b"}], "hasMoreData": False}, 200),
    ])
    devices = list(adapter.paginate_devices({"account_name": "ACME-0001"}))
    assert [d["iccid"] for d in devices] == ["a", "b"]
    assert rec.calls[1]["json"]["largestDeviceIdSeen"] == 1


def test_gateway_list_all_sims(monkeypatch):
    gw = CarrierGateway(rate_per_sec=1000).register_att()

    def fake_build():
        class A:
            def paginate_devices(self, params):
                yield {"iccid": "x"}
                yield {"iccid": "y"}
        return A()
    monkeypatch.setattr(gw._configs["att"], "build", fake_build)
    out = gw.list_all_sims("att")
    assert [d["iccid"] for d in out] == ["x", "y"]


# --------------------------------------------------------------------------- #
# AT&T native Control Center SMS (mobile-terminated to a SIM)
# --------------------------------------------------------------------------- #
def test_att_native_sms_to_iccid():
    adapter, rec = att_with_recorder()  # no messaging_url -> native path
    op = CarrierOp(kind=OpKind.SEND_MESSAGE, carrier="att",
                   params={"iccid": "8901", "text": "ping"}, consent=True)
    res = adapter.execute(op)
    assert res.ok
    call = rec.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/rws/api/v1/devices/8901/smsMessages")
    assert call["json"]["messageText"] == "ping"


def test_att_native_sms_requires_iccid():
    adapter, _ = att_with_recorder()
    op = CarrierOp(kind=OpKind.SEND_MESSAGE, carrier="att",
                   params={"text": "ping"}, consent=True)
    res = adapter.execute(op)  # missing iccid -> failed result
    assert not res.ok
    assert "iccid" in res.error.lower()
