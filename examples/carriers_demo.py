"""Distributed carrier automation over TRON — runnable demo.

Shows the shape of the API without hitting a real carrier. Set real
credentials/endpoints (see docs/carriers.md and .env.example) and drop
``dry_run`` to talk to Verizon ThingSpace / AT&T Control Center for real.

    python examples/carriers_demo.py
"""
from __future__ import annotations

import os

from tron.carriers import CarrierGateway, CarrierOp, OpKind
from tron.carriers.tasks import AdapterConfig


def build_gateway() -> CarrierGateway:
    # Credentials come from the environment on whichever machine runs the op
    # (locally here, or a TRON worker when distributed=True). See .env.example.
    return (
        CarrierGateway(rate_per_sec=5, burst=10)
        .register_verizon()  # VERIZON_CLIENT_ID / _SECRET / _API_KEY / _ACCOUNT_NAME
        .register_att()       # ATT_USERNAME / _API_KEY / _ACCOUNT_NAME
    )


def demo_local(gw: CarrierGateway) -> None:
    """A single IoT-SIM status query, run on this machine."""
    print("\n== SIM status (Verizon, local) ==")
    res = gw.manage_sim("verizon", "status", iccid="89148000001234567890")
    print("ok=", res.ok, "err=", res.error)


def demo_distributed_batch(gw: CarrierGateway) -> None:
    """Fan a batch of usage queries across the TRON worker fabric.

    Distribution scales *authorized* traffic and adds fault-tolerance; the
    gateway's single shared rate budget still caps the aggregate call rate.
    """
    print("\n== Usage queries (AT&T, distributed batch) ==")
    iccids = ["8901410000000000001", "8901410000000000002", "8901410000000000003"]
    ops = [CarrierOp(kind=OpKind.USAGE_QUERY, carrier="att",
                     params={"iccid": i}, ref=f"usage-{i}") for i in iccids]
    results = gw.map(ops, distributed=True)  # requires a running TRON server/workers
    for r in results:
        print(f"  {r.ref}: ok={r.ok} err={r.error}")


def demo_send_opt_in(gw: CarrierGateway) -> None:
    """Send a notification — only with an explicit consent marker."""
    print("\n== Notification (AT&T, consent-gated) ==")
    if not os.getenv("ATT_MESSAGING_URL"):
        print("  (skipped: set ATT_MESSAGING_URL and register_att(messaging_url=...))")
        return
    res = gw.send_message(
        "att",
        to=["+15551234567"],           # recipient who opted in
        text="Your device is back online.",
        consent=True,                   # required — refuses to send without it
        distributed=True,
    ).get()
    print("  ok=", res.ok, "err=", res.error)


if __name__ == "__main__":
    gw = build_gateway()
    print("Registered carriers:", gw.carriers())
    # These will make real API calls if credentials/endpoints are configured.
    try:
        demo_local(gw)
    except Exception as exc:  # network/creds not set up in a bare checkout
        print("  local demo needs real credentials:", exc)
    print("\nConfigure credentials (docs/carriers.md) to run the distributed"
          " and messaging demos against real carrier APIs.")
