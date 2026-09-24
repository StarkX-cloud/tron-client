"""Command-line interface for carrier automation.

    python -m tron.carriers.cli carriers
    python -m tron.carriers.cli sim    --carrier verizon --action status --iccid 8914...
    python -m tron.carriers.cli query  --carrier att --what usage --iccid 8901...
    python -m tron.carriers.cli send   --carrier att --to +15551234567 --text "hi" --consent

Credentials come from the environment (see .env.example / docs/carriers.md).
Add --distributed to run the op across a running TRON server + workers instead
of locally. Sending a message requires --consent and only to recipients who
opted in.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import List, Optional

from .base import CarrierResult, ConsentError, OpKind
from .env import load_env
from .gateway import CarrierGateway
from .tasks import ADAPTER_CLASSES


def _register_all(gw: CarrierGateway, carriers: Optional[List[str]] = None) -> None:
    registrars = {
        "verizon": gw.register_verizon,
        "att": gw.register_att,
        "tmobile": gw.register_tmobile,
    }
    for name in (carriers or list(ADAPTER_CLASSES)):
        registrars[name]()


def _emit(result: CarrierResult) -> int:
    print(json.dumps(asdict(result), indent=2, default=str))
    return 0 if result.ok else 1


def _resolve(result_or_future):
    # Distributed ops return a MagicFuture; block for the result here.
    if isinstance(result_or_future, CarrierResult):
        return result_or_future
    return result_or_future.get()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tron-carriers",
        description="Automate tasks over official Verizon / AT&T / T-Mobile APIs.",
    )
    p.add_argument("--rate", type=float, default=5.0,
                   help="Shared rate-limit budget (requests/sec). Default 5.")
    p.add_argument("--env", metavar="PATH", nargs="?", const=".env", default=None,
                   help="Load credentials from a .env file (default path: .env)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("carriers", help="List carriers that have credentials configured")

    pf = sub.add_parser("preflight",
                        help="Read-only credential/connectivity check (no side effects)")
    pf.add_argument("--carrier", choices=sorted(ADAPTER_CLASSES),
                    help="Check one carrier (default: all)")

    def add_common(sp):
        sp.add_argument("--carrier", required=True,
                        choices=sorted(ADAPTER_CLASSES), help="Target carrier")
        sp.add_argument("--distributed", action="store_true",
                        help="Run across the TRON worker fabric instead of locally")
        sp.add_argument("--account-name", help="Override the account/billing id")
        sp.add_argument("--ref", help="Correlation id recorded in the spine log")

    sim = sub.add_parser("sim", help="Manage IoT SIMs / devices")
    add_common(sim)
    sim.add_argument("--action", required=True,
                     choices=["list", "status", "activate", "deactivate"])
    sim.add_argument("--iccid", help="SIM ICCID (required for status/activate/deactivate)")
    sim.add_argument("--service-plan", help="Rate/service plan for activate")

    q = sub.add_parser("query", help="Query usage / account / network data")
    add_common(q)
    q.add_argument("--what", required=True, choices=["usage", "account", "network"])
    q.add_argument("--iccid", help="SIM ICCID (for usage)")

    s = sub.add_parser("send", help="Send an opt-in message / notification")
    add_common(s)
    s.add_argument("--to", required=True, action="append",
                   help="Recipient (repeatable). Must have opted in.")
    s.add_argument("--text", required=True, help="Message body")
    s.add_argument("--from", dest="sender", help="Sender id / short code")
    s.add_argument("--consent", action="store_true",
                   help="Required: assert recipients opted in")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.env:
        load_env(args.env)
    gw = CarrierGateway(rate_per_sec=args.rate)

    if args.command == "carriers":
        _register_all(gw)
        print(json.dumps({"registered": gw.carriers()}, indent=2))
        return 0

    if args.command == "preflight":
        _register_all(gw, [args.carrier] if args.carrier else None)
        results = gw.preflight(args.carrier)
        report = [{"carrier": r.carrier, "ok": r.ok, "detail": r.detail,
                   "base_url": r.base_url, "status": r.status} for r in results]
        print(json.dumps({"preflight": report}, indent=2))
        return 0 if all(r.ok for r in results) else 1

    _register_all(gw, [args.carrier])
    common = {"distributed": args.distributed, "ref": args.ref}
    extra = {"account_name": args.account_name} if args.account_name else {}

    try:
        if args.command == "sim":
            if args.action in ("status", "activate", "deactivate") and not args.iccid:
                print(f"--iccid is required for --action {args.action}", file=sys.stderr)
                return 2
            params = dict(extra)
            if args.iccid:
                params["iccid"] = args.iccid
            if args.service_plan:
                params["service_plan"] = args.service_plan
            res = gw.manage_sim(args.carrier, args.action, **common, **params)

        elif args.command == "query":
            params = dict(extra)
            if args.iccid:
                params["iccid"] = args.iccid
            res = gw.query(args.carrier, args.what, **common, **params)

        elif args.command == "send":
            res = gw.send_message(args.carrier, to=args.to, text=args.text,
                                  sender=args.sender, consent=args.consent,
                                  **common, **extra)
        else:  # pragma: no cover - argparse guards this
            print("unknown command", file=sys.stderr)
            return 2
    except ConsentError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3

    return _emit(_resolve(res))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
