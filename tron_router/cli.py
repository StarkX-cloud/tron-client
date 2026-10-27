"""CLI for the routing layer.

    python -m tron_router.cli decide --model EleutherAI/pythia-70m
    python -m tron_router.cli decide --model EleutherAI/pythia-70m --multi-node --master-url http://localhost:9000
    python -m tron_router.cli pool-status --master-url http://localhost:9000
"""

from __future__ import annotations

import argparse
import json

from tron_router.decision import JobRequest, decide_placement, query_pool_status


def cmd_decide(args) -> int:
    job = JobRequest(
        model_name=args.model,
        tolerant=not args.intolerant,
        requires_multi_node=args.multi_node,
    )
    decision = decide_placement(job, master_url=args.master_url)
    print(json.dumps(decision.to_dict(), indent=2))
    if decision.confidence == "low":
        print("\nNOTE: confidence is LOW -- read the warnings above before trusting this.")
    return 0


def cmd_pool_status(args) -> int:
    status = query_pool_status(args.master_url)
    print(json.dumps(status.to_dict(), indent=2))
    return 0 if status.reachable else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tron_router")
    sub = parser.add_subparsers(dest="command", required=True)

    decide_p = sub.add_parser(
        "decide", help="decide local / pooled / dedicated for a job"
    )
    decide_p.add_argument("--model", required=True)
    decide_p.add_argument(
        "--intolerant", action="store_true",
        help="job cannot tolerate a pooled worker dropping out (never routes to the pool)",
    )
    decide_p.add_argument(
        "--multi-node", action="store_true",
        help="job must be split across more than one machine regardless of local fit",
    )
    decide_p.add_argument("--master-url", default=None)
    decide_p.set_defaults(func=cmd_decide)

    pool_p = sub.add_parser("pool-status", help="query a live master's worker pool for real")
    pool_p.add_argument("--master-url", default=None)
    pool_p.set_defaults(func=cmd_pool_status)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
