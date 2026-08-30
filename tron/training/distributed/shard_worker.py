"""Run one distributed-training shard as its own OS process:

    python -m tron.training.distributed.shard_worker \
        --master http://127.0.0.1:9000 --session <id> --shard 0

Point `--master` at a different host and the same parameter vectors go
over the real internet instead of loopback — nothing else changes. This
is the "genuinely separate physical machines" path: N of these, one per
machine (or per process), against one queue_server.
"""
from __future__ import annotations

import argparse
import sys

from .shard_client import RequestsTransport, run_shard


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TRON distributed training shard worker")
    parser.add_argument("--master", default="http://127.0.0.1:9000", help="queue_server base URL")
    parser.add_argument("--session", required=True, help="training session id from POST /training/session")
    parser.add_argument("--shard", type=int, required=True, help="this worker's shard index")
    parser.add_argument("--auth-token", default=None, help="X-TRON-AUTH token, if the master requires one")
    parser.add_argument("--node-id", default=None, help="node id to attribute spine events to (default shard-N)")
    parser.add_argument("--poll-interval", type=float, default=0.25, help="barrier poll interval, seconds")
    args = parser.parse_args(argv)

    transport = RequestsTransport(args.master, auth_token=args.auth_token)
    print(f"[SHARD {args.shard}] joining session {args.session} on {args.master}")
    status = run_shard(
        transport, args.session, args.shard,
        poll_interval=args.poll_interval, node_id=args.node_id,
    )
    merged = status.get("rounds_merged", [])
    print(f"[SHARD {args.shard}] done — {len(merged)}/{status.get('num_rounds')} rounds merged, "
          f"{status.get('wire_bytes_transferred', 0):,} bytes over the wire (all shards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
