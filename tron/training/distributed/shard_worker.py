"""Run one distributed-training shard as its own OS process:

    python -m tron.training.distributed.shard_worker \
        --master http://127.0.0.1:9000 --session <id> --shard 0

Point `--master` at a different host and the same bytes go over the real
internet instead of loopback. This is the "genuinely separate physical
machines" path: N of these, one per machine (or per process), against one
queue_server.

The session's `model_kind` decides what moves over the wire: a numpy
parameter vector (`numpy_mlp`, the default) or a LoRA adapter state dict
(`lora`). For `lora` this process loads the frozen base model locally
(from the HF cache) — only the adapter is ever transmitted.
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
    meta = transport.get_json(f"/training/session/{args.session}")
    kind = meta.get("model_kind", "numpy_mlp")
    print(f"[SHARD {args.shard}] joining {kind} session {args.session} on {args.master}")

    if kind == "lora":
        from ..lora_demo import load_base_model_and_tokenizer
        from .lora_wire import run_lora_shard

        model_name = (meta.get("model_config") or {}).get("model_name", "EleutherAI/pythia-70m")
        print(f"[SHARD {args.shard}] loading base model {model_name} locally (only the adapter is sent)")
        base, _ = load_base_model_and_tokenizer(model_name)
        status = run_lora_shard(
            transport, args.session, args.shard,
            base_model_factory=lambda: base,
            poll_interval=args.poll_interval, node_id=args.node_id,
        )
    else:
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
