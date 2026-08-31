"""Run a distributed local-SGD training run across N *separate OS
processes* against one TRON master, all on this machine — the same code
path that runs across separate physical hosts, just pointed at loopback.

    python -m examples.distributed_training.run_local --shards 3 --rounds 4

What it does:
  1. starts queue_server.py on a free port (unless --master is given),
  2. POSTs /training/session to define the run,
  3. spawns one `tron.training.distributed.shard_worker` subprocess per
     shard — each a real process that fetches its data, trains locally,
     and POSTs its parameter vector back over HTTP,
  4. polls /training/session/<id>/result and prints the merged model's
     held-out accuracy plus the bytes that actually crossed a socket.

To run against genuinely separate machines: start queue_server.py on one
host, then on each other host run
    python -m tron.training.distributed.shard_worker \
        --master http://<that-host>:<port> --session <id> --shard <k>
Nothing else changes — the vectors just travel over a real network.
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"server at {base_url} never became healthy")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--master", default=None, help="use an already-running master instead of starting one")
    parser.add_argument(
        "--lora", action="store_true",
        help="send LoRA adapters (Pythia-70M) instead of a numpy vector — needs "
             "torch/transformers/peft; each shard loads the base model locally, "
             "only the ~400KB adapter crosses the socket",
    )
    args = parser.parse_args(argv)

    server_proc = None
    if args.master:
        base_url = args.master.rstrip("/")
    else:
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        env = {"PORT": str(port), "TRON_HOST": "127.0.0.1"}
        print(f"[example] starting queue_server on {base_url}")
        server_proc = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "queue_server.py")],
            cwd=str(REPO_ROOT), env={**_os_environ(), **env},
        )
        _wait_for_health(base_url)

    try:
        session_req = {
            "num_shards": args.shards,
            "num_rounds": args.rounds,
            "local_steps": args.local_steps,
        }
        if args.lora:
            session_req["model_kind"] = "lora"
        sess = requests.post(f"{base_url}/training/session", json=session_req, timeout=120).json()
        session_id = sess["session_id"]
        kind = "LoRA adapters" if args.lora else "numpy vectors"
        print(f"[example] session {session_id}: {args.shards} shards x {args.rounds} rounds, sending {kind}")

        shard_timeout = 900 if args.lora else 180  # LoRA shards load Pythia first
        shard_procs = []
        for k in range(args.shards):
            shard_procs.append(subprocess.Popen(
                [sys.executable, "-m", "tron.training.distributed.shard_worker",
                 "--master", base_url, "--session", session_id, "--shard", str(k)],
                cwd=str(REPO_ROOT), env=_os_environ(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            ))
            if args.lora:
                time.sleep(3)  # stagger the base-model loads a little

        for k, p in enumerate(shard_procs):
            out, _ = p.communicate(timeout=shard_timeout)
            tag = f"[shard {k} rc={p.returncode}]"
            for line in (out or "").splitlines():
                print(f"{tag} {line}")

        result = requests.get(f"{base_url}/training/session/{session_id}/result", timeout=120)
        if result.status_code != 200:
            print(f"[example] run did not finish (HTTP {result.status_code})")
            return 1
        body = result.json()
        wire = body["wire_bytes_transferred"]
        print("\n" + "=" * 60)
        if args.lora:
            print(f"merged adapter eval loss       : {body['eval_loss_before']:.4f} -> {body['eval_loss_after']:.4f}")
            print(f"adapter vs full model          : {body['adapter_bytes']:,} / {body['full_model_bytes']:,} B  "
                  f"({body['size_ratio']:.0f}x)")
            print(f"final merged adapter artifact  : {body['final_adapter_hash'][:16]}...")
        else:
            print(f"merged model held-out accuracy  : {body['accuracy']:.4f}")
            print(f"  in-process metric counts only : {body['algorithmic_comm_bytes']:,}  (the upload side)")
            print(f"final merged vector artifact    : {body['final_vector_hash'][:16]}...")
        print(f"bytes actually sent over sockets: {wire:,}")
        print("=" * 60)
        print(f"replay this run in the Grid: {base_url}/grid/  (events tagged session={session_id})")
        return 0
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


def _os_environ() -> dict:
    import os
    env = dict(os.environ)
    # Several torch/transformers processes each loading a model on one
    # Windows box otherwise hit a native crash from duplicate OpenMP/MKL
    # runtimes (exit code 0xC0000005). Single-threaded BLAS per process +
    # the duplicate-lib escape hatch keeps them from stepping on each
    # other; this is a local-orchestration detail, not something a real
    # multi-host deployment needs.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # the actual crash guard
    # a few threads each, not all cores, so N concurrent shard processes
    # don't oversubscribe the box (and don't run dead slow single-threaded)
    env.setdefault("OMP_NUM_THREADS", "3")
    env.setdefault("MKL_NUM_THREADS", "3")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


if __name__ == "__main__":
    sys.exit(main())
