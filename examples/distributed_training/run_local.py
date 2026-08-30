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
        sess = requests.post(f"{base_url}/training/session", json={
            "num_shards": args.shards,
            "num_rounds": args.rounds,
            "local_steps": args.local_steps,
        }, timeout=10).json()
        session_id = sess["session_id"]
        print(f"[example] session {session_id}: {args.shards} shards x {args.rounds} rounds")

        shard_procs = []
        for k in range(args.shards):
            shard_procs.append(subprocess.Popen(
                [sys.executable, "-m", "tron.training.distributed.shard_worker",
                 "--master", base_url, "--session", session_id, "--shard", str(k)],
                cwd=str(REPO_ROOT), env=_os_environ(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            ))

        for k, p in enumerate(shard_procs):
            out, _ = p.communicate(timeout=180)
            tag = f"[shard {k} rc={p.returncode}]"
            for line in (out or "").splitlines():
                print(f"{tag} {line}")

        result = requests.get(f"{base_url}/training/session/{session_id}/result", timeout=10)
        if result.status_code != 200:
            print(f"[example] run did not finish (HTTP {result.status_code})")
            return 1
        body = result.json()
        wire = body["wire_bytes_transferred"]
        algo = body["algorithmic_comm_bytes"]
        print("\n" + "=" * 60)
        print(f"merged model held-out accuracy  : {body['accuracy']:.4f}")
        print(f"bytes actually sent over sockets: {wire:,}")
        print(f"  in-process metric counts only : {algo:,}  (the upload side)")
        print(f"final merged vector artifact    : {body['final_vector_hash'][:16]}...")
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
    return dict(os.environ)


if __name__ == "__main__":
    sys.exit(main())
