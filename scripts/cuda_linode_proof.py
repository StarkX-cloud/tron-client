#!/usr/bin/env python3
"""Provision a short-lived Linode GPU, run Autopilot --real-model CUDA proof, destroy.

Requires env:
  LINODE_TOKEN   — Linode personal access token (Runtime Secret)

Optional:
  LINODE_TYPE    — default g2-gpu-rtx4000a1-s (~$0.52/hr, cheapest GPU tier)
  LINODE_REGION  — default us-ord (or first region that accepts the type)
  LINODE_IMAGE   — default linode/ubuntu22.04
  PROOF_MODEL    — default EleutherAI/pythia-410m
  PROOF_TRIALS   — default 1
  KEEP_LINODE    — if "1", skip destroy (DON'T for billing safety)

Always attempts destroy in a finally block unless KEEP_LINODE=1.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

API = "https://api.linode.com/v4"
TOKEN = os.environ.get("LINODE_TOKEN", "").strip()
TYPE = os.environ.get("LINODE_TYPE", "g2-gpu-rtx4000a1-s")
REGION = os.environ.get("LINODE_REGION", "us-ord")
IMAGE = os.environ.get("LINODE_IMAGE", "linode/ubuntu22.04")
MODEL = os.environ.get("PROOF_MODEL", "EleutherAI/pythia-410m")
TRIALS = int(os.environ.get("PROOF_TRIALS", "1"))
KEEP = os.environ.get("KEEP_LINODE", "") == "1"
REPO = os.environ.get(
    "PROOF_REPO", "https://github.com/StarkX-cloud/tron-client.git"
)
BRANCH = os.environ.get("PROOF_BRANCH", "cursor/economic-autopilot-38d1")
LABEL = f"tron-cuda-proof-{int(time.time())}"


def die(msg: str, code: int = 1) -> None:
    print(f"[cuda-proof] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def api(method: str, path: str, **kwargs):
    if not TOKEN:
        die("LINODE_TOKEN is not set — add it as a Cloud Agent Runtime Secret and start a NEW agent")
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }
    r = requests.request(method, f"{API}{path}", headers=headers, timeout=60, **kwargs)
    if r.status_code >= 400:
        die(f"{method} {path} -> {r.status_code}: {r.text[:800]}")
    if r.status_code == 204 or not r.content:
        return None
    return r.json()


def create_ssh_key(tmpdir: Path) -> tuple[str, Path]:
    key = tmpdir / "linode_proof_key"
    subprocess.check_call(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key), "-q"]
    )
    pub = key.with_suffix(".pub").read_text().strip()
    return pub, key


def wait_linode(linode_id: int, want: str = "running", timeout: int = 900) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = api("GET", f"/linode/instances/{linode_id}")
        status = data.get("status")
        print(f"[cuda-proof] linode {linode_id} status={status}")
        if status == want:
            return data
        if status in ("stopped",) and want == "running":
            # try boot
            try:
                api("POST", f"/linode/instances/{linode_id}/boot")
            except SystemExit:
                pass
        time.sleep(10)
    die(f"timeout waiting for linode {linode_id} -> {want}")


def destroy_linode(linode_id: int | None) -> None:
    if linode_id is None:
        return
    print(f"[cuda-proof] DESTROYING linode {linode_id} to stop billing…")
    try:
        api("DELETE", f"/linode/instances/{linode_id}")
        print(f"[cuda-proof] destroy requested for {linode_id}")
    except SystemExit as exc:
        print(f"[cuda-proof] destroy failed: {exc}", file=sys.stderr)


def ssh(key: Path, host: str, command: str, timeout: int = 7200) -> None:
    cmd = [
        "ssh",
        "-i",
        str(key),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=20",
        f"root@{host}",
        command,
    ]
    print(f"[cuda-proof] ssh root@{host}: {command[:120]}…")
    subprocess.check_call(cmd, timeout=timeout)


def scp_from(key: Path, host: str, remote: str, local: Path) -> None:
    cmd = [
        "scp",
        "-i",
        str(key),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"root@{host}:{remote}",
        str(local),
    ]
    subprocess.check_call(cmd, timeout=120)


def main() -> int:
    print("[cuda-proof] starting (will destroy Linode in finally unless KEEP_LINODE=1)")
    print(f"[cuda-proof] type={TYPE} region={REGION} model={MODEL} trials={TRIALS}")

    linode_id = None
    tmp = Path(tempfile.mkdtemp(prefix="tron-cuda-"))
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    try:
        pub, key = create_ssh_key(tmp)
        # root_pass required by API even with authorized_keys
        import secrets
        import string

        root_pass = "Tr0n!" + "".join(
            secrets.choice(string.ascii_letters + string.digits) for _ in range(24)
        )

        body = {
            "type": TYPE,
            "region": REGION,
            "image": IMAGE,
            "label": LABEL,
            "root_pass": root_pass,
            "authorized_keys": [pub],
            "booted": True,
            "tags": ["tron-cuda-proof", "auto-destroy"],
        }
        print(f"[cuda-proof] creating {TYPE} in {REGION}…")
        try:
            created = api("POST", "/linode/instances", json=body)
        except SystemExit as first:
            # region may lack the type — try a few GPU-friendly regions
            print(f"[cuda-proof] create failed in {REGION}: {first}")
            for alt in ("us-sea", "us-mia", "eu-west", "ap-south", "us-east"):
                if alt == REGION:
                    continue
                body["region"] = alt
                print(f"[cuda-proof] retrying region {alt}…")
                try:
                    created = api("POST", "/linode/instances", json=body)
                    break
                except SystemExit as e:
                    print(f"[cuda-proof] {alt} failed: {e}")
            else:
                die("could not create GPU Linode in any tried region")

        linode_id = created["id"]
        print(f"[cuda-proof] created id={linode_id} label={created.get('label')}")
        info = wait_linode(linode_id, "running")
        ips = info.get("ipv4") or []
        if not ips:
            die("no ipv4 on instance")
        host = ips[0]
        print(f"[cuda-proof] host={host}")

        # wait for ssh
        for i in range(60):
            try:
                ssh(key, host, "echo ssh-ok")
                break
            except Exception:
                time.sleep(5)
        else:
            die("ssh never became ready")

        setup = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git python3 python3-pip python3-venv curl
# NVIDIA driver / CUDA toolkit for Ubuntu — Linode GPU images often need this
if ! command -v nvidia-smi >/dev/null 2>&1; then
  apt-get install -y ubuntu-drivers-common || true
  ubuntu-drivers autoinstall || true
fi
nvidia-smi || true
cd /opt
rm -rf tron-client
git clone --branch BRANCH_PLACEHOLDER REPO_PLACEHOLDER tron-client
cd tron-client
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -r requirements-dev.txt -r requirements-training.txt
python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
""".replace(
            "BRANCH_PLACEHOLDER", BRANCH
        ).replace(
            "REPO_PLACEHOLDER", REPO
        )

        ssh(key, host, setup, timeout=3600)

        # If driver install required reboot, reboot and wait
        try:
            ssh(key, host, "nvidia-smi")
        except Exception:
            print("[cuda-proof] nvidia-smi failed — rebooting once for drivers…")
            try:
                ssh(key, host, "reboot || true")
            except Exception:
                pass
            time.sleep(45)
            for i in range(40):
                try:
                    ssh(key, host, "nvidia-smi")
                    break
                except Exception:
                    time.sleep(5)
            else:
                die("nvidia-smi still unavailable after reboot")

        out_json = "/tmp/economic_autopilot_cuda_benchmark.json"
        bench = f"""
set -euo pipefail
cd /opt/tron-client
. .venv/bin/activate
python -m benchmarks.economic_autopilot --real-model --trials {TRIALS} \
  --model {MODEL} --max-new-tokens 16 \
  --hourly-rate 0.52 --out {out_json}
python - <<'PY'
import torch, json
print(json.dumps({{
  "cuda_available": torch.cuda.is_available(),
  "device_count": torch.cuda.device_count(),
  "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
}}))
PY
"""
        ssh(key, host, bench, timeout=7200)
        local_json = artifacts / "economic_autopilot_cuda_benchmark.json"
        scp_from(key, host, out_json, local_json)
        try:
            scp_from(
                key,
                host,
                out_json.replace(".json", ".md"),
                artifacts / "economic_autopilot_cuda_benchmark.md",
            )
        except Exception:
            pass

        meta = {
            "linode_id": linode_id,
            "type": TYPE,
            "region": info.get("region"),
            "host": host,
            "model": MODEL,
            "trials": TRIALS,
            "destroyed": not KEEP,
            "artifact": str(local_json),
        }
        (artifacts / "cuda_linode_proof_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print("[cuda-proof] SUCCESS — results in artifacts/")
        print(local_json.read_text()[:1500])
        return 0
    finally:
        if KEEP:
            print(f"[cuda-proof] KEEP_LINODE=1 — leaving {linode_id} running (YOU pay)")
        else:
            destroy_linode(linode_id)


if __name__ == "__main__":
    raise SystemExit(main())
