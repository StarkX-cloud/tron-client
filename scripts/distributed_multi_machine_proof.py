#!/usr/bin/env python3
"""Multi-machine Distributed Execution Engine v1 proof on Linode CPU Nanodes.

Provisions TWO independent Nanodes:
  - node0: queue_server (master) + worker dist-w0
  - node1: worker dist-w1 → node0 public IP

Runs Autopilot distributed map (sum_squares), verifies closed-form total,
records wall time + cost, then DESTROYS both Linodes in finally.

Requires:
  LINODE_TOKEN  — Runtime Secret (Linodes Read/Write). CPU Nanodes work even
                  when GPU plans are account-gated.

Optional:
  LINODE_TYPE / LINODE_REGION / LINODE_IMAGE
  PROOF_REPO / PROOF_BRANCH
  KEEP_LINODE=1  — skip destroy (YOU pay)
  N_ITEMS        — default 5000
"""

from __future__ import annotations

import json
import os
import secrets
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

API = "https://api.linode.com/v4"
TOKEN = os.environ.get("LINODE_TOKEN", "").strip()
TYPE = os.environ.get("LINODE_TYPE", "g6-nanode-1")
REGION = os.environ.get("LINODE_REGION", "us-east")
IMAGE = os.environ.get("LINODE_IMAGE", "linode/ubuntu22.04")
KEEP = os.environ.get("KEEP_LINODE", "") == "1"
REPO = os.environ.get(
    "PROOF_REPO", "https://github.com/StarkX-cloud/tron-client.git"
)
BRANCH = os.environ.get("PROOF_BRANCH", "cursor/economic-autopilot-38d1")
N_ITEMS = int(os.environ.get("N_ITEMS", "5000"))
HOURLY = float(os.environ.get("PROOF_HOURLY_RATE", "0.0075"))
LABEL_PREFIX = f"tron-dist-{int(time.time())}"


def die(msg: str, code: int = 1) -> None:
    print(f"[dist-proof] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def api(method: str, path: str, **kwargs):
    if not TOKEN:
        die(
            "LINODE_TOKEN is not set — add Runtime Secret and start a NEW agent"
        )
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
    key = tmpdir / "dist_proof_key"
    subprocess.check_call(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key), "-q"]
    )
    return key.with_suffix(".pub").read_text().strip(), key


def wait_linode(linode_id: int, want: str = "running", timeout: int = 600) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = api("GET", f"/linode/instances/{linode_id}")
        status = data.get("status")
        print(f"[dist-proof] linode {linode_id} status={status}")
        if status == want:
            return data
        time.sleep(8)
    die(f"timeout waiting for {linode_id} -> {want}")


def destroy_linode(linode_id: int | None) -> None:
    if linode_id is None:
        return
    print(f"[dist-proof] DESTROYING linode {linode_id}…")
    try:
        api("DELETE", f"/linode/instances/{linode_id}")
        print(f"[dist-proof] destroy requested for {linode_id}")
    except SystemExit as exc:
        print(f"[dist-proof] destroy failed: {exc}", file=sys.stderr)


def ssh(key: Path, host: str, command: str, timeout: int = 1800) -> str:
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
    print(f"[dist-proof] ssh root@{host}: {command[:100]}…")
    out = subprocess.check_output(cmd, timeout=timeout, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace")


def create_instance(label: str, pub: str, root_pass: str) -> dict:
    body = {
        "type": TYPE,
        "region": REGION,
        "image": IMAGE,
        "label": label[:32],
        "root_pass": root_pass,
        "authorized_keys": [pub],
        "booted": True,
        "tags": ["tron-dist-proof", "auto-destroy"],
    }
    print(f"[dist-proof] creating {TYPE} label={label} region={REGION}…")
    try:
        return api("POST", "/linode/instances", json=body)
    except SystemExit:
        for alt in ("us-ord", "us-sea", "us-mia", "eu-west"):
            if alt == REGION:
                continue
            body["region"] = alt
            print(f"[dist-proof] retry region {alt}…")
            try:
                return api("POST", "/linode/instances", json=body)
            except SystemExit as e:
                print(f"[dist-proof] {alt} failed: {e}")
        die("could not create CPU Nanode in any tried region")


def wait_ssh(key: Path, host: str) -> None:
    for _ in range(60):
        try:
            ssh(key, host, "echo ssh-ok")
            return
        except Exception:
            time.sleep(5)
    die(f"ssh never ready on {host}")


def bootstrap_repo(key: Path, host: str) -> None:
    setup = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git python3 python3-pip python3-venv
cd /opt
rm -rf tron-client
git clone --branch {BRANCH} {REPO} tron-client
cd tron-client
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -q -r requirements.txt
"""
    ssh(key, host, setup, timeout=1200)


def start_master_and_worker0(key: Path, host: str) -> None:
    cmd = f"""
set -euo pipefail
cd /opt/tron-client
. .venv/bin/activate
export TRON_HOST=0.0.0.0
export TRON_PORT=9000
nohup python3 queue_server.py > /var/log/tron-master.log 2>&1 &
sleep 2
export TRON_MASTER_URL=http://127.0.0.1:9000
export TRON_WORKER_NAME=dist-w0
export TRON_WORKER_SKU={TYPE}
export TRON_WORKER_HOURLY_RATE={HOURLY}
nohup python3 worker.py > /var/log/tron-worker0.log 2>&1 &
sleep 1
curl -sf http://127.0.0.1:9000/workers | head -c 400 || true
"""
    ssh(key, host, cmd, timeout=120)


def start_worker1(key: Path, host: str, master_ip: str) -> None:
    cmd = f"""
set -euo pipefail
cd /opt/tron-client
. .venv/bin/activate
export TRON_MASTER_URL=http://{master_ip}:9000
export TRON_WORKER_NAME=dist-w1
export TRON_WORKER_SKU={TYPE}
export TRON_WORKER_HOURLY_RATE={HOURLY}
nohup python3 worker.py > /var/log/tron-worker1.log 2>&1 &
sleep 2
curl -sf http://{master_ip}:9000/workers | head -c 600 || true
"""
    ssh(key, host, cmd, timeout=120)


def run_distributed_on_master(key: Path, host: str) -> dict:
    # Write + run a small driver that uses Autopilot distributed path.
    remote = "/tmp/dist_proof_driver.py"
    driver = f'''
import json, time
from dataclasses import replace
from tron_efficiency.advisor.recommend import Recommendation
from tron_efficiency.hardware.detect import HardwareInfo
from tron_router.decision import PlacementDecision, query_pool_status
from tron_autopilot.engine import EconomicAutopilot
from tron_autopilot.execute import execute_plan
from tron_autopilot.types import WorkloadSpec
from tron_autopilot.distributed.ops import closed_form_sum_squares
from tron_autopilot.store import RuntimeHistory

master = "http://127.0.0.1:9000"
deadline = time.time() + 180
status = None
while time.time() < deadline:
    status = query_pool_status(master)
    print("alive", status.alive_worker_count, "reachable", status.reachable)
    if status.reachable and status.alive_worker_count >= 2:
        break
    time.sleep(2)
else:
    raise SystemExit("never saw 2 alive workers")

n = {N_ITEMS}
items = list(range(n))
serial = 30.0

rec = Recommendation(
    model_name="dist/proof",
    estimated_param_count=1000,
    param_count_is_estimate=True,
    device="cpu",
    dtype="fp32",
    policy="baseline",
    memory_budget_bytes=None,
    confidence="medium",
    rationale=["dist proof stub — no HF download"],
    warnings=[],
)
hw = HardwareInfo(
    platform="linux",
    python_version="3.10",
    cpu_count=1,
    cpu_total_memory_bytes=1 * 1024**3,
    cuda_available=False,
    torch_version=None,
    gpus=[],
    warnings=[],
)

def recommend_fn(*a, **k):
    return rec

def decide_fn(job, hardware=None, pool_status=None, master_url=None):
    return PlacementDecision(
        target="pooled",
        confidence="medium",
        rationale=["dist proof"],
        local_recommendation=rec.to_dict(),
        pool_status=status.to_dict(),
        required_memory_gb=0.25,
    )

ap = EconomicAutopilot(
    log_path="/tmp/dist_decisions.jsonl",
    history_path="/tmp/dist_history.json",
    hardware=hw,
    recommend_fn=recommend_fn,
    decide_placement_fn=decide_fn,
    query_pool_status_fn=lambda master_url=None: status,
)
hist_key = RuntimeHistory.key("dist/proof", "local", "baseline", "cpu", "fp32")
ap.history.record(hist_key, serial)

spec = WorkloadSpec(
    model_name="dist/proof",
    hourly_rate_usd={HOURLY},
    pooled_hourly_rate_usd={HOURLY},
    master_url=master,
    parallelizable=True,
    partition_items=items,
    map_op="sum_squares",
    expected_runtime_seconds=serial,
    bytes_per_item=8.0,
    reliability="tolerant",
)
plan = ap.plan(spec, pool_status=status, strategy="autopilot")
chosen = plan.chosen.target if plan.chosen else None
dist = next(c for c in plan.candidates if c.target == "distributed")
print("chosen", chosen, "dist_feasible", dist.feasible, dist.rejection_reasons)
if not dist.feasible:
    raise SystemExit("distributed not feasible: " + str(dist.rejection_reasons))
if plan.chosen is None or plan.chosen.target != "distributed":
    plan = replace(plan, chosen=replace(dist, feasible=True))

t0 = time.time()
outcome = execute_plan(plan, strategy="autopilot", wait_timeout_seconds=300)
wall = time.time() - t0
expected = closed_form_sum_squares(n)
got = (outcome.metrics or {{}}).get("got_total")
report = {{
    "success": outcome.success,
    "target": outcome.target,
    "error": outcome.error,
    "expected_total": expected,
    "got_total": got,
    "correct": got == expected,
    "actual_runtime_seconds": outcome.actual_runtime_seconds,
    "wall_seconds": wall,
    "actual_cost_usd": outcome.actual_cost_usd,
    "predicted_runtime_seconds": outcome.predicted_runtime_seconds,
    "predicted_cost_usd": outcome.predicted_cost_usd,
    "metrics": outcome.metrics,
    "recovery": outcome.recovery,
    "n_items": n,
    "alive_workers": status.alive_worker_count,
    "pool_status": status.to_dict(),
    "decision_rationale": outcome.decision_rationale[:12],
}}
print(json.dumps(report))
open("/tmp/distributed_multi_machine_proof.json","w").write(json.dumps(report, indent=2))
if not outcome.success or got != expected:
    raise SystemExit("proof failed")
'''
    # Escape for remote write via python -c is messy; use heredoc over ssh.
    b64 = __import__("base64").b64encode(driver.encode()).decode()
    ssh(
        key,
        host,
        f"python3 -c \"import base64; open('{remote}','wb').write(base64.b64decode('{b64}'))\"",
        timeout=60,
    )
    out = ssh(
        key,
        host,
        "cd /opt/tron-client && . .venv/bin/activate && python3 "
        + remote,
        timeout=600,
    )
    print(out[-2000:])
    # Fetch JSON
    raw = ssh(key, host, "cat /tmp/distributed_multi_machine_proof.json", timeout=60)
    return json.loads(raw)


def main() -> int:
    print("[dist-proof] starting TWO-Nanode distributed proof (destroy in finally)")
    print(f"[dist-proof] type={TYPE} region={REGION} n_items={N_ITEMS}")
    ids: list[int | None] = [None, None]
    tmp = Path(tempfile.mkdtemp(prefix="tron-dist-"))
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    try:
        pub, key = create_ssh_key(tmp)
        root_pass = "Tr0n!" + "".join(
            secrets.choice(string.ascii_letters + string.digits) for _ in range(20)
        )
        created = []
        for i in range(2):
            label = f"{LABEL_PREFIX}-{i}"
            c = create_instance(label, pub, root_pass)
            ids[i] = c["id"]
            created.append(c)
            print(f"[dist-proof] created id={c['id']} label={c.get('label')}")

        hosts = []
        for i, lid in enumerate(ids):
            info = wait_linode(lid)  # type: ignore[arg-type]
            ips = info.get("ipv4") or []
            if not ips:
                die(f"no ipv4 on {lid}")
            hosts.append(ips[0])
            print(f"[dist-proof] node{i} host={ips[0]} id={lid}")

        for h in hosts:
            wait_ssh(key, h)
            bootstrap_repo(key, h)

        start_master_and_worker0(key, hosts[0])
        # Open firewall: ufw may block 9000 — ensure listen is public
        ssh(
            key,
            hosts[0],
            "command -v ufw >/dev/null && ufw allow 9000/tcp || true; "
            "ss -lntp | grep 9000 || true",
            timeout=60,
        )
        start_worker1(key, hosts[1], hosts[0])

        report = run_distributed_on_master(key, hosts[0])
        report["hosts"] = hosts
        report["linode_ids"] = ids
        report["type"] = TYPE
        report["region"] = REGION
        report["multi_machine"] = True
        report["same_host_simulation"] = False
        report["destroyed"] = not KEEP
        out_path = artifacts / "distributed_multi_machine_proof.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[dist-proof] SUCCESS")
        print(json.dumps(report, indent=2)[:2000])
        return 0 if report.get("correct") and report.get("success") else 1
    finally:
        if KEEP:
            print(f"[dist-proof] KEEP_LINODE=1 — leaving {ids} running (YOU pay)")
        else:
            for lid in ids:
                destroy_linode(lid)


if __name__ == "__main__":
    raise SystemExit(main())
