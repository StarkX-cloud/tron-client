"""Provision a short-lived Linode, bootstrap a TRON worker, tear down.

Defaults to a cheap CPU Nanode (account can create these). Set
``LINODE_TYPE`` to a GPU plan when the account is unlocked.
"""

from __future__ import annotations

import base64
import os
import secrets
import time
from dataclasses import dataclass

import requests

from tron_autopilot.pricing import hourly_usd_for_sku, resolve_hourly_rate

API = "https://api.linode.com/v4"


@dataclass
class LinodeWorker:
    worker_name: str
    linode_id: int
    hourly_rate_usd: float
    sku: str
    root_pass: str
    ip: str | None = None

    def destroy(self) -> None:
        token = os.environ.get("LINODE_TOKEN", "").strip()
        if not token or not self.linode_id:
            return
        headers = {"Authorization": f"Bearer {token}"}
        try:
            requests.delete(
                f"{API}/linode/instances/{self.linode_id}",
                headers=headers,
                timeout=60,
            )
        except Exception:  # noqa: BLE001
            pass


class LinodeProvisioner:
    def provision(
        self,
        *,
        master_url: str,
        label_prefix: str = "tron-ap",
        wait_seconds: float = 300.0,
    ) -> LinodeWorker:
        token = os.environ.get("LINODE_TOKEN", "").strip()
        if not token:
            raise RuntimeError("LINODE_TOKEN not set")
        sku = os.environ.get("LINODE_TYPE", "g6-nanode-1")
        region = os.environ.get("LINODE_REGION", "us-east")
        image = os.environ.get("LINODE_IMAGE", "linode/ubuntu22.04")
        repo = os.environ.get(
            "PROOF_REPO", "https://github.com/StarkX-cloud/tron-client.git"
        )
        branch = os.environ.get("PROOF_BRANCH", "cursor/economic-autopilot-38d1")
        worker_name = f"{label_prefix}-{secrets.token_hex(3)}"
        root_pass = secrets.token_urlsafe(24)
        rate = resolve_hourly_rate(
            sku=sku,
            advertised=None,
            fallback=hourly_usd_for_sku(sku) or 0.05,
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        # cloud-init: clone repo, install minimal deps, start worker
        userdata = f"""#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-pip git
cd /opt
git clone --branch {branch} {repo} tron-client
cd tron-client
pip3 install -q -r requirements.txt || true
export TRON_MASTER_URL={master_url.rstrip('/')}
export TRON_WORKER_NAME={worker_name}
export TRON_WORKER_SKU={sku}
export TRON_WORKER_HOURLY_RATE={rate}
nohup python3 worker.py > /var/log/tron-worker.log 2>&1 &
"""
        body = {
            "type": sku,
            "region": region,
            "image": image,
            "label": worker_name[:32],
            "root_pass": root_pass,
            "authorized_keys": [],
            "booted": True,
            "metadata": {
                "user_data": base64.b64encode(userdata.encode("utf-8")).decode("ascii")
            },
        }
        r = requests.post(
            f"{API}/linode/instances", headers=headers, json=body, timeout=60
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Linode create failed: {r.status_code} {r.text[:500]}")
        data = r.json()
        linode_id = int(data["id"])
        ip = None
        ips = data.get("ipv4") or []
        if ips:
            ip = ips[0]
        worker = LinodeWorker(
            worker_name=worker_name,
            linode_id=linode_id,
            hourly_rate_usd=float(rate or 0.05),
            sku=sku,
            root_pass=root_pass,
            ip=ip,
        )
        # Wait until the worker registers with the master (best-effort).
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            try:
                wr = requests.get(f"{master_url.rstrip('/')}/workers", timeout=5)
                if wr.ok:
                    workers = wr.json() or {}
                    if worker_name in workers:
                        return worker
            except Exception:  # noqa: BLE001
                pass
            time.sleep(5)
        # Still return — caller may use SSH/debug; destroy on failure path.
        return worker
