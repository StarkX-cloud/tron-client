#!/usr/bin/env python3
"""Open a Linode Support ticket requesting GPU plan access.

Requires env:
  LINODE_TOKEN — PAT with scopes:
    - linodes:read_write  (create/destroy proof VMs)
    - account:read_write  (support tickets)  ← missing on the current secret

Usage:
  python3 scripts/linode_request_gpu_access.py
"""

from __future__ import annotations

import json
import os
import sys

import requests

API = "https://api.linode.com/v4"
TOKEN = os.environ.get("LINODE_TOKEN", "").strip()
PLAN = os.environ.get("LINODE_TYPE", "g2-gpu-rtx4000a1-s")

SUMMARY = "Please enable GPU Linode plans on this account"
DESCRIPTION = f"""We need to deploy {PLAN} (or any RTX 4000 Ada / GPU plan) for a short Autopilot CUDA validation of open-source project TRON (StarkX-cloud/tron-client).

API create currently returns:
"A limit on your account is preventing the deployment of the selected Linode plan. To request access to the plan, please contact Support and provide the Linode plan name."

Please enable GPU plan access (or advise if a $100 deposit is required). We will delete the instance immediately after a ~1 hour proof run.
"""


def main() -> int:
    if not TOKEN:
        print("ERROR: LINODE_TOKEN not set", file=sys.stderr)
        return 1
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }
    r = requests.post(
        f"{API}/support/tickets",
        headers=headers,
        json={"summary": SUMMARY, "description": DESCRIPTION},
        timeout=60,
    )
    print(f"HTTP {r.status_code}")
    try:
        body = r.json()
    except Exception:
        print(r.text[:800])
        return 1
    print(json.dumps(body, indent=2)[:2000])
    if r.status_code == 401:
        print(
            "\nToken lacks account scope. Regenerate PAT with Account: Read/Write "
            "(keep Linodes: Read/Write), update the LINODE_TOKEN secret, then re-run.",
            file=sys.stderr,
        )
        return 2
    if r.status_code >= 400:
        return 1
    ticket_id = body.get("id") or (body.get("ticket") or {}).get("id")
    print(f"\nOpened ticket id={ticket_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
