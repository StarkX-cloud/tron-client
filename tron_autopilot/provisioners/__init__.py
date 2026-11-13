"""Optional dedicated-capacity provisioners for Autopilot.

Unset = dedicated stays a refused flag (honest). Set
``TRON_DEDICATED_PROVISIONER=linode`` (+ ``LINODE_TOKEN``) to create a
short-lived worker, attach it to the master, run, then destroy.
"""

from __future__ import annotations

import os
from typing import Protocol


class ProvisionedWorker(Protocol):
    worker_name: str
    linode_id: int | None
    hourly_rate_usd: float
    sku: str

    def destroy(self) -> None: ...


def get_provisioner():
    name = (os.environ.get("TRON_DEDICATED_PROVISIONER") or "").strip().lower()
    if not name:
        return None
    if name == "linode":
        from tron_autopilot.provisioners.linode import LinodeProvisioner

        return LinodeProvisioner()
    raise ValueError(f"unknown TRON_DEDICATED_PROVISIONER={name!r}")
