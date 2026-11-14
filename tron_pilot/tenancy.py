"""Soft tenant tagging — not a sandbox.

Jobs and usage events may carry ``customer_id`` / ``tenant_id`` for
filtering reports. This does **not** isolate process memory or defend
against malicious cloudpickle. Trust boundary: operators must only
admit verified providers they control (see ``tron_pilot.security``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TenantContext:
    customer_id: str
    tenant_id: str | None = None
    project_id: str | None = None

    def tags(self) -> dict:
        out = {"customer_id": self.customer_id}
        if self.tenant_id:
            out["tenant_id"] = self.tenant_id
        if self.project_id:
            out["project_id"] = self.project_id
        return out
