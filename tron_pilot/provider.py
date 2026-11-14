"""Provider onboarding: identity, capacity advertise, verification flag.

Wraps existing ``/register_worker`` — does not replace the queue_server
worker map. Verification is an operator attestation stored in a local
registry file (not KYC / not hardware attestation).
"""

from __future__ import annotations

import json
import os
import secrets
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests

from tron_pilot.security import redact_secrets


@dataclass
class ProviderRecord:
    provider_id: str
    worker_name: str
    contact: str | None = None
    region: str | None = None
    sku: str | None = None
    hourly_rate_usd: float | None = None
    memory_gb: float | None = None
    efficiency_runtime: bool = False
    cuda_advertised: bool = False
    verified: bool = False
    verified_at: float | None = None
    verified_by: str | None = None
    registered_at: float = field(default_factory=time.time)
    last_seen_at: float | None = None
    master_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ProviderRegistry:
    def __init__(self, path: str | os.PathLike | None = None):
        default = os.environ.get("TRON_PROVIDER_REGISTRY", "tron_providers.json")
        self.path = Path(path or default)
        self._providers: dict[str, ProviderRecord] = {}
        self._load()

    def register(
        self,
        *,
        worker_name: str,
        master_url: str,
        contact: str | None = None,
        region: str | None = None,
        sku: str | None = None,
        hourly_rate_usd: float | None = None,
        memory_gb: float | None = None,
        efficiency_runtime: bool = False,
        cuda_advertised: bool = False,
        metadata: dict | None = None,
        register_with_master: bool = True,
        timeout: float = 15.0,
    ) -> tuple[ProviderRecord, str | None]:
        """Create a durable provider_id and optionally hit ``/register_worker``.

        Returns ``(record, worker_auth_token_or_None)``.
        """
        provider_id = f"prov_{uuid.uuid4().hex[:12]}"
        auth_token = None
        if register_with_master:
            payload = {
                "name": worker_name,
                "memory_gb": memory_gb,
                "sku": sku,
                "hourly_rate_usd": hourly_rate_usd,
                "efficiency_runtime": efficiency_runtime,
                "location": region,
            }
            # Drop Nones so queue_server uses defaults.
            payload = {k: v for k, v in payload.items() if v is not None}
            r = requests.post(
                f"{master_url.rstrip('/')}/register_worker",
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            auth_token = data.get("auth_token")
            # Prefer server-assigned name if present
            worker_name = data.get("name") or worker_name

        rec = ProviderRecord(
            provider_id=provider_id,
            worker_name=worker_name,
            contact=contact,
            region=region,
            sku=sku or os.environ.get("TRON_WORKER_SKU"),
            hourly_rate_usd=hourly_rate_usd,
            memory_gb=memory_gb,
            efficiency_runtime=efficiency_runtime,
            cuda_advertised=cuda_advertised,
            master_url=master_url.rstrip("/"),
            metadata=dict(metadata or {}),
            last_seen_at=time.time(),
        )
        self._providers[provider_id] = rec
        self._persist()
        return rec, auth_token

    def verify(
        self,
        provider_id: str,
        *,
        verified_by: str = "operator",
        challenge: str | None = None,
    ) -> ProviderRecord:
        """Mark provider verified after operator attestation.

        Optional ``challenge``: if set, provider must present the same value
        previously issued via ``issue_challenge`` (shared-secret style).
        """
        rec = self._require(provider_id)
        pending = (rec.metadata or {}).get("pending_challenge")
        if pending is not None:
            if not challenge or not secrets.compare_digest(str(challenge), str(pending)):
                raise PermissionError("challenge mismatch — provider not verified")
            rec.metadata.pop("pending_challenge", None)
        rec.verified = True
        rec.verified_at = time.time()
        rec.verified_by = verified_by
        self._persist()
        return rec

    def issue_challenge(self, provider_id: str) -> str:
        rec = self._require(provider_id)
        challenge = secrets.token_urlsafe(16)
        rec.metadata["pending_challenge"] = challenge
        self._persist()
        return challenge

    def touch(self, provider_id: str) -> None:
        rec = self._require(provider_id)
        rec.last_seen_at = time.time()
        self._persist()

    def get(self, provider_id: str) -> ProviderRecord | None:
        return self._providers.get(provider_id)

    def by_worker(self, worker_name: str) -> ProviderRecord | None:
        for rec in self._providers.values():
            if rec.worker_name == worker_name:
                return rec
        return None

    def list_providers(self, *, verified_only: bool = False) -> list[dict]:
        rows = []
        for rec in self._providers.values():
            if verified_only and not rec.verified:
                continue
            rows.append(redact_secrets(rec.to_dict()))
        return rows

    def sync_health_from_master(self, master_url: str | None = None) -> dict:
        """Refresh last_seen from GET /workers (alive workers only)."""
        url = (master_url or os.environ.get("TRON_MASTER_URL") or "http://localhost:9000")
        url = url.rstrip("/")
        r = requests.get(f"{url}/workers", timeout=10)
        r.raise_for_status()
        workers = r.json()
        alive_names = set()
        # queue_server returns {worker_name: {...}} or {"workers":[...]}
        if isinstance(workers, dict):
            if "workers" in workers and isinstance(workers["workers"], list):
                iterable = workers["workers"]
            elif "items" in workers and isinstance(workers["items"], list):
                iterable = workers["items"]
            else:
                iterable = [
                    {"name": k, **(v if isinstance(v, dict) else {})}
                    for k, v in workers.items()
                    if isinstance(v, dict)
                ]
        else:
            iterable = workers or []
        for w in iterable:
            if not isinstance(w, dict):
                continue
            name = w.get("name") or w.get("worker_name")
            if name:
                alive_names.add(name)
        updated = 0
        for rec in self._providers.values():
            if rec.worker_name in alive_names:
                rec.last_seen_at = time.time()
                updated += 1
        self._persist()
        return {
            "alive_workers": sorted(alive_names),
            "providers_touched": updated,
            "providers_total": len(self._providers),
        }

    def _require(self, provider_id: str) -> ProviderRecord:
        rec = self._providers.get(provider_id)
        if rec is None:
            raise KeyError(f"unknown provider_id {provider_id!r}")
        return rec

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {pid: rec.to_dict() for pid, rec in self._providers.items()}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        for pid, row in data.items():
            if not isinstance(row, dict):
                continue
            self._providers[pid] = ProviderRecord(
                provider_id=row.get("provider_id") or pid,
                worker_name=str(row.get("worker_name") or ""),
                contact=row.get("contact"),
                region=row.get("region"),
                sku=row.get("sku"),
                hourly_rate_usd=row.get("hourly_rate_usd"),
                memory_gb=row.get("memory_gb"),
                efficiency_runtime=bool(row.get("efficiency_runtime")),
                cuda_advertised=bool(row.get("cuda_advertised")),
                verified=bool(row.get("verified")),
                verified_at=row.get("verified_at"),
                verified_by=row.get("verified_by"),
                registered_at=float(row.get("registered_at") or time.time()),
                last_seen_at=row.get("last_seen_at"),
                master_url=row.get("master_url"),
                metadata=dict(row.get("metadata") or {}),
            )
