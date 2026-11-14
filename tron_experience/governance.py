"""Data governance for the Experience Engine.

- Tenant isolation on ledger reads (default)
- Explicit opt-in for cross-customer shared learning
- Configurable retention
- Optional at-rest encryption for model artifacts (Fernet)
- Never trains on prompts / code / model I/O (enforced by fingerprint schema)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GovernancePolicy:
    allow_shared_learning: bool = False
    retention_days: int | None = None
    encrypt_artifacts: bool = False
    # customer_id → opted into shared learning
    shared_opt_in: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "GovernancePolicy":
        shared = os.environ.get("TRON_EXPERIENCE_SHARED_LEARNING", "").lower() in (
            "1",
            "true",
            "yes",
        )
        retention = os.environ.get("TRON_EXPERIENCE_RETENTION_DAYS")
        encrypt = bool(os.environ.get("TRON_ARTIFACT_ENCRYPTION_KEY"))
        opt_in_raw = os.environ.get("TRON_EXPERIENCE_SHARED_OPT_IN", "")
        opt_in = {}
        for part in opt_in_raw.split(","):
            part = part.strip()
            if part:
                opt_in[part] = True
        return cls(
            allow_shared_learning=shared,
            retention_days=int(retention) if retention else None,
            encrypt_artifacts=encrypt,
            shared_opt_in=opt_in,
        )

    def retention_cutoff(self) -> float | None:
        if not self.retention_days:
            return None
        return time.time() - (self.retention_days * 86400)

    def customer_opted_into_shared(self, customer_id: str) -> bool:
        return bool(self.shared_opt_in.get(customer_id))

    def allows_record(self, record: Any) -> bool:
        """Reject records that accidentally carry confidential content keys."""
        d = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        banned = ("prompt", "function", "code", "input_text", "output_text", "logits")
        return not any(k in d for k in banned)


@dataclass
class ModelVersion:
    version_id: str
    created_at: float
    train_n: int
    holdout_n: int
    metric: str
    mae: float
    baseline_mae: float
    features: list[str]
    artifact_path: str | None
    status: str  # candidate | active | retired
    notes: str = ""
    beats_baseline: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    def __init__(self, path: str | os.PathLike | None = None):
        default = os.environ.get("TRON_EXPERIENCE_MODELS", "tron_experience_models.json")
        self.path = Path(path or default)
        self._versions: dict[str, ModelVersion] = {}
        self._load()

    def register(self, version: ModelVersion) -> ModelVersion:
        self._versions[version.version_id] = version
        self._persist()
        return version

    def promote(self, version_id: str) -> ModelVersion:
        v = self._require(version_id)
        if not v.beats_baseline:
            raise PermissionError(
                f"refusing to promote {version_id}: does not beat baseline "
                f"(mae={v.mae} baseline={v.baseline_mae})"
            )
        for other in self._versions.values():
            if other.status == "active":
                other.status = "retired"
        v.status = "active"
        self._persist()
        return v

    def rollback(self, version_id: str) -> ModelVersion:
        """Make a retired/candidate version active; retire current active."""
        v = self._require(version_id)
        for other in self._versions.values():
            if other.status == "active":
                other.status = "retired"
        v.status = "active"
        self._persist()
        return v

    def active(self) -> ModelVersion | None:
        for v in self._versions.values():
            if v.status == "active":
                return v
        return None

    def list_versions(self) -> list[dict]:
        return [v.to_dict() for v in sorted(self._versions.values(), key=lambda x: x.created_at)]

    def _require(self, version_id: str) -> ModelVersion:
        if version_id not in self._versions:
            raise KeyError(f"unknown model version {version_id!r}")
        return self._versions[version_id]

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {vid: v.to_dict() for vid, v in self._versions.items()}
        raw = json.dumps(payload, indent=2)
        key = os.environ.get("TRON_ARTIFACT_ENCRYPTION_KEY")
        if key:
            try:
                from cryptography.fernet import Fernet

                token = Fernet(key.encode() if isinstance(key, str) else key).encrypt(
                    raw.encode("utf-8")
                )
                self.path.write_bytes(b"FERNET:" + token)
                return
            except Exception:  # noqa: BLE001 — fall back to plaintext
                pass
        self.path.write_text(raw, encoding="utf-8")

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw_bytes = self.path.read_bytes()
        text: str
        if raw_bytes.startswith(b"FERNET:"):
            key = os.environ.get("TRON_ARTIFACT_ENCRYPTION_KEY")
            if not key:
                return
            from cryptography.fernet import Fernet

            text = Fernet(key.encode() if isinstance(key, str) else key).decrypt(
                raw_bytes[len(b"FERNET:") :]
            ).decode("utf-8")
        else:
            text = raw_bytes.decode("utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        for vid, row in (data or {}).items():
            self._versions[vid] = ModelVersion(**row)
