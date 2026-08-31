"""Content-addressed artifact storage.

Any bytes handed to `put()` are deduplicated by content hash: submitting the
same function or the same input value twice — from the same node or a
different one — stores it once. This is what lets the scheduler later place
work where its inputs already live instead of shipping them again.

v1 is a local filesystem store (one file per artifact, named by hash). It is
deliberately not clustered yet — that is Phase 2's job (topology-aware
placement needs this interface to exist first, not the other way around).

Optionally encrypted at rest (Fernet/AES-128-CBC+HMAC via the
`cryptography` package). This closes a real gap from the "verified against
a live public master" WAN test (see writeup/distributed-training.md):
TRON_TRAINING_AUTH_TOKEN controls who can *submit or fetch* an artifact
over the wire, but says nothing about someone who gets at the master's
disk directly (a stolen backup, a misconfigured bucket, a curious host
provider) — every training vector and adapter sat there in plaintext.
Encryption is content-addressing-transparent: `artifact_id` is always
`content_hash` of the *plaintext* (computed before encryption, checked
after decryption), so turning encryption on or off never changes what an
artifact's id is or breaks anything that references one by hash.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from .model import Artifact, content_hash

try:
    from cryptography.fernet import Fernet, InvalidToken
    _HAS_CRYPTO = True
except ImportError:  # pragma: no cover - cryptography is a normal dependency
    _HAS_CRYPTO = False


class ArtifactStore:
    def __init__(self, root: Optional[Path] = None, encryption_key: Optional[str] = None):
        self.root = Path(root) if root is not None else Path(".tron_spine") / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)
        self._fernet = None
        if encryption_key:
            if not _HAS_CRYPTO:
                raise RuntimeError(
                    "encryption_key was given but the 'cryptography' package is not "
                    "installed (pip install cryptography)"
                )
            key = encryption_key.encode() if isinstance(encryption_key, str) else encryption_key
            self._fernet = Fernet(key)

    def _path_for(self, artifact_id: str) -> Path:
        # Two-level fan-out so a single directory never holds huge numbers
        # of files.
        return self.root / artifact_id[:2] / artifact_id

    def put(self, data: bytes) -> Artifact:
        # Hash BEFORE encrypting: artifact_id is always the plaintext's
        # content address, so encryption is invisible to every caller that
        # references an artifact by hash (which is all of them).
        artifact = Artifact.from_bytes(data)
        path = self._path_for(artifact.artifact_id)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".tmp")
            on_disk = self._fernet.encrypt(data) if self._fernet else data
            tmp_path.write_bytes(on_disk)
            tmp_path.replace(path)
        return artifact

    def get(self, artifact_id: str) -> bytes:
        path = self._path_for(artifact_id)
        if not path.exists():
            raise KeyError(f"artifact not found: {artifact_id}")
        raw = path.read_bytes()
        if self._fernet is None:
            return raw
        try:
            return self._fernet.decrypt(raw)
        except InvalidToken:
            # Either the key is wrong for this artifact, or it's plaintext
            # written before encryption was turned on for this deployment
            # (a real, expected case: enabling TRON_ARTIFACT_ENCRYPTION_KEY
            # on an existing store doesn't retroactively encrypt old data).
            # Content-addressing lets us tell the difference honestly
            # instead of guessing: if the raw bytes themselves hash to the
            # id being asked for, they really are pre-encryption plaintext
            # — return them rather than treat a routine key-enable as data
            # loss. Otherwise the key genuinely doesn't open this artifact.
            if content_hash(raw) == artifact_id:
                return raw
            raise

    def exists(self, artifact_id: str) -> bool:
        return self._path_for(artifact_id).exists()

    def verify(self, artifact_id: str) -> bool:
        """Re-hash the stored bytes and confirm they still match the id —
        cheap corruption/tamper check."""
        if not self.exists(artifact_id):
            return False
        try:
            return content_hash(self.get(artifact_id)) == artifact_id
        except InvalidToken:
            return False

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
