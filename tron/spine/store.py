"""Content-addressed artifact storage.

Any bytes handed to `put()` are deduplicated by content hash: submitting the
same function or the same input value twice — from the same node or a
different one — stores it once. This is what lets the scheduler later place
work where its inputs already live instead of shipping them again.

v1 is a local filesystem store (one file per artifact, named by hash). It is
deliberately not clustered yet — that is Phase 2's job (topology-aware
placement needs this interface to exist first, not the other way around).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from .model import Artifact, content_hash


class ArtifactStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root is not None else Path(".tron_spine") / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, artifact_id: str) -> Path:
        # Two-level fan-out so a single directory never holds huge numbers
        # of files.
        return self.root / artifact_id[:2] / artifact_id

    def put(self, data: bytes) -> Artifact:
        artifact = Artifact.from_bytes(data)
        path = self._path_for(artifact.artifact_id)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_bytes(data)
            tmp_path.replace(path)
        return artifact

    def get(self, artifact_id: str) -> bytes:
        path = self._path_for(artifact_id)
        if not path.exists():
            raise KeyError(f"artifact not found: {artifact_id}")
        return path.read_bytes()

    def exists(self, artifact_id: str) -> bool:
        return self._path_for(artifact_id).exists()

    def verify(self, artifact_id: str) -> bool:
        """Re-hash the stored bytes and confirm they still match the id —
        cheap corruption/tamper check."""
        if not self.exists(artifact_id):
            return False
        return content_hash(self.get(artifact_id)) == artifact_id

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
