"""Core object model for the TRON execution spine.

Everything that flows through TRON — a task's code, its inputs, its result —
is content-addressed: identified by the hash of its bytes, not by an
arbitrarily assigned id. This is what makes the rest of the spine possible:

- Two tasks with identical code + identical inputs hash to the same id, so
  re-submitting work that already ran is a cache hit, not duplicate compute.
- A task's *lineage* (which artifacts it read, which task produced each of
  those artifacts) is just data sitting next to it, so a lost task can be
  re-derived on another node instead of requiring a checkpoint.
- The event log (see log.py) that records what happened is, by construction,
  a complete enough record to replay — which is what both fault recovery and
  the future 3D visualizer need.

This module intentionally has no I/O and no dependency on the rest of TRON —
it is pure data + hashing so it is easy to reason about and to test.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


def content_hash(data: bytes) -> str:
    """Return the content address (sha256 hex digest) for a blob of bytes."""
    return hashlib.sha256(data).hexdigest()


class TaskStatus(str, Enum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUEUED = "requeued"


@dataclass(frozen=True)
class Artifact:
    """A content-addressed blob: a task's serialized function, an input value,
    or a result. `artifact_id` is always `content_hash(data)` — constructing
    one with a mismatched id is a bug, not a valid state, so callers should
    build these via `Artifact.from_bytes` rather than the constructor.
    """

    artifact_id: str
    size_bytes: int
    created_at: float

    @staticmethod
    def from_bytes(data: bytes, created_at: Optional[float] = None) -> "Artifact":
        return Artifact(
            artifact_id=content_hash(data),
            size_bytes=len(data),
            created_at=created_at if created_at is not None else time.time(),
        )


@dataclass
class Task:
    """A unit of work in the execution graph.

    `task_id` is a content hash over (fn_hash, sorted input_hashes) plus the
    attempt number, so identical work submitted twice before it has a
    recorded result is recognized as identical. `attempt` is bumped on
    lineage-based re-derivation (see log.py `derive_recovery_plan`) so a
    retried task gets a distinct id from the original — retries are new
    graph nodes, not silent overwrites, which keeps replay honest.
    """

    task_id: str
    fn_hash: str
    input_hashes: tuple[str, ...]
    parent_task_ids: tuple[str, ...] = field(default_factory=tuple)
    attempt: int = 0
    status: TaskStatus = TaskStatus.QUEUED
    node_id: Optional[str] = None
    output_hash: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def compute_id(fn_hash: str, input_hashes: tuple[str, ...], attempt: int = 0) -> str:
        joined = fn_hash + "|" + "|".join(sorted(input_hashes)) + f"|attempt={attempt}"
        return content_hash(joined.encode("utf-8"))

    @staticmethod
    def new(
        fn_hash: str,
        input_hashes: tuple[str, ...],
        parent_task_ids: tuple[str, ...] = (),
        attempt: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Task":
        return Task(
            task_id=Task.compute_id(fn_hash, input_hashes, attempt),
            fn_hash=fn_hash,
            input_hashes=tuple(input_hashes),
            parent_task_ids=tuple(parent_task_ids),
            attempt=attempt,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class Event:
    """One entry in the append-only execution log. `seq` is assigned by the
    log itself on append (monotonically increasing) — never set it by hand.
    """

    seq: int
    task_id: str
    type: str
    data: dict[str, Any]
    timestamp: float
    node_id: Optional[str] = None
