"""TRON execution spine: content-addressed artifacts + an append-only,
replayable event log for tasks.

This is Phase 1 of the rebuild. Everything downstream depends on it:

- Phase 2 (heterogeneous fault-tolerant fabric) recovers dead nodes' work
  via `EventLog.recover_orphaned_tasks`.
- Phase 3 (distributed training) treats a training step's local-SGD delta
  as just another content-addressed Artifact flowing through the same log.
- Phase 4 (the 3D Grid) is a renderer over `EventLog.replay()` /
  `EventLog.snapshot()` — it has no separate data source.

See model.py, store.py, log.py for the pieces; ARCHITECTURE.md for the plan.
"""
from .model import Artifact, Event, Task, TaskStatus, content_hash
from .store import ArtifactStore
from .log import EventLog
from .topology import TopologyMap
from .matcher import match_jobs_to_workers, score_pair

__all__ = [
    "Artifact",
    "Event",
    "Task",
    "TaskStatus",
    "content_hash",
    "ArtifactStore",
    "EventLog",
    "TopologyMap",
    "match_jobs_to_workers",
    "score_pair",
]
