"""The append-only execution log — the single source of truth for the spine.

Every task lifecycle transition (queued, assigned, started, completed,
failed, requeued) is appended here as an `Event`, never mutated in place.
Three things fall out of that for free, which is the whole point of doing
it this way instead of an ad-hoc mutable job dict:

1. **Replay** — `snapshot()` reconstructs current task state by folding the
   log from the start. Anyone (a new node, a debugger, later the 3D
   visualizer) can rebuild the same state independently, or rewind to any
   point in the past by folding only up to that `seq`.

2. **Lineage-based recovery** — if a node dies mid-task, we don't need a
   checkpoint of its progress. `recover_orphaned_tasks()` looks up the dead
   task's (fn_hash, input_hashes) from the log and returns enough to
   re-derive it — the same content-addressed inputs it started with, on any
   other node.

3. **Dedup across the log's whole history** — because task ids are content
   hashes (see model.py), re-submitting identical work is visible in the log
   as an already-completed task, not a fresh one.

v1 storage is SQLite (one row per event, autoincrement `seq`). This is a
single-process log; distributing it is Phase 2 work and should not change
this module's interface.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from .model import Event, Task, TaskStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    type TEXT NOT NULL,
    data TEXT NOT NULL,
    timestamp REAL NOT NULL,
    node_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_task_id ON events(task_id);
"""


class EventLog:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else Path(".tron_spine") / "log.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def append(
        self,
        task_id: str,
        type: str,
        data: Optional[dict] = None,
        node_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> Event:
        ts = timestamp if timestamp is not None else time.time()
        payload = json.dumps(data or {})
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (task_id, type, data, timestamp, node_id) VALUES (?, ?, ?, ?, ?)",
                (task_id, type, payload, ts, node_id),
            )
            self._conn.commit()
            seq = cur.lastrowid
        return Event(seq=seq, task_id=task_id, type=type, data=data or {}, timestamp=ts, node_id=node_id)

    def replay(self, since_seq: int = 0) -> Iterable[Event]:
        """Yield every event with seq > since_seq, in order."""
        cur = self._conn.execute(
            "SELECT seq, task_id, type, data, timestamp, node_id FROM events WHERE seq > ? ORDER BY seq ASC",
            (since_seq,),
        )
        for seq, task_id, type_, data, ts, node_id in cur.fetchall():
            yield Event(seq=seq, task_id=task_id, type=type_, data=json.loads(data), timestamp=ts, node_id=node_id)

    def events_for(self, task_id: str) -> list[Event]:
        cur = self._conn.execute(
            "SELECT seq, task_id, type, data, timestamp, node_id FROM events WHERE task_id = ? ORDER BY seq ASC",
            (task_id,),
        )
        return [
            Event(seq=seq, task_id=tid, type=type_, data=json.loads(data), timestamp=ts, node_id=node_id)
            for seq, tid, type_, data, ts, node_id in cur.fetchall()
        ]

    def latest_seq(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM events").fetchone()
        return row[0]

    def snapshot(self, up_to_seq: Optional[int] = None) -> dict[str, Task]:
        """Fold the log into current Task objects. Passing `up_to_seq` gives
        you the state of the world as of that point — this is the "rewind"
        operation a debugger or the 3D time-scrubber needs.
        """
        tasks: dict[str, Task] = {}
        for event in self.replay(since_seq=0):
            if up_to_seq is not None and event.seq > up_to_seq:
                break
            self._fold(tasks, event)
        return tasks

    @staticmethod
    def _fold(tasks: dict[str, Task], event: Event) -> None:
        task = tasks.get(event.task_id)

        if event.type == "queued":
            if task is None:
                data = event.data
                task = Task(
                    task_id=event.task_id,
                    fn_hash=data.get("fn_hash", ""),
                    input_hashes=tuple(data.get("input_hashes", ())),
                    parent_task_ids=tuple(data.get("parent_task_ids", ())),
                    attempt=data.get("attempt", 0),
                    metadata=data.get("metadata", {}),
                )
                tasks[event.task_id] = task
            task.status = TaskStatus.QUEUED
            return

        if task is None:
            # An event arrived for a task we have no "queued" record for
            # (e.g. replaying from a truncated log) — synthesize a minimal
            # entry rather than dropping the event on the floor.
            task = Task(task_id=event.task_id, fn_hash="", input_hashes=())
            tasks[event.task_id] = task

        if event.type == "assigned":
            task.status = TaskStatus.ASSIGNED
            task.node_id = event.data.get("node_id") or event.node_id
        elif event.type == "started":
            task.status = TaskStatus.RUNNING
            task.node_id = event.data.get("node_id") or event.node_id or task.node_id
        elif event.type == "completed":
            task.status = TaskStatus.COMPLETED
            task.output_hash = event.data.get("output_hash")
        elif event.type == "failed":
            task.status = TaskStatus.FAILED
        elif event.type == "requeued":
            task.status = TaskStatus.REQUEUED
            task.node_id = None

    def recover_orphaned_tasks(self, dead_node_id: str) -> list[Task]:
        """Return tasks last assigned to `dead_node_id` that never reached
        completed/failed — i.e. what a watchdog should re-derive elsewhere.
        Because task identity is content-addressed, the caller can hand
        these straight back to the scheduler as fresh work: same fn_hash,
        same input_hashes, next attempt number.
        """
        tasks = self.snapshot()
        orphaned = []
        for task in tasks.values():
            if task.node_id == dead_node_id and task.status in (
                TaskStatus.ASSIGNED,
                TaskStatus.RUNNING,
            ):
                orphaned.append(task)
        return orphaned

    def close(self) -> None:
        self._conn.close()
