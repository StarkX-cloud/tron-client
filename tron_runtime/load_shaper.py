"""Congestion-aware job release.

Phase 1 shipped this as a pass-through: `reshape()` returned `delay: 0`
for every job and `shape()` returned the job untouched — an empty
interface, exactly the kind of module the rebuild's "implement it or
delete it" rule targets. Phase 2c gave it the one number it was missing:
measured per-worker bandwidth. So it is now real.

The model: when the master hands a job to a worker it must first ship
that job's inputs (`job["transfer_bytes"]`, per Phase 2c) over the
master->worker link. A worker whose measured downlink is `B` Mbps can
absorb about `B * 1e6 / 8 * window_seconds` bytes before further
transfers start queuing on the wire and every transfer to that worker
slows down. `LoadShaper` tracks the bytes dispatched to each worker over
a sliding `window_seconds` window and refuses to pile a new transfer
onto a worker already at its window budget — the job waits for a later
cycle instead.

The window is a **link cooldown**, not a job-lifetime tracker: it ages
out on its own `window_seconds` after dispatch, whether or not the job
that carried those bytes has finished. A slow-pipe worker that just
received a big transfer and immediately reported the job done is still
recovering its link, so it still shouldn't be handed the next big
transfer right away. `release()` exists for the one case where the bytes
provably never went (a dead worker whose job is being re-derived
elsewhere), not for normal completion.

Inert where it should be: a worker whose bandwidth has never been
measured has capacity `None` (unlimited) — same "don't act on what
hasn't been measured" rule as `TopologyMap` and `GlobalDecisionBrain`.
A job with no `transfer_bytes` is never held. So on a cluster with no
bandwidth data, or for the existing no-`transfer_bytes` job traffic,
this behaves exactly like the old pass-through.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Optional

DEFAULT_WINDOW_SECONDS = 2.0


class LoadShaper:
    def __init__(self, window_seconds: float = DEFAULT_WINDOW_SECONDS):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        # worker_name -> {job_id: (dispatched_at, transfer_bytes)}
        self._inflight: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)

    # -- window bookkeeping --------------------------------------------

    def _prune(self, worker_name: str, now: float) -> None:
        cutoff = now - self.window_seconds
        entries = self._inflight.get(worker_name)
        if not entries:
            return
        stale = [jid for jid, (ts, _) in entries.items() if ts < cutoff]
        for jid in stale:
            del entries[jid]

    def inflight_bytes(self, worker_name: str, now: Optional[float] = None) -> float:
        now = time.time() if now is None else now
        with self._lock:
            self._prune(worker_name, now)
            return sum(b for _, b in self._inflight.get(worker_name, {}).values())

    def reserve(self, worker_name: str, job_id: str, transfer_bytes: float, now: Optional[float] = None) -> None:
        """Record `transfer_bytes` as dispatched to `worker_name` for
        `job_id`. Idempotent on `job_id` — safe to call both when the
        match cycle reserves the pairing and again when the worker
        actually claims it."""
        now = time.time() if now is None else now
        with self._lock:
            # a job can only be in flight to one worker; drop any prior holder
            for entries in self._inflight.values():
                entries.pop(job_id, None)
            self._inflight[worker_name][job_id] = (now, max(0.0, float(transfer_bytes or 0)))

    def release(self, job_id: str) -> None:
        """Forget this job's dispatched bytes *now* instead of waiting for
        the window to age them out. Only for the case where the transfer
        provably didn't happen — a dead worker whose job is being
        re-derived elsewhere. Normal completion does NOT call this: the
        link is still cooling down (see the module docstring)."""
        with self._lock:
            for entries in self._inflight.values():
                entries.pop(job_id, None)

    # -- capacity decisions ------------------------------------------------

    def capacity_bytes(self, worker_name: str, topology, now: Optional[float] = None) -> Optional[float]:
        """Bytes this worker's measured downlink can absorb in one window,
        or None (treat as unlimited) if bandwidth was never measured."""
        mbps = topology.bandwidth("master", worker_name)
        if not mbps:
            return None
        return (mbps * 1_000_000.0 / 8.0) * self.window_seconds

    def can_accept(self, worker_name: str, job_bytes: float, topology, now: Optional[float] = None) -> bool:
        """True if dispatching `job_bytes` to this worker now won't pile
        onto an already-busy link. An *idle* link accepts anything — even
        a single job larger than one window's budget; the shaper prevents
        pile-up, it doesn't cap job size (a job that big would otherwise
        be unschedulable forever). Once something is in flight, the
        window budget applies."""
        cap = self.capacity_bytes(worker_name, topology, now)
        if cap is None:
            return True
        job_bytes = max(0.0, float(job_bytes or 0))
        if job_bytes == 0:
            return True
        inflight = self.inflight_bytes(worker_name, now)
        if inflight == 0:
            return True
        return inflight + job_bytes <= cap

    # -- the pull-path shaper (kept; now backed by can_accept) ------------

    def reshape(self, job_queue, workers, topology=None, now: Optional[float] = None):
        """Return `[{"job": job, "delay": 0|1}, ...]`. `delay == 1` tells
        /next_job to skip that job this cycle. Without `topology` (or with
        no measured bandwidth anywhere) this is the old pass-through. With
        it, a job whose `transfer_bytes` no measured idle worker currently
        has window room for is held until one does."""
        if topology is None:
            return [{"job": job, "delay": 0} for job in job_queue]

        idle = [n for n, w in workers.items() if w.get("status") == "idle"]
        measured_idle = [n for n in idle if self.capacity_bytes(n, topology, now) is not None]
        # An idle worker with no measured bandwidth is an escape valve —
        # "unlimited" capacity by the same don't-act-on-the-unmeasured
        # rule — so a job that could go there is never held.
        has_unmeasured_idle = len(measured_idle) < len(idle)

        shaped = []
        for job in job_queue:
            job_bytes = float(job.get("transfer_bytes", 0) or 0)
            if job_bytes <= 0 or has_unmeasured_idle or not measured_idle:
                shaped.append({"job": job, "delay": 0})
                continue
            room = any(self.can_accept(n, job_bytes, topology, now) for n in measured_idle)
            shaped.append({"job": job, "delay": 0 if room else 1})

        # releasable jobs first, but otherwise keep queue order (stable sort)
        shaped.sort(key=lambda e: e["delay"])
        return shaped
