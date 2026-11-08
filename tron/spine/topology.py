"""Topology awareness for placement decisions.

Phase 2's actual claim is "place work by measured network cost, not by
pretending every node is equidistant." It measures two real signals, both
self-reported by the worker:

- **Latency** — round-trip time of the worker's own heartbeat request
  (genuine network RTT plus server processing time). This is what matters
  for "is this worker cheap or expensive to hand a small unit of work to."
- **Bandwidth** — throughput in Mbps, measured by the worker actually
  transferring a payload to and from the master (`/probe/blob`,
  `/probe/sink` in queue_server.py) and dividing bytes by seconds. This is
  what matters for "how expensive is it to ship this job's *inputs* to
  that worker" — a heavy artifact over a thin pipe is a real cost that
  latency alone doesn't capture. `matcher.py` / `global_brain.py` fold
  both into one placement score.
- **Free memory** — GB currently available on the worker (real GPU VRAM
  via `nvidia-smi`, or system RAM via `psutil` on a CPU-only worker),
  self-reported on every heartbeat (see worker.py's
  `measure_free_memory_gb`). Unlike latency/bandwidth this is used as a
  **hard filter**, not a soft penalty, in `matcher.py` — a job that
  genuinely will not fit in a node's currently-free memory is excluded
  from that node, not merely scored lower, because the failure mode is an
  OOM, not "a bit slower."

This is deliberately a star-topology model: workers talk to the master,
not to each other, which is the actual shape of the current cluster. A
peer-to-peer mesh (for e.g. moving artifacts worker-to-worker instead of
through the master) is future work once workers can reach one another
directly — see ROADMAP.md.

Pure logic, no I/O — same design as model.py, so it's cheap to test.
Bandwidth is stored with the exact same EWMA-plus-age-out machinery as
latency, because a stale throughput number is just as misleading as a
stale RTT one.
"""
from __future__ import annotations

import threading
import time
from typing import Optional


class TopologyMap:
    def __init__(self, smoothing: float = 0.3, max_age_seconds: float = 120.0):
        """`smoothing` is the EWMA weight given to each new sample (higher
        = more reactive to recent measurements, lower = steadier under a
        noisy network). `max_age_seconds` bounds how long a measurement is
        trusted before a node is treated as unmeasured again — a worker
        that stopped reporting shouldn't keep coasting on a stale good
        score.
        """
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        self._smoothing = smoothing
        self._max_age = max_age_seconds
        self._lock = threading.Lock()
        self._latency_ms: dict[tuple[str, str], float] = {}
        self._last_seen: dict[tuple[str, str], float] = {}
        self._bandwidth_mbps: dict[tuple[str, str], float] = {}
        self._bw_last_seen: dict[tuple[str, str], float] = {}
        # Memory is a per-node property (how much is free on that node right
        # now), not a per-link measurement like latency/bandwidth — so it's
        # keyed by node name alone, not a (from, to) pair.
        self._free_memory_gb: dict[str, float] = {}
        self._mem_last_seen: dict[str, float] = {}

    @staticmethod
    def _key(from_node: str, to_node: str) -> tuple[str, str]:
        return (from_node, to_node)

    def record_latency(
        self,
        from_node: str,
        to_node: str,
        latency_ms: float,
        timestamp: Optional[float] = None,
    ) -> None:
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        ts = timestamp if timestamp is not None else time.time()
        key = self._key(from_node, to_node)
        with self._lock:
            previous = self._latency_ms.get(key)
            smoothed = latency_ms if previous is None else previous + self._smoothing * (latency_ms - previous)
            self._latency_ms[key] = smoothed
            self._last_seen[key] = ts

    def latency(self, from_node: str, to_node: str, now: Optional[float] = None) -> Optional[float]:
        """Smoothed latency estimate in ms, or None if never measured or
        the measurement has aged out."""
        key = self._key(from_node, to_node)
        with self._lock:
            value = self._latency_ms.get(key)
            last_seen = self._last_seen.get(key)
        if value is None or last_seen is None:
            return None
        current_time = now if now is not None else time.time()
        if current_time - last_seen > self._max_age:
            return None
        return value

    def record_bandwidth(
        self,
        from_node: str,
        to_node: str,
        mbps: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a measured throughput sample (megabits/sec) for the link
        from `from_node` to `to_node`. Smoothed with the same EWMA weight
        as latency. Zero is rejected alongside negatives: a real transfer
        that completed took some non-zero time, so 0 Mbps is a
        measurement bug, not a slow link."""
        if mbps <= 0:
            raise ValueError("mbps must be positive")
        ts = timestamp if timestamp is not None else time.time()
        key = self._key(from_node, to_node)
        with self._lock:
            previous = self._bandwidth_mbps.get(key)
            smoothed = mbps if previous is None else previous + self._smoothing * (mbps - previous)
            self._bandwidth_mbps[key] = smoothed
            self._bw_last_seen[key] = ts

    def bandwidth(self, from_node: str, to_node: str, now: Optional[float] = None) -> Optional[float]:
        """Smoothed throughput estimate in Mbps, or None if never measured
        or the measurement has aged out — same contract as `latency()`."""
        key = self._key(from_node, to_node)
        with self._lock:
            value = self._bandwidth_mbps.get(key)
            last_seen = self._bw_last_seen.get(key)
        if value is None or last_seen is None:
            return None
        current_time = now if now is not None else time.time()
        if current_time - last_seen > self._max_age:
            return None
        return value

    def record_memory(
        self,
        node: str,
        free_gb: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a self-reported free-memory sample (GB) for `node` — real
        currently-available GPU VRAM if the worker has a GPU, else free
        system RAM (see worker.py's `measure_free_memory_gb`). Same EWMA
        smoothing and age-out as latency/bandwidth: a node that stopped
        reporting shouldn't keep coasting on a stale "plenty of room"
        reading. Zero is accepted (a node can genuinely be out of room);
        only negative values are rejected as a measurement bug."""
        if free_gb < 0:
            raise ValueError("free_gb must be non-negative")
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            previous = self._free_memory_gb.get(node)
            smoothed = free_gb if previous is None else previous + self._smoothing * (free_gb - previous)
            self._free_memory_gb[node] = smoothed
            self._mem_last_seen[node] = ts

    def memory(self, node: str, now: Optional[float] = None) -> Optional[float]:
        """Smoothed free-memory estimate in GB for `node`, or None if never
        measured or the measurement has aged out — same contract as
        `latency()`/`bandwidth()`."""
        with self._lock:
            value = self._free_memory_gb.get(node)
            last_seen = self._mem_last_seen.get(node)
        if value is None or last_seen is None:
            return None
        current_time = now if now is not None else time.time()
        if current_time - last_seen > self._max_age:
            return None
        return value

    def rank_nodes(self, from_node: str, candidate_nodes: list[str]) -> list[str]:
        """`candidate_nodes` sorted by ascending measured latency from
        `from_node`. Unmeasured nodes are placed in the middle of the
        ranking rather than first or last — that avoids both starving new
        nodes of work (never placing them because they're unproven) and
        blindly trusting them (treating "unknown" as "best")."""
        measured: list[tuple[float, str]] = []
        unmeasured: list[str] = []
        for node in candidate_nodes:
            lat = self.latency(from_node, node)
            if lat is None:
                unmeasured.append(node)
            else:
                measured.append((lat, node))
        measured.sort(key=lambda pair: pair[0])
        mid = len(measured) // 2
        return [n for _, n in measured[:mid]] + unmeasured + [n for _, n in measured[mid:]]
