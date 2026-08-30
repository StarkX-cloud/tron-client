"""Topology awareness for placement decisions.

Phase 2's actual claim is "place work by measured network cost, not by
pretending every node is equidistant." v1 measures one real signal:
round-trip latency between the master and each worker, self-reported by
the worker (it times its own heartbeat request — that captures genuine
network RTT plus server processing time, which is the number that matters
for "is this worker cheap or expensive to hand work to").

This is deliberately a star-topology model: workers talk to the master,
not to each other, which is the actual shape of the current cluster. A
peer-to-peer mesh (for e.g. moving artifacts worker-to-worker instead of
through the master) is future work once workers can reach one another
directly — see ROADMAP.md.

Pure logic, no I/O — same design as model.py, so it's cheap to test and
easy to reuse from a future bandwidth prober without rewriting this.
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
