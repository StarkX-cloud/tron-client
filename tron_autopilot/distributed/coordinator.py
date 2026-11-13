"""Coordinate independent shard jobs on the existing worker pool.

Uses ``submit_to_pool`` + matcher + watchdog. Deduplicates by
``idempotency_key`` so retries never double-apply results. Partial
failures retry individually; unrecoverable shards fail the wave.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from tron_router.decision import JobRequest, PlacementDecision
from tron_router.handoff import HandoffError, submit_to_pool

from tron_autopilot.distributed.ops import reduce_map_results, run_map_shard
from tron_autopilot.distributed.partition import ShardSpec


@dataclass
class ShardResult:
    shard_id: int
    idempotency_key: str
    success: bool
    result: dict | None = None
    error: str | None = None
    job_id: str | None = None
    attempts: int = 0
    requeue_count: int = 0


@dataclass
class DistributedRunResult:
    success: bool
    op: str
    reduced: dict | None
    shard_results: list[ShardResult] = field(default_factory=list)
    wall_seconds: float = 0.0
    error: str | None = None
    attempts_total: int = 0
    duplicates_suppressed: int = 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "op": self.op,
            "reduced": self.reduced,
            "wall_seconds": self.wall_seconds,
            "error": self.error,
            "attempts_total": self.attempts_total,
            "duplicates_suppressed": self.duplicates_suppressed,
            "shards": [
                {
                    "shard_id": s.shard_id,
                    "idempotency_key": s.idempotency_key,
                    "success": s.success,
                    "job_id": s.job_id,
                    "attempts": s.attempts,
                    "requeue_count": s.requeue_count,
                    "error": s.error,
                    "result": s.result,
                }
                for s in self.shard_results
            ],
        }


class DistributedCoordinator:
    """Submit shards, wait, retry failures, reduce — with idempotent collect."""

    def __init__(
        self,
        master_url: str,
        *,
        max_attempts_per_shard: int = 3,
        wait_timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 0.1,
        required_memory_gb: float | None = 0.25,
    ):
        self.master_url = master_url.rstrip("/")
        self.max_attempts_per_shard = max_attempts_per_shard
        self.wait_timeout_seconds = wait_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.required_memory_gb = required_memory_gb
        # idempotency_key → first successful result (prevents corruption)
        self._completed: dict[str, dict] = {}
        self.duplicates_suppressed = 0

    def _submit_one(self, shard: ShardSpec, op: str):
        job = JobRequest(model_name=f"distributed-map/{op}", tolerant=True)
        decision = PlacementDecision(
            target="pooled",
            confidence="medium",
            rationale=["distributed shard"],
            required_memory_gb=self.required_memory_gb,
        )
        return submit_to_pool(
            job,
            decision,
            run_map_shard,
            func_kwargs={
                "shard_id": shard.shard_id,
                "items": list(shard.items),
                "op": op,
                "idempotency_key": shard.idempotency_key,
            },
            master_url=self.master_url,
            required_memory_gb=self.required_memory_gb,
            min_memory_gb=self.required_memory_gb or 1.0,
        )

    def _await(self, future) -> tuple[Any, int, str | None]:
        requeue = 0
        try:
            status = future.status() if hasattr(future, "status") else {}
            if isinstance(status, dict):
                requeue = int(status.get("requeue_count") or 0)
            result = future.get(
                timeout=self.wait_timeout_seconds,
                poll=self.poll_interval_seconds,
            )
            status = future.status() if hasattr(future, "status") else status
            if isinstance(status, dict):
                requeue = max(requeue, int(status.get("requeue_count") or 0))
            return result, requeue, None
        except Exception as exc:  # noqa: BLE001
            return None, requeue, f"{type(exc).__name__}: {exc}"

    def run(
        self,
        shards: list[ShardSpec],
        *,
        op: str = "sum_squares",
    ) -> DistributedRunResult:
        start = time.monotonic()
        attempts_total = 0
        final: dict[int, ShardResult] = {}

        pending = list(shards)
        for attempt_round in range(1, self.max_attempts_per_shard + 1):
            if not pending:
                break
            wave = list(pending)
            pending = []
            futures = []
            for shard in wave:
                # Already have a successful result for this key? Skip submit.
                if shard.idempotency_key in self._completed:
                    self.duplicates_suppressed += 1
                    final[shard.shard_id] = ShardResult(
                        shard_id=shard.shard_id,
                        idempotency_key=shard.idempotency_key,
                        success=True,
                        result=self._completed[shard.idempotency_key],
                        attempts=attempt_round,
                    )
                    continue
                try:
                    fut = self._submit_one(shard, op)
                    futures.append((shard, fut))
                    attempts_total += 1
                except HandoffError as exc:
                    pending.append(shard)
                    final[shard.shard_id] = ShardResult(
                        shard_id=shard.shard_id,
                        idempotency_key=shard.idempotency_key,
                        success=False,
                        error=str(exc),
                        attempts=attempt_round,
                    )

            for shard, fut in futures:
                result, requeue, err = self._await(fut)
                job_id = getattr(fut, "job_id", None)
                if err or not isinstance(result, dict):
                    pending.append(shard)
                    final[shard.shard_id] = ShardResult(
                        shard_id=shard.shard_id,
                        idempotency_key=shard.idempotency_key,
                        success=False,
                        error=err or "invalid shard result",
                        job_id=job_id,
                        attempts=attempt_round,
                        requeue_count=requeue,
                    )
                    continue
                # Idempotent accept: first success wins
                key = shard.idempotency_key
                if key in self._completed:
                    self.duplicates_suppressed += 1
                    stored = self._completed[key]
                else:
                    self._completed[key] = result
                    stored = result
                final[shard.shard_id] = ShardResult(
                    shard_id=shard.shard_id,
                    idempotency_key=key,
                    success=True,
                    result=stored,
                    job_id=job_id,
                    attempts=attempt_round,
                    requeue_count=requeue,
                )

        wall = time.monotonic() - start
        shard_results = [final[i] for i in sorted(final)]
        if len(shard_results) != len(shards) or any(not s.success for s in shard_results):
            return DistributedRunResult(
                success=False,
                op=op,
                reduced=None,
                shard_results=shard_results,
                wall_seconds=wall,
                error="one or more shards failed after retries",
                attempts_total=attempts_total,
                duplicates_suppressed=self.duplicates_suppressed,
            )

        try:
            reduced = reduce_map_results(
                [s.result for s in shard_results if s.result], op=op
            )
        except ValueError as exc:
            return DistributedRunResult(
                success=False,
                op=op,
                reduced=None,
                shard_results=shard_results,
                wall_seconds=wall,
                error=str(exc),
                attempts_total=attempts_total,
                duplicates_suppressed=self.duplicates_suppressed,
            )

        return DistributedRunResult(
            success=True,
            op=op,
            reduced=reduced,
            shard_results=shard_results,
            wall_seconds=wall,
            attempts_total=attempts_total,
            duplicates_suppressed=self.duplicates_suppressed,
        )


def run_distributed_map(
    master_url: str,
    shards: list[ShardSpec],
    *,
    op: str = "sum_squares",
    max_attempts_per_shard: int = 3,
    wait_timeout_seconds: float = 120.0,
) -> DistributedRunResult:
    return DistributedCoordinator(
        master_url,
        max_attempts_per_shard=max_attempts_per_shard,
        wait_timeout_seconds=wait_timeout_seconds,
    ).run(shards, op=op)
