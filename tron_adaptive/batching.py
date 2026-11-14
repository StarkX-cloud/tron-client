"""Dynamic workload grouping for independent items.

Groups items into batches sized by measured capacity — not a claim that
batching always wins. Small batches reduce latency; larger batches raise
throughput when memory allows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchPlan:
    batch_size: int
    n_batches: int
    rationale: list[str]


def choose_batch_size(
    n_items: int,
    *,
    measured_peak_memory_bytes: int | None,
    memory_budget_bytes: int | None,
    max_batch_size: int = 8,
    min_batch_size: int = 1,
    history_best_batch: int | None = None,
) -> BatchPlan:
    """Pick a batch size from measured memory headroom + optional history."""
    rationale: list[str] = []
    if n_items <= 1:
        return BatchPlan(1, max(1, n_items), ["single item — batch_size=1"])

    if history_best_batch is not None and history_best_batch >= 1:
        b = max(min_batch_size, min(int(history_best_batch), max_batch_size, n_items))
        rationale.append(
            f"using measured best batch_size={b} from performance history"
        )
        return BatchPlan(b, (n_items + b - 1) // b, rationale)

    if (
        measured_peak_memory_bytes is not None
        and memory_budget_bytes is not None
        and measured_peak_memory_bytes > 0
        and memory_budget_bytes > measured_peak_memory_bytes
    ):
        # Conservative: allow batch growth proportional to unused budget,
        # capped — never assume linear scaling is free.
        headroom = memory_budget_bytes / float(measured_peak_memory_bytes)
        b = int(min(max_batch_size, max(min_batch_size, headroom // 1.5)))
        b = max(min_batch_size, min(b, n_items))
        rationale.append(
            f"memory headroom ~{headroom:.2f}x → batch_size={b} "
            f"(budget {memory_budget_bytes}, measured peak {measured_peak_memory_bytes})"
        )
        return BatchPlan(b, (n_items + b - 1) // b, rationale)

    rationale.append(
        "no measured headroom/history — default batch_size=1 "
        "(does not invent free throughput)"
    )
    return BatchPlan(1, n_items, rationale)


def group_items(items: list, batch_size: int) -> list[list]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
