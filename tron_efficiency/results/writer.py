"""Persist and export benchmark results.

Runs are stored as individual JSON files under ``TRON_EFFICIENCY_RESULTS_DIR``
(default ``.tron_efficiency_results/runs/`` at the repo root, gitignored) —
one file per run, so a crashed later run never corrupts earlier results.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from tron_efficiency.results.schema import BenchmarkResult


def _results_dir() -> Path:
    base = os.environ.get("TRON_EFFICIENCY_RESULTS_DIR", ".tron_efficiency_results")
    path = Path(base) / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(result: BenchmarkResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)


_CSV_FIELDNAMES = [
    "run_id",
    "timestamp",
    "policy_name",
    "model_name",
    "device",
    "batch_size",
    "max_new_tokens",
    "success",
    "error",
    "load_time_seconds",
    "time_to_first_token_seconds",
    "end_to_end_latency_seconds",
    "tokens_generated",
    "tokens_per_second",
    "peak_cpu_memory_bytes",
    "peak_gpu_memory_allocated_bytes",
    "peak_gpu_memory_reserved_bytes",
]


def write_csv_rows(results: list[BenchmarkResult], fileobj) -> None:
    """Flatten each result's outcome metrics into one CSV row, written to
    any open text file-like object (so callers can target a real file or an
    in-memory buffer, e.g. for a streamed HTTP response)."""
    writer = csv.DictWriter(fileobj, fieldnames=_CSV_FIELDNAMES)
    writer.writeheader()
    for r in results:
        o = r.outcome
        writer.writerow(
            {
                "run_id": r.run_id,
                "timestamp": r.timestamp,
                "policy_name": o.policy_name,
                "model_name": r.config.model_name,
                "device": r.config.device,
                "batch_size": r.config.batch_size,
                "max_new_tokens": r.config.max_new_tokens,
                "success": o.success,
                "error": o.error or "",
                "load_time_seconds": o.load_time_seconds,
                "time_to_first_token_seconds": o.time_to_first_token_seconds,
                "end_to_end_latency_seconds": o.end_to_end_latency_seconds,
                "tokens_generated": o.tokens_generated,
                "tokens_per_second": o.tokens_per_second,
                "peak_cpu_memory_bytes": o.peak_cpu_memory_bytes,
                "peak_gpu_memory_allocated_bytes": o.peak_gpu_memory_allocated_bytes,
                "peak_gpu_memory_reserved_bytes": o.peak_gpu_memory_reserved_bytes,
            }
        )


def write_csv(results: list[BenchmarkResult], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        write_csv_rows(results, f)


def save_run(result: BenchmarkResult) -> Path:
    path = _results_dir() / f"{result.run_id}.json"
    write_json(result, str(path))
    return path


def load_run(run_id: str) -> BenchmarkResult:
    path = _results_dir() / f"{run_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return BenchmarkResult.from_dict(json.load(f))


def list_runs() -> list[BenchmarkResult]:
    results = []
    for path in sorted(_results_dir().glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            results.append(BenchmarkResult.from_dict(json.load(f)))
    return results
