"""Phase 2b: the master-side match step.

Phase 2a found a real limit and documented it rather than hiding it (see
tron_runtime/global_brain.py and ARCHITECTURE.md): /next_job is
worker-pull, so no two workers' scores for the same job were ever compared
in one decision. This module is the actual fix — it solves the assignment
problem directly: given N queued jobs and M idle workers, find the
assignment that maximizes total placement score across *all* pairs at
once, not N independent per-worker argmax calls.

This uses scipy's linear_sum_assignment (the Hungarian algorithm) rather
than a hand-rolled "sort all pairs, greedily take the best" heuristic,
because greedy is provably not optimal here. Worked example that failed
under greedy during development:

    worker "near" (latency 5ms), worker "far" (latency 300ms)
    job "heavy" (weight 20, priority 5), job "light" (weight 1, priority 5)

    scores: heavy-near=4.0  heavy-far=-55.0  light-near=4.95  light-far=2.0

Greedy picks the single highest-scoring pair first (light-near, 4.95),
which stimulate strands "heavy" with "far" (-55.0) — total score -50.05.
The globally optimal pairing is heavy-near + light-far, total score 6.0:
worse for light individually, far better overall, because it puts the
job that suffers most from latency on the low-latency worker. Greedy
cannot see that trade because it commits to the best-looking single pair
before considering the pair it strands. For the queue sizes this system
deals with (workers and jobs per match cycle: tens, not millions), the
O(n^3) cost of doing this exactly is not a real concern.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .topology import TopologyMap

# A pair that fails a hard constraint (job needs a GPU, worker has none)
# gets this score so it is effectively never chosen — finite, not -inf,
# because linear_sum_assignment requires a finite cost matrix.
INFEASIBLE_SCORE = -1e9


def score_pair(
    job: dict,
    worker_name: str,
    worker: dict,
    topology: TopologyMap,
    master_node: str = "master",
) -> float:
    """Same scoring logic as GlobalDecisionBrain.decide, exposed standalone
    so the matcher can build a full cost matrix without going through the
    per-worker /next_job code path."""
    if job.get("gpu") and not worker.get("gpu"):
        return INFEASIBLE_SCORE

    priority = float(job.get("priority", 1))
    weight = float(job.get("compute_weight", job.get("memory_gb", 1)))
    latency_ms = topology.latency(master_node, worker_name)
    penalty = 0.0 if latency_ms is None else (latency_ms / 100.0) * weight
    return priority - penalty


def match_jobs_to_workers(
    jobs: list[dict],
    idle_workers: dict[str, dict],
    topology: TopologyMap,
    master_node: str = "master",
) -> list[tuple[str, str]]:
    """Return (job_id, worker_name) pairs forming the assignment that
    maximizes total score across all pairs simultaneously. Leftover jobs
    or workers (unequal counts, or pairs that hit INFEASIBLE_SCORE) are
    simply omitted — they're picked up on the next match cycle, or by
    /next_job's own per-worker fallback for anything not yet matched.
    """
    if not jobs or not idle_workers:
        return []

    worker_names = list(idle_workers.keys())
    n_jobs = len(jobs)
    n_workers = len(worker_names)

    cost = np.zeros((n_jobs, n_workers))
    for i, job in enumerate(jobs):
        for j, worker_name in enumerate(worker_names):
            score = score_pair(job, worker_name, idle_workers[worker_name], topology, master_node)
            cost[i, j] = -score  # linear_sum_assignment minimizes; we want to maximize score

    row_indices, col_indices = linear_sum_assignment(cost)

    assignments = []
    for row, col in zip(row_indices, col_indices):
        score = -cost[row, col]
        if score <= INFEASIBLE_SCORE:
            continue  # this pairing broke a hard constraint; leave both unmatched
        assignments.append((jobs[row]["id"], worker_names[col]))
    return assignments
