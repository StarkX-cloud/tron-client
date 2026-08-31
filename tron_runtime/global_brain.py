"""Placement scorer for /next_job.

Phase 1 shipped this as a stub returning the job's priority unchanged.
Phase 2 makes it real: score is the job's priority, adjusted by how
expensive it measurably is, in network terms, to place this job on the
worker being considered — a latency term (Phase 2a) and an
input-transfer term over measured bandwidth (Phase 2c). See
tron/spine/topology.py for how both numbers are collected, and
tron/spine/matcher.py, which shares this formula so the periodic match
step and this per-worker fallback never disagree about what a placement
costs. One deliberate difference: matcher.score_pair also takes an
optional outcome-history bonus (how well a node's past training runs
went), which is NOT replicated here. That term depends only on
worker_name, not on the job — and decide() below picks the best JOB for
one fixed worker in a single call, so a worker-only term would shift
every candidate job's score by the same constant and therefore never
change which one wins argmax. It would be dead weight here; it earns its
keep only in the matcher's cross-worker assignment.
"""
from __future__ import annotations

from typing import Optional

from tron.spine import TopologyMap
from tron.spine.matcher import _transfer_penalty


class GlobalDecisionBrain:
    def __init__(self, topology: TopologyMap, swarm=None, load_shaper=None):
        self.topology = topology
        self.swarm = swarm
        self.load_shaper = load_shaper

    def decide(self, job_queue, workers, job, worker_name: Optional[str] = None) -> dict:
        base_score = float(job.get("priority", 1))

        if worker_name is None:
            return {"score": base_score, "latency_ms": None}

        latency_ms = self.topology.latency("master", worker_name)
        if latency_ms is None:
            # No measurement yet (new or infrequently-heartbeating worker)
            # — don't penalize what hasn't been measured.
            return {"score": base_score, "latency_ms": None}

        # IMPORTANT: /next_job picks the highest-scoring job out of the
        # queue for ONE specific worker in a single call. A penalty that's
        # constant across every job in that call (e.g. "latency_ms / 100")
        # shifts every candidate's score by the same amount and therefore
        # has zero effect on which job wins argmax — it only changes the
        # absolute score value, not the selection. (The pre-existing
        # gpu_bonus/memory_factor multipliers in queue_server.py have this
        # same property, for the same reason.)
        #
        # So the penalty has to scale with something job-specific to
        # actually influence which job this worker is handed: a heavier
        # job (compute_weight) suffers more from a slow link than a light
        # one, which also happens to be the right policy — don't route
        # your expensive work through your worst connection.
        #
        # This does NOT solve cross-worker arbitration (choosing which of
        # several idle workers should get a specific job) — the current
        # architecture is worker-pull, not master-push, so there is no
        # single decision point where two workers' scores for the same job
        # are ever compared. That needs a push/match scheduler; see
        # ROADMAP.md.
        compute_weight = float(job.get("compute_weight", job.get("memory_gb", 1)))
        penalty = (latency_ms / 100.0) * compute_weight

        # Phase 2c: shipping a heavy artifact to this worker costs measured
        # transfer time. Same helper (and therefore the same formula) as
        # the match step. Zero unless the job declares `transfer_bytes` and
        # this link's bandwidth has actually been measured — so every
        # pre-bandwidth score is unchanged. Like the latency penalty this
        # is job-specific (scales with the job's own byte count), which is
        # what lets it actually influence which job a worker is handed
        # rather than just shifting every candidate equally.
        transfer_penalty = _transfer_penalty(job, worker_name, self.topology, "master")

        return {
            "score": base_score - penalty - transfer_penalty,
            "latency_ms": latency_ms,
        }
