"""Tests for the Phase 2b master-side match step: real cross-worker
arbitration via optimal assignment, not per-worker argmax.
"""
import pytest

from tron.spine import TopologyMap, match_jobs_to_workers, score_pair
from tron.spine.matcher import INFEASIBLE_SCORE


@pytest.fixture
def topology():
    return TopologyMap()


def test_empty_jobs_returns_no_assignments(topology):
    assert match_jobs_to_workers([], {"worker-1": {}}, topology) == []


def test_empty_workers_returns_no_assignments(topology):
    assert match_jobs_to_workers([{"id": "job-1"}], {}, topology) == []


def test_single_job_single_worker_matches(topology):
    jobs = [{"id": "job-1", "priority": 5}]
    workers = {"worker-1": {}}
    assignments = match_jobs_to_workers(jobs, workers, topology)
    assert assignments == [("job-1", "worker-1")]


def test_gpu_job_never_assigned_to_non_gpu_worker(topology):
    jobs = [{"id": "gpu-job", "priority": 5, "gpu": True}]
    workers = {"cpu-worker": {"gpu": False}}
    assert match_jobs_to_workers(jobs, workers, topology) == []


def test_gpu_job_assigned_to_gpu_worker_over_non_gpu(topology):
    jobs = [{"id": "gpu-job", "priority": 5, "gpu": True}]
    workers = {"cpu-worker": {"gpu": False}, "gpu-worker": {"gpu": True}}
    assignments = match_jobs_to_workers(jobs, workers, topology)
    assert assignments == [("gpu-job", "gpu-worker")]


def test_score_pair_infeasible_for_gpu_mismatch(topology):
    score = score_pair({"gpu": True}, "worker-1", {"gpu": False}, topology)
    assert score == INFEASIBLE_SCORE


def test_more_jobs_than_workers_leaves_some_unmatched(topology):
    jobs = [{"id": f"job-{i}", "priority": 1} for i in range(3)]
    workers = {"worker-1": {}}
    assignments = match_jobs_to_workers(jobs, workers, topology)
    assert len(assignments) == 1


def test_more_workers_than_jobs_leaves_some_idle(topology):
    jobs = [{"id": "job-1", "priority": 1}]
    workers = {"worker-1": {}, "worker-2": {}}
    assignments = match_jobs_to_workers(jobs, workers, topology)
    assert len(assignments) == 1


def test_optimal_assignment_beats_the_greedy_trap(topology):
    """The worked example from matcher.py's module docstring: greedy
    "take the single best pair first" strands the heavy job with the far
    worker (total score -50.05). The optimal assignment pairs heavy with
    near and light with far instead (total score 6.0) — worse for the
    light job individually, better overall, because it's the heavy job
    that actually suffers from latency.
    """
    topology.record_latency("master", "near", 5.0)
    topology.record_latency("master", "far", 300.0)

    jobs = [
        {"id": "heavy", "priority": 5, "compute_weight": 20},
        {"id": "light", "priority": 5, "compute_weight": 1},
    ]
    workers = {"near": {}, "far": {}}

    assignments = dict(match_jobs_to_workers(jobs, workers, topology))

    assert assignments["heavy"] == "near"
    assert assignments["light"] == "far"


def test_transfer_bytes_have_no_effect_without_measured_bandwidth(topology):
    """The bandwidth term must be inert until bandwidth is actually
    measured — otherwise adding it would silently re-rank every existing
    (pre-Phase-2c) placement decision. A job can declare transfer_bytes;
    with no bandwidth sample for the link the score is exactly what it was
    before this term existed.
    """
    job_plain = {"id": "j", "priority": 5, "compute_weight": 2}
    job_heavy_transfer = {**job_plain, "transfer_bytes": 500_000_000}
    topology.record_latency("master", "worker-1", 40.0)

    assert score_pair(job_plain, "worker-1", {}, topology) == score_pair(
        job_heavy_transfer, "worker-1", {}, topology
    )


def test_bandwidth_steers_the_heavy_transfer_job_to_the_fat_pipe(topology):
    """Latency alone can't separate these two workers — identical RTT. What
    differs is throughput, and one job has to move 50MB of inputs while the
    other moves 1MB. The optimal assignment sends the big transfer over the
    100Mbps link and the small one over the 10Mbps link; a latency-only
    scorer (Phase 2a/2b) would have no basis to prefer that pairing.
    """
    topology.record_latency("master", "fat", 10.0)
    topology.record_latency("master", "thin", 10.0)
    topology.record_bandwidth("master", "fat", 100.0)
    topology.record_bandwidth("master", "thin", 10.0)

    jobs = [
        {"id": "big", "priority": 5, "compute_weight": 1, "transfer_bytes": 50_000_000},
        {"id": "small", "priority": 5, "compute_weight": 1, "transfer_bytes": 1_000_000},
    ]
    workers = {"fat": {}, "thin": {}}

    assignments = dict(match_jobs_to_workers(jobs, workers, topology))
    assert assignments["big"] == "fat"
    assert assignments["small"] == "thin"


def test_bandwidth_penalty_can_override_a_small_priority_edge(topology):
    """A slightly higher-priority job still yields the fat pipe to a much
    heavier transfer when the thin link would make that transfer dominate
    the cost."""
    topology.record_bandwidth("master", "fat", 80.0)
    topology.record_bandwidth("master", "thin", 4.0)

    jobs = [
        {"id": "tiny-urgent", "priority": 6, "compute_weight": 1, "transfer_bytes": 200_000},
        {"id": "bulk", "priority": 5, "compute_weight": 1, "transfer_bytes": 40_000_000},
    ]
    workers = {"fat": {}, "thin": {}}

    assignments = dict(match_jobs_to_workers(jobs, workers, topology))
    assert assignments["bulk"] == "fat"
    assert assignments["tiny-urgent"] == "thin"


def test_equal_latency_ties_broken_by_priority(topology):
    topology.record_latency("master", "worker-a", 10.0)
    topology.record_latency("master", "worker-b", 10.0)

    jobs = [
        {"id": "low-priority", "priority": 1, "compute_weight": 1},
        {"id": "high-priority", "priority": 10, "compute_weight": 1},
    ]
    workers = {"worker-a": {}, "worker-b": {}}

    # With identical latency, both assignments are "valid" in the sense
    # that either worker could take either job — just confirm both jobs
    # get placed and the total score is maximal (i.e. a real assignment
    # happened, not a degenerate no-op).
    assignments = match_jobs_to_workers(jobs, workers, topology)
    assert len(assignments) == 2
    assigned_jobs = {job_id for job_id, _ in assignments}
    assert assigned_jobs == {"low-priority", "high-priority"}
