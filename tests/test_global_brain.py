"""Tests for the Phase 2 real GlobalDecisionBrain: placement scoring driven
by measured topology instead of a flat pass-through.
"""
import pytest

from tron.spine import TopologyMap
from tron_runtime.global_brain import GlobalDecisionBrain


@pytest.fixture
def topology():
    return TopologyMap()


@pytest.fixture
def brain(topology):
    return GlobalDecisionBrain(topology)


def test_no_worker_name_returns_bare_priority(brain):
    decision = brain.decide([], {}, {"priority": 5})
    assert decision["score"] == 5.0
    assert decision["latency_ms"] is None


def test_unmeasured_worker_gets_no_penalty(brain):
    decision = brain.decide([], {}, {"priority": 5}, worker_name="worker-1")
    assert decision["score"] == 5.0
    assert decision["latency_ms"] is None


def test_measured_latency_reduces_score(brain, topology):
    topology.record_latency("master", "worker-1", 200.0)
    decision = brain.decide([], {}, {"priority": 5}, worker_name="worker-1")
    assert decision["score"] < 5.0
    assert decision["latency_ms"] == 200.0


def test_lower_latency_worker_scores_higher_for_same_job(brain, topology):
    topology.record_latency("master", "near", 5.0)
    topology.record_latency("master", "far", 300.0)

    near_decision = brain.decide([], {}, {"priority": 3}, worker_name="near")
    far_decision = brain.decide([], {}, {"priority": 3}, worker_name="far")

    assert near_decision["score"] > far_decision["score"]


def test_default_priority_is_one(brain):
    decision = brain.decide([], {}, {})
    assert decision["score"] == 1.0


def test_penalty_scales_with_job_compute_weight(brain, topology):
    # This is the property that actually matters for scheduling: /next_job
    # picks the best job out of a worker's *own* queue in one call, so a
    # penalty that's the same for every job in that call can never change
    # which job wins (it shifts every score equally). The penalty has to
    # differ per job — here, by how heavy the job is — to have any effect
    # on the outcome. See tron_runtime/global_brain.py's decide() docstring.
    topology.record_latency("master", "worker-1", 100.0)

    light = brain.decide([], {}, {"priority": 5, "compute_weight": 1}, worker_name="worker-1")
    heavy = brain.decide([], {}, {"priority": 5, "compute_weight": 20}, worker_name="worker-1")

    assert light["score"] > heavy["score"]


def test_zero_latency_makes_compute_weight_irrelevant(brain, topology):
    topology.record_latency("master", "worker-1", 0.0)

    light = brain.decide([], {}, {"priority": 5, "compute_weight": 1}, worker_name="worker-1")
    heavy = brain.decide([], {}, {"priority": 5, "compute_weight": 20}, worker_name="worker-1")

    assert light["score"] == heavy["score"] == 5.0
