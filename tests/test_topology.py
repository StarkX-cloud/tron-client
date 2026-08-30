"""Tests for the Phase 2 topology map: latency recording, aging, and
placement ranking.
"""
import pytest

from tron.spine import TopologyMap


@pytest.fixture
def topology():
    return TopologyMap(smoothing=0.5, max_age_seconds=60.0)


def test_unmeasured_latency_is_none(topology):
    assert topology.latency("master", "worker-1") is None


def test_record_and_read_latency(topology):
    topology.record_latency("master", "worker-1", 10.0)
    assert topology.latency("master", "worker-1") == 10.0


def test_latency_is_directional(topology):
    topology.record_latency("master", "worker-1", 10.0)
    assert topology.latency("worker-1", "master") is None


def test_repeated_samples_smooth_via_ewma(topology):
    # smoothing=0.5: each new sample pulls the estimate halfway toward it.
    topology.record_latency("master", "worker-1", 10.0)
    topology.record_latency("master", "worker-1", 30.0)
    assert topology.latency("master", "worker-1") == pytest.approx(20.0)


def test_rejects_negative_latency(topology):
    with pytest.raises(ValueError):
        topology.record_latency("master", "worker-1", -5.0)


def test_stale_measurement_ages_out(topology):
    topology.record_latency("master", "worker-1", 10.0, timestamp=1000.0)
    # 61 seconds later, past max_age_seconds=60
    assert topology.latency("master", "worker-1", now=1061.0) is None


def test_fresh_measurement_within_max_age(topology):
    topology.record_latency("master", "worker-1", 10.0, timestamp=1000.0)
    assert topology.latency("master", "worker-1", now=1059.0) == 10.0


def test_rank_nodes_orders_by_ascending_latency(topology):
    topology.record_latency("master", "fast", 5.0)
    topology.record_latency("master", "slow", 200.0)
    topology.record_latency("master", "medium", 50.0)

    ranked = topology.rank_nodes("master", ["slow", "fast", "medium"])
    assert ranked == ["fast", "medium", "slow"]


def test_rank_nodes_places_unmeasured_in_the_middle(topology):
    topology.record_latency("master", "fast", 5.0)
    topology.record_latency("master", "slow", 200.0)

    ranked = topology.rank_nodes("master", ["slow", "fast", "new-node"])
    assert ranked == ["fast", "new-node", "slow"]


def test_rank_nodes_handles_all_unmeasured(topology):
    ranked = topology.rank_nodes("master", ["a", "b", "c"])
    assert set(ranked) == {"a", "b", "c"}
    assert len(ranked) == 3


def test_invalid_smoothing_rejected():
    with pytest.raises(ValueError):
        TopologyMap(smoothing=0.0)
    with pytest.raises(ValueError):
        TopologyMap(smoothing=1.5)
