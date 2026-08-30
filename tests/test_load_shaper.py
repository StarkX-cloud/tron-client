"""Tests for the real LoadShaper: a sliding-window byte budget per worker,
driven by measured bandwidth, that holds a transfer back rather than pile
it onto a saturated link.

The pass-through behaviour the old stub had must survive exactly: no
topology, no measured bandwidth, or a job with no transfer_bytes -> every
job releasable, queue order unchanged.
"""
import pytest

from tron.spine import TopologyMap
from tron_runtime.load_shaper import LoadShaper


@pytest.fixture
def topology():
    return TopologyMap()


def test_window_seconds_must_be_positive():
    with pytest.raises(ValueError):
        LoadShaper(window_seconds=0)


def test_capacity_is_none_when_bandwidth_unmeasured(topology):
    shaper = LoadShaper(window_seconds=2.0)
    assert shaper.capacity_bytes("w", topology) is None


def test_capacity_is_bandwidth_times_window(topology):
    shaper = LoadShaper(window_seconds=2.0)
    topology.record_bandwidth("master", "w", 10.0)  # 10 Mbps
    # 10e6 bits/s / 8 = 1.25 MB/s; * 2s window = 2.5 MB
    assert shaper.capacity_bytes("w", topology) == pytest.approx(2_500_000.0)


def test_can_accept_until_window_budget_is_spent(topology):
    shaper = LoadShaper(window_seconds=2.0)
    topology.record_bandwidth("master", "w", 10.0)  # cap 2.5 MB

    now = 1000.0
    assert shaper.can_accept("w", 1_000_000, topology, now=now) is True
    shaper.reserve("w", "job-1", 1_000_000, now=now)
    shaper.reserve("w", "job-2", 1_000_000, now=now)  # 2.0 MB in flight
    assert shaper.can_accept("w", 1_000_000, topology, now=now) is False  # 3.0 > 2.5
    assert shaper.can_accept("w", 400_000, topology, now=now) is True     # 2.4 <= 2.5


def test_budget_frees_as_the_window_slides(topology):
    shaper = LoadShaper(window_seconds=2.0)
    topology.record_bandwidth("master", "w", 10.0)

    shaper.reserve("w", "job-1", 2_000_000, now=1000.0)
    assert shaper.can_accept("w", 1_000_000, topology, now=1000.0) is False
    # 2.01s later the reservation has aged out of the window
    assert shaper.inflight_bytes("w", now=1002.01) == 0
    assert shaper.can_accept("w", 1_000_000, topology, now=1002.01) is True


def test_release_frees_budget_immediately(topology):
    shaper = LoadShaper(window_seconds=5.0)
    topology.record_bandwidth("master", "w", 10.0)  # cap 6.25 MB

    shaper.reserve("w", "job-1", 5_000_000, now=1000.0)
    assert shaper.can_accept("w", 2_000_000, topology, now=1000.0) is False
    shaper.release("job-1")
    assert shaper.can_accept("w", 2_000_000, topology, now=1000.0) is True


def test_reserve_is_idempotent_on_job_id(topology):
    shaper = LoadShaper(window_seconds=2.0)
    topology.record_bandwidth("master", "w", 10.0)
    shaper.reserve("w", "job-1", 1_000_000, now=1000.0)
    shaper.reserve("w", "job-1", 1_000_000, now=1000.0)  # same job again
    assert shaper.inflight_bytes("w", now=1000.0) == 1_000_000  # counted once


def test_reserve_moves_a_job_between_workers(topology):
    shaper = LoadShaper(window_seconds=2.0)
    shaper.reserve("w1", "job-1", 1_000_000, now=1000.0)
    shaper.reserve("w2", "job-1", 1_000_000, now=1000.0)
    assert shaper.inflight_bytes("w1", now=1000.0) == 0
    assert shaper.inflight_bytes("w2", now=1000.0) == 1_000_000


# -- reshape: pass-through unless there's something real to act on ---------

def test_reshape_without_topology_is_pass_through():
    shaper = LoadShaper()
    queue = [{"id": "a", "transfer_bytes": 10**9}, {"id": "b"}]
    shaped = shaper.reshape(queue, {"w": {"status": "idle"}})
    assert [e["delay"] for e in shaped] == [0, 0]
    assert [e["job"]["id"] for e in shaped] == ["a", "b"]


def test_reshape_ignores_jobs_without_transfer_bytes(topology):
    shaper = LoadShaper(window_seconds=2.0)
    topology.record_bandwidth("master", "w", 1.0)  # tiny pipe
    shaper.reserve("w", "filler", 10**9, now=1000.0)  # saturated
    shaped = shaper.reshape([{"id": "a"}], {"w": {"status": "idle"}}, topology, now=1000.0)
    assert shaped[0]["delay"] == 0


def test_reshape_holds_a_job_no_measured_idle_worker_has_room_for(topology):
    shaper = LoadShaper(window_seconds=2.0)
    topology.record_bandwidth("master", "slow", 8.0)  # cap 2.0 MB
    shaper.reserve("slow", "filler", 1_900_000, now=1000.0)

    workers = {"slow": {"status": "idle"}}
    big = [{"id": "big", "transfer_bytes": 1_000_000}]
    assert shaper.reshape(big, workers, topology, now=1000.0)[0]["delay"] == 1

    # a second, unmeasured idle worker gives the job somewhere to go
    workers["unmeasured"] = {"status": "idle"}
    assert shaper.reshape(big, workers, topology, now=1000.0)[0]["delay"] == 0


def test_reshape_puts_releasable_jobs_first(topology):
    shaper = LoadShaper(window_seconds=2.0)
    topology.record_bandwidth("master", "slow", 8.0)
    shaper.reserve("slow", "filler", 1_900_000, now=1000.0)
    workers = {"slow": {"status": "idle"}}

    queue = [
        {"id": "held", "transfer_bytes": 1_000_000},
        {"id": "ok", "transfer_bytes": 10_000},
    ]
    shaped = shaper.reshape(queue, workers, topology, now=1000.0)
    assert [e["job"]["id"] for e in shaped] == ["ok", "held"]
    assert [e["delay"] for e in shaped] == [0, 1]
