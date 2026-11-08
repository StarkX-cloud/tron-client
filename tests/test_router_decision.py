"""Tests for tron_router/decision.py's placement logic -- pure, via
dependency injection (fake HardwareInfo, fake/injected PoolStatus), no
network calls and no live queue_server needed.
"""

import pytest

from tron_efficiency.hardware.detect import HardwareInfo
from tron_router import decision as decision_module
from tron_router.decision import (
    DEFAULT_MASTER_URL,
    JobRequest,
    PoolStatus,
    decide_placement,
)


# Captured before any test's autouse monkeypatch below replaces the
# module attribute -- a direct reference to the real function, so the
# unit tests for estimate_required_memory_gb itself (bottom of this file)
# can still exercise it even though every other test in this file gets
# the fast fake via the autouse fixture.
_REAL_ESTIMATE_REQUIRED_MEMORY_GB = decision_module.estimate_required_memory_gb


@pytest.fixture(autouse=True)
def _no_real_model_size_lookup(monkeypatch):
    """estimate_required_memory_gb (called for every 'pooled' decision)
    would otherwise fetch a real HF config.json over the network for a
    model name that doesn't exist in most of these tests -- this module's
    own docstring promises "no network calls," so every test gets a fast
    fake by default. Tests that actually exercise this wiring override it
    explicitly below (same pattern as the `recommend` monkeypatches)."""
    monkeypatch.setattr(decision_module, "estimate_required_memory_gb", lambda model_name, dtype="fp16": 8.0)


def _fake_hardware(ram_gb=16, cuda=False):
    return HardwareInfo(
        platform="fake", python_version="3.11", cpu_count=8,
        cpu_total_memory_bytes=ram_gb * 1024**3, cuda_available=cuda,
        torch_version="2.x", gpus=[], warnings=[],
    )


def _fake_recommend(fits_without_offload=True, confidence="high"):
    """Monkeypatch target for tron_efficiency.advisor.recommend.recommend,
    imported lazily inside decide_placement -- patch where it's looked up."""
    from tron_efficiency.advisor.recommend import Recommendation

    def fake(model_name, hardware=None, gpu_budget_bytes_override=None, size_estimate=None):
        return Recommendation(
            model_name=model_name, estimated_param_count=70_000_000,
            param_count_is_estimate=True, device="cpu", dtype="fp32",
            policy="baseline" if fits_without_offload else "offload",
            memory_budget_bytes=None, confidence=confidence,
            rationale=["stub"], warnings=[],
        )

    return fake


def test_fits_locally_recommends_local(monkeypatch):
    import importlib
    recommend_module = importlib.import_module("tron_efficiency.advisor.recommend")

    monkeypatch.setattr(recommend_module, "recommend", _fake_recommend(fits_without_offload=True))
    decision = decide_placement(
        JobRequest(model_name="fake/small-model"), hardware=_fake_hardware()
    )
    assert decision.target == "local"
    assert decision.confidence == "high"
    assert decision.pool_status is None  # never needed to check the pool


def test_does_not_fit_locally_checks_pool_and_recommends_pooled_when_healthy(monkeypatch):
    import importlib
    recommend_module = importlib.import_module("tron_efficiency.advisor.recommend")

    monkeypatch.setattr(recommend_module, "recommend", _fake_recommend(fits_without_offload=False))
    healthy_pool = PoolStatus(
        reachable=True, master_url=DEFAULT_MASTER_URL,
        alive_worker_count=3, total_reported_memory_gb=48.0, checked_worker_count=3,
    )
    decision = decide_placement(
        JobRequest(model_name="fake/big-model", tolerant=True),
        hardware=_fake_hardware(), pool_status=healthy_pool,
    )
    assert decision.target == "pooled"
    assert decision.confidence == "medium"
    assert decision.pool_status["alive_worker_count"] == 3


def test_unreachable_pool_recommends_dedicated(monkeypatch):
    import importlib
    recommend_module = importlib.import_module("tron_efficiency.advisor.recommend")

    monkeypatch.setattr(recommend_module, "recommend", _fake_recommend(fits_without_offload=False))
    unreachable = PoolStatus(
        reachable=False, master_url=DEFAULT_MASTER_URL, error="ConnectionError: refused"
    )
    decision = decide_placement(
        JobRequest(model_name="fake/big-model"), hardware=_fake_hardware(), pool_status=unreachable
    )
    assert decision.target == "dedicated"
    assert decision.confidence == "low"
    assert any("no live pool reachable" in w for w in decision.warnings)


def test_reachable_but_empty_pool_recommends_dedicated(monkeypatch):
    import importlib
    recommend_module = importlib.import_module("tron_efficiency.advisor.recommend")

    monkeypatch.setattr(recommend_module, "recommend", _fake_recommend(fits_without_offload=False))
    empty_pool = PoolStatus(reachable=True, master_url=DEFAULT_MASTER_URL, alive_worker_count=0)
    decision = decide_placement(
        JobRequest(model_name="fake/big-model"), hardware=_fake_hardware(), pool_status=empty_pool
    )
    assert decision.target == "dedicated"
    assert any("0 alive workers" in w for w in decision.warnings)


def test_intolerant_job_never_goes_to_the_pool_even_if_healthy(monkeypatch):
    import importlib
    recommend_module = importlib.import_module("tron_efficiency.advisor.recommend")

    monkeypatch.setattr(recommend_module, "recommend", _fake_recommend(fits_without_offload=False))
    healthy_pool = PoolStatus(
        reachable=True, master_url=DEFAULT_MASTER_URL,
        alive_worker_count=5, total_reported_memory_gb=100.0, checked_worker_count=5,
    )
    decision = decide_placement(
        JobRequest(model_name="fake/big-model", tolerant=False),
        hardware=_fake_hardware(), pool_status=healthy_pool,
    )
    assert decision.target == "dedicated"
    assert any("intolerant" in w for w in decision.warnings)


def test_single_alive_worker_is_low_confidence(monkeypatch):
    import importlib
    recommend_module = importlib.import_module("tron_efficiency.advisor.recommend")

    monkeypatch.setattr(recommend_module, "recommend", _fake_recommend(fits_without_offload=False))
    one_worker_pool = PoolStatus(
        reachable=True, master_url=DEFAULT_MASTER_URL,
        alive_worker_count=1, total_reported_memory_gb=16.0, checked_worker_count=1,
    )
    decision = decide_placement(
        JobRequest(model_name="fake/big-model"), hardware=_fake_hardware(), pool_status=one_worker_pool
    )
    assert decision.target == "pooled"
    assert decision.confidence == "low"
    assert any("little to actually optimize" in w for w in decision.warnings)


def test_pooled_decision_cites_real_measured_network_data_when_available(monkeypatch):
    import importlib
    recommend_module = importlib.import_module("tron_efficiency.advisor.recommend")

    monkeypatch.setattr(recommend_module, "recommend", _fake_recommend(fits_without_offload=False))
    pool_with_topology = PoolStatus(
        reachable=True, master_url=DEFAULT_MASTER_URL,
        alive_worker_count=2, total_reported_memory_gb=32.0, checked_worker_count=2,
        measured_links={
            "worker-a": {"latency_ms": 10.0, "bandwidth_mbps": 100.0},
            "worker-b": {"latency_ms": 30.0, "bandwidth_mbps": 50.0},
        },
    )
    decision = decide_placement(
        JobRequest(model_name="fake/big-model"), hardware=_fake_hardware(), pool_status=pool_with_topology
    )
    assert decision.target == "pooled"
    assert any("avg latency 20.0ms" in r and "avg bandwidth 75.0Mbps" in r for r in decision.rationale)
    assert not any("could not read real measured" in w for w in decision.warnings)


def test_pooled_decision_warns_when_no_worker_has_been_measured_yet(monkeypatch):
    import importlib
    recommend_module = importlib.import_module("tron_efficiency.advisor.recommend")

    monkeypatch.setattr(recommend_module, "recommend", _fake_recommend(fits_without_offload=False))
    pool_unmeasured = PoolStatus(
        reachable=True, master_url=DEFAULT_MASTER_URL,
        alive_worker_count=2, total_reported_memory_gb=32.0, checked_worker_count=2,
        measured_links={
            "worker-a": {"latency_ms": None, "bandwidth_mbps": None},
            "worker-b": {"latency_ms": None, "bandwidth_mbps": None},
        },
    )
    decision = decide_placement(
        JobRequest(model_name="fake/big-model"), hardware=_fake_hardware(), pool_status=pool_unmeasured
    )
    assert decision.target == "pooled"
    assert any("no alive worker has a real measured" in w for w in decision.warnings)


def test_pooled_decision_warns_when_topology_endpoint_was_unreachable(monkeypatch):
    import importlib
    recommend_module = importlib.import_module("tron_efficiency.advisor.recommend")

    monkeypatch.setattr(recommend_module, "recommend", _fake_recommend(fits_without_offload=False))
    pool_no_topology = PoolStatus(
        reachable=True, master_url=DEFAULT_MASTER_URL,
        alive_worker_count=2, total_reported_memory_gb=32.0, checked_worker_count=2,
        measured_links=None,
    )
    decision = decide_placement(
        JobRequest(model_name="fake/big-model"), hardware=_fake_hardware(), pool_status=pool_no_topology
    )
    assert decision.target == "pooled"
    assert any("could not read real measured" in w for w in decision.warnings)


def test_requires_multi_node_skips_local_check_entirely(monkeypatch):
    import importlib
    recommend_module = importlib.import_module("tron_efficiency.advisor.recommend")

    calls = []

    def tracking_recommend(*args, **kwargs):
        calls.append(1)
        return _fake_recommend(fits_without_offload=True)(*args, **kwargs)

    monkeypatch.setattr(recommend_module, "recommend", tracking_recommend)
    healthy_pool = PoolStatus(
        reachable=True, master_url=DEFAULT_MASTER_URL,
        alive_worker_count=2, total_reported_memory_gb=32.0, checked_worker_count=2,
    )
    decision = decide_placement(
        JobRequest(model_name="fake/model", requires_multi_node=True),
        hardware=_fake_hardware(), pool_status=healthy_pool,
    )
    assert decision.target == "pooled"
    assert decision.local_recommendation is None
    assert not calls  # local advisor was never consulted


# -- required_memory_gb: feeds tron/spine/matcher.py's hard memory-fit
# filter once the job reaches the real matcher (see tron_router/handoff.py).


def test_pooled_decision_carries_required_memory_gb(monkeypatch):
    import importlib
    recommend_module = importlib.import_module("tron_efficiency.advisor.recommend")

    monkeypatch.setattr(recommend_module, "recommend", _fake_recommend(fits_without_offload=False))
    monkeypatch.setattr(decision_module, "estimate_required_memory_gb", lambda model_name, dtype="fp16": 12.3)
    healthy_pool = PoolStatus(
        reachable=True, master_url=DEFAULT_MASTER_URL,
        alive_worker_count=2, total_reported_memory_gb=32.0, checked_worker_count=2,
    )
    decision = decide_placement(
        JobRequest(model_name="fake/big-model"), hardware=_fake_hardware(), pool_status=healthy_pool,
    )
    assert decision.target == "pooled"
    assert decision.required_memory_gb == 12.3


def test_pooled_decision_warns_when_memory_estimate_is_unavailable(monkeypatch):
    import importlib
    recommend_module = importlib.import_module("tron_efficiency.advisor.recommend")

    monkeypatch.setattr(recommend_module, "recommend", _fake_recommend(fits_without_offload=False))
    monkeypatch.setattr(decision_module, "estimate_required_memory_gb", lambda model_name, dtype="fp16": None)
    healthy_pool = PoolStatus(
        reachable=True, master_url=DEFAULT_MASTER_URL,
        alive_worker_count=2, total_reported_memory_gb=32.0, checked_worker_count=2,
    )
    decision = decide_placement(
        JobRequest(model_name="fake/big-model"), hardware=_fake_hardware(), pool_status=healthy_pool,
    )
    assert decision.target == "pooled"
    assert decision.required_memory_gb is None
    assert any("could not estimate" in w for w in decision.warnings)


def test_required_memory_gb_uses_local_recommendations_dtype(monkeypatch):
    import importlib
    recommend_module = importlib.import_module("tron_efficiency.advisor.recommend")

    monkeypatch.setattr(recommend_module, "recommend", _fake_recommend(fits_without_offload=False))
    captured = {}

    def fake_estimate(model_name, dtype="fp16"):
        captured["dtype"] = dtype
        return 5.0

    monkeypatch.setattr(decision_module, "estimate_required_memory_gb", fake_estimate)
    healthy_pool = PoolStatus(
        reachable=True, master_url=DEFAULT_MASTER_URL,
        alive_worker_count=2, total_reported_memory_gb=32.0, checked_worker_count=2,
    )
    decide_placement(
        JobRequest(model_name="fake/big-model"), hardware=_fake_hardware(), pool_status=healthy_pool,
    )
    # _fake_recommend always returns dtype="fp32" -- confirms decide_placement
    # actually reads local_rec.dtype rather than hard-coding a default.
    assert captured["dtype"] == "fp32"


def test_required_memory_gb_defaults_to_fp16_when_multi_node_skips_local_check(monkeypatch):
    import importlib
    recommend_module = importlib.import_module("tron_efficiency.advisor.recommend")

    monkeypatch.setattr(recommend_module, "recommend", _fake_recommend(fits_without_offload=True))
    captured = {}

    def fake_estimate(model_name, dtype="fp16"):
        captured["dtype"] = dtype
        return 5.0

    monkeypatch.setattr(decision_module, "estimate_required_memory_gb", fake_estimate)
    healthy_pool = PoolStatus(
        reachable=True, master_url=DEFAULT_MASTER_URL,
        alive_worker_count=2, total_reported_memory_gb=32.0, checked_worker_count=2,
    )
    decide_placement(
        JobRequest(model_name="fake/model", requires_multi_node=True),
        hardware=_fake_hardware(), pool_status=healthy_pool,
    )
    assert captured["dtype"] == "fp16"


# -- estimate_required_memory_gb itself, as a unit -- pure, via dependency
# injection on tron_efficiency.advisor.model_info.estimate_model_size
# (patched at its source, the same pattern _fake_recommend above uses for
# tron_efficiency.advisor.recommend.recommend), so this never touches the
# network either.


def _fake_model_size_estimate(param_count, bytes_by_dtype=None):
    import importlib
    model_info_module = importlib.import_module("tron_efficiency.advisor.model_info")

    if bytes_by_dtype is None:
        bytes_by_dtype = {
            dtype: int(param_count * bpp)
            for dtype, bpp in model_info_module.BYTES_PER_PARAM.items()
        }
    return model_info_module.ModelSizeEstimate(
        model_name="fake/model", param_count=param_count, param_count_is_estimate=True,
        architecture="FakeArch", bytes_by_dtype=bytes_by_dtype, warnings=[],
    )


def test_estimate_required_memory_gb_converts_bytes_to_gb(monkeypatch):
    import importlib
    model_info_module = importlib.import_module("tron_efficiency.advisor.model_info")

    estimate = _fake_model_size_estimate(70_000_000)
    monkeypatch.setattr(model_info_module, "estimate_model_size", lambda model_name: estimate)

    gb = _REAL_ESTIMATE_REQUIRED_MEMORY_GB("fake/model", dtype="fp16")
    assert gb == pytest.approx(estimate.bytes_by_dtype["fp16"] / 1_000_000_000.0)


def test_estimate_required_memory_gb_returns_none_when_param_count_unknown(monkeypatch):
    import importlib
    model_info_module = importlib.import_module("tron_efficiency.advisor.model_info")

    estimate = _fake_model_size_estimate(0, bytes_by_dtype={})
    monkeypatch.setattr(model_info_module, "estimate_model_size", lambda model_name: estimate)

    assert _REAL_ESTIMATE_REQUIRED_MEMORY_GB("fake/model") is None


def test_estimate_required_memory_gb_returns_none_on_fetch_failure(monkeypatch):
    import importlib
    model_info_module = importlib.import_module("tron_efficiency.advisor.model_info")

    def raises(model_name):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(model_info_module, "estimate_model_size", raises)

    assert _REAL_ESTIMATE_REQUIRED_MEMORY_GB("fake/model") is None
