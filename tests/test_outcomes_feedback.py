"""Closing the outcomes feedback loop: outcomes were being recorded from
every run (numpy and LoRA, both in-process and over-the-wire) but nothing
read them back to influence future estimates or placement. This covers
the three pieces that now do:

  A. estimate refinement  — TrainingOrchestrator's hardcoded per-adapter
     capability/cost guesses are overridden by an adapter's own history
     once it has run at least once.
  B. placement feedback   — tron/spine/matcher.py can bias assignment
     toward nodes with a good training-outcome track record, on top of
     (not instead of) the measured latency/bandwidth terms.
  C. module specificity   — outcome stats are keyed by (adapter, module),
     not just adapter, so an adapter's mixed record across task types
     doesn't get flattened into one number.
"""
import pytest

from tron.orchestrator.outcomes import OutcomeLog, TrainingOutcome
from tron.orchestrator.policy import TrainingPolicy
from tron.orchestrator.orchestrator import TrainingOrchestrator
from tron.spine import TopologyMap, match_jobs_to_workers, score_pair


def _outcome(adapter_name, module_id, gain, cost, success, node_ids=None):
    return TrainingOutcome(
        artifact_id=f"{adapter_name}-{module_id}-{gain}",
        adapter_name=adapter_name,
        module_id=module_id,
        expected_capability_gain=gain,
        actual_capability_gain=gain,
        expected_cost=cost,
        actual_cost=cost,
        success=success,
        timestamp="2026-01-01T00:00:00Z",
        node_ids=node_ids or [],
    )


# -- C: module-specific stats on OutcomeLog ---------------------------------


def test_pair_accuracy_is_none_with_no_history():
    log = OutcomeLog()
    assert log.pair_accuracy("ray", "vision") is None
    assert log.pair_success_rate("ray", "vision") is None


def test_pair_stats_do_not_mix_across_modules():
    """The same adapter can be reliable for one module and not another —
    pair_success_rate must see only that module's outcomes, not every
    outcome for the adapter."""
    log = OutcomeLog()
    log.record(_outcome("transformers", "nlp", gain=0.9, cost=1.0, success=True))
    log.record(_outcome("transformers", "nlp", gain=0.8, cost=1.0, success=True))
    log.record(_outcome("transformers", "vision", gain=0.0, cost=1.0, success=False))

    assert log.pair_success_rate("transformers", "nlp") == 1.0
    assert log.pair_success_rate("transformers", "vision") == 0.0
    # adapter-wide is still the flattened, mixed number
    assert log.adapter_success_rate("transformers") == pytest.approx(2 / 3)


# -- A: estimate refinement --------------------------------------------------


def test_estimate_capability_gain_prefers_pair_over_adapter_wide():
    log = OutcomeLog()
    log.record(_outcome("ray", "nlp", gain=1.0, cost=2.0, success=True))
    log.record(_outcome("ray", "vision", gain=0.2, cost=2.0, success=True))

    # module-specific estimate for "nlp" ignores the "vision" outcome
    assert log.estimate_capability_gain("ray", "nlp") == pytest.approx(1.0)
    # with no module given, or a module never seen, falls back to adapter-wide
    assert log.estimate_capability_gain("ray") == pytest.approx(0.6)
    assert log.estimate_capability_gain("ray", "audio") == pytest.approx(0.6)


def test_estimate_is_none_for_a_never_seen_adapter():
    log = OutcomeLog()
    assert log.estimate_capability_gain("nonexistent-adapter") is None
    assert log.estimate_cost("nonexistent-adapter") is None


def test_orchestrator_uses_history_over_hardcoded_default():
    """_estimate_capability_gain hardcodes ray at 0.9. Once real outcomes
    say ray actually delivers ~0.3 for this module, the estimate used for
    the next adapter-selection decision should reflect that, not the
    original guess."""
    orch = TrainingOrchestrator()
    # sanity: the hardcoded prior, with no history yet
    assert orch._estimate_capability_gain("ray", plan=None, module_id="rl-task") == 0.9

    orch.outcome_log.record(_outcome("ray", "rl-task", gain=0.3, cost=0.5, success=True))
    orch.outcome_log.record(_outcome("ray", "rl-task", gain=0.3, cost=0.5, success=True))

    assert orch._estimate_capability_gain("ray", plan=None, module_id="rl-task") == pytest.approx(0.3)
    assert orch._estimate_adapter_cost("ray", substrate_name=None, module_id="rl-task") == pytest.approx(0.5)

    # a different, never-seen module falls back to ray's adapter-wide
    # history (still 0.3, its only outcome so far) rather than straight to
    # the hardcoded prior — same pair -> adapter-wide -> hardcoded chain
    # OutcomeLog.estimate_capability_gain documents.
    assert orch._estimate_capability_gain("ray", plan=None, module_id="other-task") == pytest.approx(0.3)

    # an adapter that has never run at all still gets its hardcoded prior
    assert orch._estimate_capability_gain("sb3", plan=None, module_id="rl-task") == 1.0


# -- C continued: policy scoring uses pair stats -----------------------------


def test_policy_prefers_pair_accuracy_when_available():
    log = OutcomeLog()
    # "ray" has a perfect record for "nlp" but a terrible one adapter-wide
    log.record(_outcome("ray", "nlp", gain=1.0, cost=1.0, success=True))
    log.record(_outcome("ray", "other", gain=0.0, cost=1.0, success=False))
    log.record(_outcome("ray", "other", gain=0.0, cost=1.0, success=False))

    policy = TrainingPolicy(outcome_log=log)
    estimate = {"capability_gain": 1.0, "cost": 1.0}

    score_with_module = policy.score_estimate(estimate, None, "ray", module_id="nlp")
    score_without_module = policy.score_estimate(estimate, None, "ray", module_id=None)

    # scoped to "nlp" (perfect record) should score higher than the
    # adapter-wide number, which is dragged down by the "other" failures
    assert score_with_module > score_without_module


# -- B: placement feedback in the matcher ------------------------------------


def test_node_quality_is_none_for_an_untouched_node():
    log = OutcomeLog()
    log.record(_outcome("sess-1", "distributed-numpy_mlp", gain=0.1, cost=1.0, success=True, node_ids=["shard-0"]))
    assert log.node_quality("shard-1") is None


def test_node_quality_reflects_success_rate_of_runs_it_took_part_in():
    log = OutcomeLog()
    log.record(_outcome("sess-1", "m", gain=0.1, cost=1.0, success=True, node_ids=["shard-0", "shard-1"]))
    log.record(_outcome("sess-2", "m", gain=-0.1, cost=1.0, success=False, node_ids=["shard-0"]))

    assert log.node_quality("shard-0") == pytest.approx(0.5)  # 1 success, 1 failure
    assert log.node_quality("shard-1") == pytest.approx(1.0)  # only ever in the successful run


def test_score_pair_unaffected_by_outcome_log_when_node_never_seen():
    """Same 'zero unless known' contract as the bandwidth term: a node
    with no recorded outcomes scores exactly as it did before outcome
    feedback existed."""
    topology = TopologyMap()
    log = OutcomeLog()
    job = {"priority": 5, "compute_weight": 1}
    without = score_pair(job, "worker-1", {}, topology)
    with_log = score_pair(job, "worker-1", {}, topology, outcome_log=log)
    assert without == with_log


def test_match_jobs_to_workers_prefers_the_node_with_the_better_track_record():
    """Two workers, identical latency — nothing to separate them on
    network terms alone. One has a track record of successful training
    outcomes, the other of failures. The match should steer the job to
    the proven node."""
    topology = TopologyMap()
    topology.record_latency("master", "reliable", 10.0)
    topology.record_latency("master", "flaky", 10.0)

    log = OutcomeLog()
    for _ in range(3):
        log.record(_outcome("s", "m", gain=0.5, cost=1.0, success=True, node_ids=["reliable"]))
    for _ in range(3):
        log.record(_outcome("s", "m", gain=-0.5, cost=1.0, success=False, node_ids=["flaky"]))

    jobs = [{"id": "job-1", "priority": 5, "compute_weight": 1}]
    workers = {"reliable": {}, "flaky": {}}

    assignments = dict(match_jobs_to_workers(jobs, workers, topology, outcome_log=log))
    assert assignments["job-1"] == "reliable"


# -- round trip / persistence -------------------------------------------------


def test_node_ids_survive_a_save_load_round_trip(tmp_path):
    path = tmp_path / "outcomes.json"
    log = OutcomeLog(storage_path=str(path))
    log.record(_outcome("s", "m", gain=0.1, cost=1.0, success=True, node_ids=["shard-0", "shard-1"]))
    log.save()

    reloaded = OutcomeLog(storage_path=str(path))
    assert reloaded.outcomes[0].node_ids == ["shard-0", "shard-1"]


def test_loading_an_outcome_predating_node_ids_defaults_to_empty_list(tmp_path):
    """outcomes.json files written before this field existed have no
    "node_ids" key at all — loading one must not raise."""
    import json

    path = tmp_path / "old_outcomes.json"
    path.write_text(json.dumps([{
        "artifact_id": "a1", "adapter_name": "ray", "module_id": "m",
        "expected_capability_gain": 0.5, "actual_capability_gain": 0.5,
        "expected_cost": 1.0, "actual_cost": 1.0, "success": True,
        "timestamp": "2025-01-01T00:00:00Z",
    }]))

    log = OutcomeLog(storage_path=str(path))
    assert log.outcomes[0].node_ids == []
