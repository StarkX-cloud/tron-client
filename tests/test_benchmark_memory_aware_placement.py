"""Tests for benchmarks/memory_aware_placement.py — the real validation
harness for the memory-aware-placement feature (ROADMAP.md). Deterministic
via a fixed seed; no real network or GPU involved.
"""

from benchmarks.memory_aware_placement import (
    PYTHIA_70M_REAL_PARAM_COUNT,
    _make_jobs,
    _make_pool,
    _would_oom,
    generate_estimate_accuracy_report,
    generate_report,
    run_benchmark,
)


def test_would_oom_true_when_job_exceeds_worker_capacity():
    pool = _make_pool([8])
    worker_name = next(iter(pool))
    job = {"required_memory_gb": 16}
    assert _would_oom(job, worker_name, pool) is True


def test_would_oom_false_when_job_fits():
    pool = _make_pool([32])
    worker_name = next(iter(pool))
    job = {"required_memory_gb": 16}
    assert _would_oom(job, worker_name, pool) is False


def test_make_jobs_is_deterministic_for_a_fixed_seed():
    a = _make_jobs(50, seed=1)
    b = _make_jobs(50, seed=1)
    assert a == b


def test_make_jobs_different_seeds_can_differ():
    a = _make_jobs(50, seed=1)
    b = _make_jobs(50, seed=2)
    assert a != b


def test_memory_aware_run_never_produces_a_would_oom_placement():
    # The actual property this feature exists to guarantee — run many
    # seeds/pool shapes so this isn't a single lucky draw.
    for seed in range(5):
        results = run_benchmark(n_jobs=100, trials=1, pool_gb=[4, 8, 16, 32], seed=seed)
        aware = next(r for r in results if "aware" in r.label)
        assert aware.n_would_oom == 0


def test_memory_blind_baseline_can_produce_would_oom_placements():
    # Confirms the benchmark's baseline condition is actually exercising
    # the gap (a benchmark that never finds a real difference proves
    # nothing) -- at least one of several seeds/pool shapes should trip it.
    found_any = False
    for seed in range(10):
        results = run_benchmark(n_jobs=200, trials=1, pool_gb=[4, 8, 8, 16, 16, 32, 32], seed=seed)
        blind = next(r for r in results if "blind" in r.label)
        if blind.n_would_oom > 0:
            found_any = True
            break
    assert found_any, "no seed in range produced a would-OOM baseline placement -- benchmark may not be exercising the real gap"


def test_report_contains_both_modes_and_the_would_oom_numbers():
    results = run_benchmark(n_jobs=200, trials=2, pool_gb=[4, 8, 8, 16, 16, 32, 32], seed=3)
    report = generate_report(results, [4, 8, 8, 16, 16, 32, 32], trials=2)
    assert "memory-blind" in report
    assert "memory-aware" in report
    assert "Would-OOM" in report


def test_report_labels_overhead_within_noise_honestly_when_delta_is_small():
    # Construct results where the timing delta is smaller than combined
    # stdev -- the report must say "not distinguishable from zero", not
    # quote a false-precision number.
    from benchmarks.memory_aware_placement import RunResult

    blind = RunResult(label="memory-blind (x)", n_jobs=10, n_assigned=5, n_would_oom=1,
                       n_unassigned=5, mean_seconds=0.010, stdev_seconds=0.005)
    aware = RunResult(label="memory-aware (x)", n_jobs=10, n_assigned=5, n_would_oom=0,
                       n_unassigned=5, mean_seconds=0.0105, stdev_seconds=0.005)
    report = generate_report([blind, aware], [8, 16], trials=10)
    assert "not distinguishable from zero overhead" in report


def test_estimate_accuracy_report_matches_known_real_error():
    report = generate_estimate_accuracy_report()
    assert "pythia-70m" in report
    assert f"{PYTHIA_70M_REAL_PARAM_COUNT:,}" in report
    assert "0.04" in report  # the real ~0.04% error this formula gets on this architecture
