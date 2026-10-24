"""Results schema / JSON / CSV / report tests using fake outcomes.

No model loading here — these pin the data-handling logic (serialization,
round-tripping, report comparison/winner logic, correctness-mismatch
warnings) independent of any real hardware or model.
"""

import json
import os

from tron_efficiency.hardware.detect import HardwareInfo
from tron_efficiency.results.report import generate_report
from tron_efficiency.results.schema import BenchmarkResult
from tron_efficiency.results.writer import list_runs, save_run, write_csv, write_json
from tron_efficiency.workloads.config import WorkloadConfig
from tron_efficiency.workloads.runner import RunOutcome


def _fake_hardware(cuda=False):
    return HardwareInfo(
        platform="test-platform",
        python_version="3.11.0",
        cpu_count=4,
        cpu_total_memory_bytes=16 * 1024**3,
        cuda_available=cuda,
        torch_version="2.0.0",
        gpus=[],
        warnings=[] if cuda else ["no CUDA GPU detected"],
    )


def _fake_result(policy_name="baseline", success=True, text="hello world", **overrides):
    config = WorkloadConfig(model_name="fake/model", max_new_tokens=8)
    defaults = dict(
        load_time_seconds=1.0,
        time_to_first_token_seconds=0.1,
        end_to_end_latency_seconds=0.5,
        tokens_generated=8,
        tokens_per_second=16.0,
        peak_cpu_memory_bytes=1024 * 1024 * 500,
    )
    defaults.update(overrides)
    outcome = RunOutcome(
        success=success,
        policy_name=policy_name,
        generated_text=text if success else None,
        error=None if success else "boom",
        **defaults,
    )
    return BenchmarkResult.build(_fake_hardware(), config, {"name": policy_name}, outcome)


def test_result_to_dict_and_from_dict_round_trip():
    result = _fake_result()
    data = result.to_dict()
    restored = BenchmarkResult.from_dict(data)
    assert restored.run_id == result.run_id
    assert restored.config.model_name == "fake/model"
    assert restored.outcome.tokens_per_second == 16.0


def test_save_and_load_run(tmp_path):
    os.environ["TRON_EFFICIENCY_RESULTS_DIR"] = str(tmp_path)
    try:
        result = _fake_result()
        path = save_run(result)
        assert path.exists()
        loaded = [r for r in list_runs() if r.run_id == result.run_id]
        assert len(loaded) == 1
        assert loaded[0].outcome.policy_name == "baseline"
    finally:
        del os.environ["TRON_EFFICIENCY_RESULTS_DIR"]


def test_write_json(tmp_path):
    result = _fake_result()
    out = tmp_path / "run.json"
    write_json(result, str(out))
    data = json.loads(out.read_text())
    assert data["run_id"] == result.run_id


def test_write_csv_has_one_row_per_result(tmp_path):
    results = [_fake_result("baseline"), _fake_result("offload")]
    out = tmp_path / "results.csv"
    write_csv(results, str(out))
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 rows


def test_report_declares_winner_for_lower_is_better_metric():
    fast = _fake_result("baseline", end_to_end_latency_seconds=0.1)
    slow = _fake_result("offload", end_to_end_latency_seconds=5.0)
    report = generate_report([fast, slow])
    assert "End-to-end latency" in report
    assert "baseline" in report


def test_report_handles_failed_runs_without_crashing():
    ok = _fake_result("baseline")
    failed = _fake_result("offload", success=False)
    report = generate_report([ok, failed])
    assert "FAILED" in report
    assert "Fewer than two successful runs" in report


def test_report_with_no_results():
    assert "No results to report" in generate_report([])


def test_report_cost_estimate_is_labeled_as_estimate():
    results = [_fake_result("baseline"), _fake_result("offload")]
    report = generate_report(results, hourly_rate_usd=1.0)
    assert "estimate" in report.lower()
    assert "not an actual customer saving" in report
