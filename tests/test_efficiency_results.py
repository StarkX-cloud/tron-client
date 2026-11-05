"""Results schema / JSON / CSV / report tests using fake outcomes.

No model loading here — these pin the data-handling logic (serialization,
round-tripping, report comparison/winner logic, correctness-mismatch
warnings) independent of any real hardware or model.
"""

import json
import os

from tron_efficiency.hardware.detect import HardwareInfo
from tron_efficiency.results.report import (
    generate_dtype_sweep_report,
    generate_matrix_report,
    generate_report,
    generate_sweep_report,
)
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


def _fake_result(policy_name="baseline", success=True, text="hello world", policy=None, dtype=None, error=None, **overrides):
    config = WorkloadConfig(model_name="fake/model", max_new_tokens=8, dtype=dtype)
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
        error=error if error is not None else (None if success else "boom"),
        **defaults,
    )
    return BenchmarkResult.build(_fake_hardware(), config, policy or {"name": policy_name}, outcome)


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


def _fake_sweep_result(budget_mb, end_to_end_latency_seconds, peak_cpu_memory_bytes):
    budget_bytes = budget_mb * 1024 * 1024
    return _fake_result(
        "offload",
        policy={"name": "offload", "memory_budget_bytes": budget_bytes},
        end_to_end_latency_seconds=end_to_end_latency_seconds,
        peak_cpu_memory_bytes=peak_cpu_memory_bytes,
    )


def test_sweep_report_labels_rows_by_budget_not_just_policy_name():
    results = [
        _fake_result("baseline"),
        _fake_sweep_result(100, 1.0, 500 * 1024 * 1024),
        _fake_sweep_result(50, 3.0, 300 * 1024 * 1024),
    ]
    report = generate_sweep_report(results)
    assert "baseline" in report
    assert "offload @ 100MB" in report
    assert "offload @ 50MB" in report


def test_sweep_report_orders_largest_budget_first():
    results = [
        _fake_sweep_result(50, 3.0, 300 * 1024 * 1024),
        _fake_sweep_result(200, 0.5, 700 * 1024 * 1024),
        _fake_sweep_result(100, 1.0, 500 * 1024 * 1024),
    ]
    report = generate_sweep_report(results)
    pos_200 = report.index("offload @ 200MB")
    pos_100 = report.index("offload @ 100MB")
    pos_50 = report.index("offload @ 50MB")
    assert pos_200 < pos_100 < pos_50


def test_sweep_report_baseline_sorts_before_budgets():
    results = [
        _fake_sweep_result(100, 1.0, 500 * 1024 * 1024),
        _fake_result("baseline"),
    ]
    report = generate_sweep_report(results)
    assert report.index("baseline") < report.index("offload @ 100MB")


def test_sweep_report_with_no_results():
    assert "No results to report" in generate_sweep_report([])


def test_sweep_report_all_failed_reports_no_trend():
    results = [_fake_result("baseline", success=False)]
    report = generate_sweep_report(results)
    assert "No successful runs in this sweep" in report


def test_dtype_sweep_report_labels_rows_by_dtype():
    results = [
        _fake_result("baseline", dtype="fp32", text="fp32 output"),
        _fake_result("baseline", dtype="int8", text="int8 output"),
    ]
    report = generate_dtype_sweep_report(results)
    assert "fp32" in report
    assert "int8" in report


def test_dtype_sweep_report_notes_divergent_output_without_warning_framing():
    results = [
        _fake_result("baseline", dtype="fp32", text="output A"),
        _fake_result("baseline", dtype="int4", text="output B (garbled)"),
    ]
    report = generate_dtype_sweep_report(results)
    assert "lossy by design" in report
    assert "2 distinct output(s)" in report
    assert "output A" in report and "output B (garbled)" in report


def test_dtype_sweep_report_notes_identical_output():
    results = [
        _fake_result("baseline", dtype="fp32", text="same text"),
        _fake_result("baseline", dtype="fp16", text="same text"),
    ]
    report = generate_dtype_sweep_report(results)
    assert "identical output" in report


def test_dtype_sweep_report_checkpoint_default_label():
    results = [_fake_result("baseline", dtype=None)]
    report = generate_dtype_sweep_report(results)
    assert "checkpoint-default" in report


def test_dtype_sweep_report_with_no_results():
    assert "No results to report" in generate_dtype_sweep_report([])


def test_dtype_sweep_report_all_failed_reports_no_trend():
    results = [_fake_result("baseline", dtype="fp32", success=False)]
    report = generate_dtype_sweep_report(results)
    assert "No successful runs in this sweep" in report


def _fake_matrix_result(dtype, budget_mb, success=True, tok_per_s=10.0, error=None):
    budget_bytes = budget_mb * 1024 * 1024
    return _fake_result(
        "offload",
        policy={"name": "offload", "memory_budget_bytes": budget_bytes},
        dtype=dtype,
        success=success,
        error=error,
        tokens_per_second=tok_per_s,
    )


def test_matrix_report_shows_grid_of_dtype_x_budget():
    results = [
        _fake_matrix_result("fp32", 100, tok_per_s=15.45),
        _fake_matrix_result("int8", 100, tok_per_s=6.03),
        _fake_matrix_result("fp32", 50, tok_per_s=5.0),
        _fake_matrix_result("int8", 50, tok_per_s=2.0),
    ]
    report = generate_matrix_report(results)
    assert "15.4" in report  # 15.45 formatted to 3 sig figs
    assert "6.03" in report
    assert "fp32" in report and "int8" in report
    assert "100MB" in report and "50MB" in report


def test_matrix_report_shows_real_failures_not_hidden():
    results = [
        _fake_matrix_result("int4", 100, success=False, error="int4+offload unsupported"),
    ]
    report = generate_matrix_report(results)
    assert "int4+offload unsupported" in report
    assert "1 failed cell(s)" in report


def test_matrix_report_with_no_results():
    assert "No results to report" in generate_matrix_report([])
