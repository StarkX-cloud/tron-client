from tron_costtrace.report import generate_task_report, summarize_by_task
from tron_costtrace.schema import CallRecord


def _record(task, model="gpt-4o-mini", cost=0.01, success=True, retry_count=0, error=None):
    return CallRecord(
        task=task,
        model=model,
        input_tokens=100,
        output_tokens=50,
        cost_usd=cost,
        latency_seconds=0.2,
        retry_count=retry_count,
        success=success,
        error=error,
        timestamp=0.0,
    )


def test_empty_records_reports_no_data():
    assert "No recorded calls" in generate_task_report([])


def test_cost_per_successful_task_excludes_failed_from_denominator():
    records = [
        _record("classify", cost=0.01, success=True),
        _record("classify", cost=0.01, success=True),
        _record("classify", cost=0.02, success=False),
    ]
    summaries = summarize_by_task(records)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.n_success == 2
    assert s.n_failed == 1
    assert s.total_known_cost_usd == 0.04
    assert s.cost_per_successful_task_usd == 0.02
    assert s.failed_cost_usd == 0.02


def test_all_failures_reports_undefined_per_task_cost():
    records = [_record("classify", success=False)]
    summaries = summarize_by_task(records)
    assert summaries[0].cost_per_successful_task_usd is None
    report = generate_task_report(records)
    assert "undefined (no successes)" in report


def test_unknown_cost_tracked_separately():
    records = [_record("classify", cost=None, success=True)]
    summaries = summarize_by_task(records)
    assert summaries[0].n_unknown_cost == 1
    report = generate_task_report(records)
    assert "unknown cost" in report


def test_retry_overhead_warning_present():
    records = [_record("classify", retry_count=3, success=True)]
    report = generate_task_report(records)
    assert "3 retry attempt" in report


def test_multiple_tasks_sorted_and_both_present():
    records = [_record("b_task"), _record("a_task")]
    report = generate_task_report(records)
    assert "`a_task`" in report and "`b_task`" in report
    assert report.index("`a_task`") < report.index("`b_task`")


def test_failed_call_cost_surfaced_as_overhead_warning():
    records = [_record("classify", cost=0.05, success=False)]
    report = generate_task_report(records)
    assert "hidden overhead" in report
