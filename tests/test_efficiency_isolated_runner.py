"""Tests for isolated_runner.py's subprocess-crash-handling logic.

These use real subprocesses (via ``run_target_in_subprocess``, the
primitive ``run_workload_isolated`` is built on) against small, fast,
deliberately-crashing/hanging/succeeding target functions -- NOT a real
model load, which would be slow and whose one real crash turned out not
to reproduce reliably even outside tests (see
writeup/compute-efficiency-matrix-sweep-crash.md). The crash-handling
*logic* itself is what's under test here, proven deterministically rather
than hoping a flaky real-world trigger fires on demand.

Target/worker functions must be module-level (not closures/lambdas) so
the ``spawn`` start method can pickle and re-import them in the child --
this is also why ``run_workload_isolated`` takes an injectable ``worker``
parameter: a monkeypatched local closure would silently fail to pickle,
so tests inject a real module-level fake worker instead.
"""

import os
import time

from tron_efficiency.workloads.config import WorkloadConfig
from tron_efficiency.workloads.isolated_runner import (
    run_target_in_subprocess,
    run_workload_isolated,
)
from tron_efficiency.workloads.runner import RunOutcome


def _succeeding_target(value, result_queue):
    result_queue.put(f"got: {value}")


def _crashing_target(_unused, result_queue):
    # os._exit bypasses Python cleanup and atexit, closer to how a real
    # native-level crash (segfault) terminates a process than
    # sys.exit()/an exception would -- the parent sees a nonzero exit
    # code either way, which is the actual thing under test.
    os._exit(1)


def _hanging_target(_unused, result_queue):
    time.sleep(30)


def test_run_target_in_subprocess_success_returns_the_queued_result():
    status, payload = run_target_in_subprocess(_succeeding_target, ("hello",), timeout_seconds=30)
    assert status == "ok"
    assert payload == "got: hello"


def test_run_target_in_subprocess_detects_a_crash():
    status, payload = run_target_in_subprocess(_crashing_target, (None,), timeout_seconds=30)
    assert status == "crashed"
    assert payload == 1  # the exit code os._exit(1) produced


def test_run_target_in_subprocess_detects_a_timeout():
    status, payload = run_target_in_subprocess(_hanging_target, (None,), timeout_seconds=2)
    assert status == "timeout"
    assert payload is None


class _FakePolicy:
    name = "offload"

    def describe(self):
        return {"name": "offload"}


def _crashing_worker(config, policy, result_queue):
    os._exit(1)


def _hanging_worker(config, policy, result_queue):
    time.sleep(30)


def _succeeding_worker(config, policy, result_queue):
    result_queue.put(RunOutcome(success=True, policy_name="offload", generated_text="real result"))


def test_run_workload_isolated_reports_a_crash_as_a_failed_outcome():
    outcome = run_workload_isolated(
        WorkloadConfig(model_name="fake/model"), _FakePolicy(),
        timeout_seconds=30, worker=_crashing_worker,
    )
    assert isinstance(outcome, RunOutcome)
    assert outcome.success is False
    assert "crashed" in outcome.error
    assert "exit code 1" in outcome.error


def test_run_workload_isolated_reports_a_timeout_as_a_failed_outcome():
    outcome = run_workload_isolated(
        WorkloadConfig(model_name="fake/model"), _FakePolicy(),
        timeout_seconds=2, worker=_hanging_worker,
    )
    assert outcome.success is False
    assert "timed out" in outcome.error


def test_run_workload_isolated_passes_through_a_real_outcome():
    outcome = run_workload_isolated(
        WorkloadConfig(model_name="fake/model"), _FakePolicy(),
        timeout_seconds=30, worker=_succeeding_worker,
    )
    assert outcome.success is True
    assert outcome.generated_text == "real result"
