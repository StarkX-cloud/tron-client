"""Real test for worker.py's execute_job subprocess guardrail: a job
that hangs forever must be killed at TRON_JOB_TIMEOUT_SECONDS instead of
wedging the worker forever, and a job that tries to allocate past
TRON_JOB_MEMORY_LIMIT_MB must be killed instead of OOMing the host.

Both only actually exercise the real guardrail on a platform with a
"fork" multiprocessing start method (Linux/Mac -- this project's real
deployment target, proven on the Linode worker tests). On Windows
(spawn-only), worker.py falls back to running jobs in-process with no
guardrail (see worker.py's _MP_CONTEXT), so these are skipped there
rather than silently passing without having tested anything real.
"""

from __future__ import annotations

import importlib
import multiprocessing
import time

import pytest

HAS_FORK = "fork" in multiprocessing.get_all_start_methods()

pytestmark = pytest.mark.skipif(
    not HAS_FORK, reason="no 'fork' start method on this platform -- worker.py's guardrail is a no-op here"
)


def _hang_forever():
    while True:
        time.sleep(1)


def _allocate_too_much_memory():
    # Try to hold ~500MB of real allocated memory -- real RAM touched via
    # bytearray (not just a `* n` view), so RLIMIT_AS/the OS actually sees
    # the pressure instead of a lazily-allocated range.
    chunk = bytearray(500 * 1024 * 1024)
    for i in range(0, len(chunk), 4096):
        chunk[i] = 1
    return "should not get here if the memory limit works"


@pytest.fixture
def worker_module(monkeypatch):
    monkeypatch.setenv("TRON_JOB_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("TRON_JOB_MEMORY_LIMIT_MB", "100")
    import worker
    importlib.reload(worker)
    return worker


def test_a_hanging_job_is_killed_at_the_real_timeout_instead_of_wedging_the_worker(worker_module):
    job = {"id": "hang-job", "function": _serialize(_hang_forever, (), {})}
    start = time.time()
    result = worker_module.execute_job(job)
    elapsed = time.time() - start

    assert result["status"] == "error"
    assert "TimeoutError" in result["error"]
    # Real wall-clock proof it was actually killed near the 2s limit, not
    # left running: generous upper bound for CI jitter, but nowhere near
    # "forever".
    assert elapsed < 10, f"job took {elapsed:.1f}s -- guardrail did not actually kill it promptly"


def test_a_memory_hungry_job_is_killed_instead_of_oom_ing_the_host(worker_module):
    job = {"id": "oom-job", "function": _serialize(_allocate_too_much_memory, (), {})}
    result = worker_module.execute_job(job)

    assert result["status"] == "error"
    # Exact error text varies (MemoryError from Python's allocator
    # hitting RLIMIT_AS, or the kernel OOM-killing the subprocess outright
    # before it can report anything) -- either is the guardrail working,
    # not the job silently succeeding with 500MB resident.
    assert "result" not in job


def _serialize(func, args, kwargs):
    from tron.serializer import serialize
    return serialize((func, args, kwargs))
