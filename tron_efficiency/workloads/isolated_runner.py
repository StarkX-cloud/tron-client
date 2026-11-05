"""Run a workload in a subprocess so a native-level crash can't take
everything down with it.

Real, reproduced finding (writeup/compute-efficiency-matrix-sweep-crash.md):
``OffloadPolicy`` on this CPU-only dev machine can segfault for real --
confirmed reproducible with ``EleutherAI/pythia-70m``, ``dtype=None``,
``device="cpu"``, a 130MB memory budget. Not a Python exception: the
process exits with code 139 (SIGSEGV), and ``run_workload``'s own
``except Exception`` (workloads/runner.py) categorically cannot catch
this -- a segfault happens below the Python interpreter, in native code
(almost certainly inside accelerate's disk-offload / safetensors mmap
path or a C extension it calls into). No amount of try/except in
``run_workload`` fixes this; the only real mitigation is running the
risky work somewhere a crash can be detected and contained instead of
taking the whole process (and every other result in a sweep) with it.
"""

from __future__ import annotations

import multiprocessing as mp

from tron_efficiency.workloads.config import WorkloadConfig
from tron_efficiency.workloads.runner import RunOutcome

DEFAULT_ISOLATION_TIMEOUT_SECONDS = 600


def run_target_in_subprocess(target, args: tuple, timeout_seconds: float):
    """Generic primitive: run ``target(*args, result_queue)`` in a fresh
    ``spawn``-context subprocess and report what happened, without ever
    raising in the caller. Returns ``(status, payload)`` where ``status``
    is one of ``"ok"`` (``payload`` is whatever the target put on the
    queue), ``"crashed"`` (``payload`` is the exit code), ``"timeout"``
    (``payload`` is ``None``), or ``"no_result"`` (the process exited 0
    but never queued anything -- ``payload`` is ``None``).

    Separated from ``run_workload_isolated`` specifically so the
    subprocess-crash-handling *logic* can be unit tested directly against
    a small, fast, deliberately-crashing/hanging target function, instead
    of only being provable by loading a real model (slow, and the one
    real crash this project found turned out to not even reproduce every
    time under isolation -- see ``run_workload_isolated``'s docstring).

    Uses the ``spawn`` start method explicitly (Windows' only real option
    anyway, and the safer choice generally) so the child gets a genuinely
    fresh process, not a fork sharing potentially-corrupted memory state
    with a parent that's about to run more work.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=target, args=(*args, result_queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return "timeout", None

    if process.exitcode != 0:
        return "crashed", process.exitcode

    try:
        return "ok", result_queue.get_nowait()
    except Exception:
        return "no_result", None


def _worker(config: WorkloadConfig, policy, result_queue) -> None:
    """Runs in the child process. Imported lazily so the (slow) torch/
    transformers import only happens in the child, not also paid for by
    the parent process that's just orchestrating."""
    from tron_efficiency.workloads.runner import run_workload

    outcome = run_workload(config, policy)
    result_queue.put(outcome)


def run_workload_isolated(
    config: WorkloadConfig,
    policy,
    timeout_seconds: float = DEFAULT_ISOLATION_TIMEOUT_SECONDS,
    worker=_worker,
) -> RunOutcome:
    """Same contract as ``run_workload`` (always returns a ``RunOutcome``,
    never raises) but runs the actual work in a subprocess, so:

    - A real crash (segfault, native abort) in the child shows up as a
      nonzero/negative exit code the parent can detect and report as a
      failed ``RunOutcome``, instead of killing the caller too.
    - A hang (the child never finishes) is bounded by ``timeout_seconds``
      instead of blocking a sweep forever.

    Honest note on the real crash this was built for (writeup/compute-
    efficiency-matrix-sweep-crash.md): re-running that exact scenario
    (``OffloadPolicy``, ``dtype=None``, 130MB, pythia-70m, CPU) *through*
    this isolation path did not crash -- it succeeded. This does not mean
    the underlying crash is fixed or explained; the crash was
    independently reproduced twice outside isolation (same config, same
    exit code 139) and might simply be non-deterministic, or isolation's
    cleaner per-process import state might incidentally avoid whatever
    triggers it. Either way, the property this module actually
    guarantees -- a crash in one subprocess cannot propagate to the
    caller or kill a sweep in progress -- is verified separately and
    deterministically in tests/test_efficiency_isolated_runner.py against
    a target that crashes on purpose, not by relying on the flaky real
    trigger reproducing on demand.

    ``worker`` defaults to the real ``_worker`` (which loads a real model
    via ``run_workload``); tests substitute a small, fast, module-level
    function instead (must be module-level, not a closure/lambda, so
    ``spawn`` can pickle and re-import it in the child).
    """
    status, payload = run_target_in_subprocess(worker, (config, policy), timeout_seconds)

    if status == "timeout":
        return RunOutcome(
            success=False,
            policy_name=getattr(policy, "name", "unknown"),
            error=(
                f"worker process timed out after {timeout_seconds}s and was "
                "terminated -- the workload may be hung, not just slow"
            ),
        )
    if status == "crashed":
        return RunOutcome(
            success=False,
            policy_name=getattr(policy, "name", "unknown"),
            error=(
                f"worker process crashed (exit code {payload}) -- this is a "
                "native-level crash (e.g. a segfault), not a Python exception, "
                "so it cannot be caught with try/except inside run_workload. "
                f"Config: model={config.model_name!r}, dtype={config.dtype!r}, "
                f"device={config.device!r}, policy={getattr(policy, 'describe', lambda: {})()}"
            ),
        )
    if status == "no_result":
        return RunOutcome(
            success=False,
            policy_name=getattr(policy, "name", "unknown"),
            error="worker process exited cleanly (code 0) but produced no result -- unexpected",
        )
    return payload
