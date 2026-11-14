"""Surface existing execution-safety knobs — do not reimplement them.

Recovery for pooled jobs is already owned by ``queue_server._watchdog_loop``
+ ``MAX_JOB_REQUEUE_ATTEMPTS`` + the spine event log. Autopilot only
*declares* those guarantees in the plan and ensures ``required_memory_gb``
reaches ``/submit`` so the matcher's hard memory filter can act.

Checkpoint recovery for arbitrary Python callables is **not** supported
here (cloudpickle jobs are not checkpointed). Training sessions that
already barrier-average via ``tron/training/distributed`` keep their own
recovery story; Autopilot does not invent a second one.
"""

from __future__ import annotations

import os


def safety_summary() -> dict:
    """Report the live safety contract Autopilot relies on."""
    max_requeues = int(os.environ.get("TRON_MAX_JOB_REQUEUE_ATTEMPTS", "3"))
    return {
        "pooled_failure_recovery": "watchdog_requeue",
        "max_job_requeue_attempts": max_requeues,
        "memory_enforcement": "required_memory_gb → matcher.memory_fits",
        "local_hardware_gate": "@tron.remote + detect_hardware (when using @remote)",
        "checkpoint_recovery": "not supported for arbitrary callables",
        "distributed_vram": "not claimed — not demonstrated",
        "notes": [
            "Pooled mid-job worker death is recovered by queue_server's "
            "watchdog via the spine event log (proven in "
            "tests/test_pool_reliability.py).",
            "After max_job_requeue_attempts, the job fails with an explicit "
            "error result instead of requeuing forever.",
            "Autopilot does not provision 'dedicated' infrastructure.",
        ],
    }


def reliability_allows_pool(reliability: str) -> bool:
    """strict workloads must not be sent to the untrusted pool path."""
    return reliability != "strict"
