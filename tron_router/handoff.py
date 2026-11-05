"""Execute a 'pooled' PlacementDecision for real: submit a job to the live
queue_server.py instance the decision was checked against.

This is the piece that was missing before this module existed:
tron_router.decision.decide_placement only ever *decided* that a job
should go to the pool -- nothing in this repo actually handed it off.
This module does that hand-off using the exact same mechanism
tron.remote already uses for its own remote path (tron/serializer.py's
cloudpickle+base64 encoding, POSTed to /submit) -- it does not invent a
second submission protocol.
"""

from __future__ import annotations

import requests
from requests.exceptions import RequestException

from tron.magic_future import MagicFuture
from tron.serializer import serialize

from .decision import DEFAULT_MASTER_URL, JobRequest, PlacementDecision


class HandoffError(RuntimeError):
    """Raised when a pooled decision can't actually be submitted."""


def submit_to_pool(
    job: JobRequest,
    decision: PlacementDecision,
    func,
    func_args: tuple = (),
    func_kwargs: dict | None = None,
    master_url: str | None = None,
    gpu_required: bool = False,
    min_memory_gb: float = 1.0,
    priority: int = 1,
    timeout_seconds: float = 10.0,
) -> MagicFuture:
    """Hand a 'pooled' PlacementDecision off to the real job queue.

    ``func`` is the callable that should actually run on a worker (for
    example a closure around ``tron_efficiency.workloads.runner.run_workload``
    for ``job.model_name``) -- this module does not decide what runs, it
    only submits what the caller gives it, honestly, through the real
    /submit path tron/remote.py already exercises. Resource hints
    (``gpu_required``/``min_memory_gb``/``priority``) are passed straight
    through to queue_server.py's own fields; this module adds no second
    resource model on top of the existing one.

    Raises ``HandoffError`` if ``decision.target`` isn't ``"pooled"`` --
    this function executes a pooled decision, it does not re-decide
    placement or silently fall back to local/dedicated.
    """
    if decision.target != "pooled":
        raise HandoffError(
            f"decision.target is {decision.target!r}, not 'pooled' -- "
            "submit_to_pool only executes a pooled decision. Act on "
            "'local' by running the job here, or on 'dedicated' through "
            "real dedicated infrastructure, instead of calling this."
        )

    url = (master_url or DEFAULT_MASTER_URL).rstrip("/")
    payload = serialize((func, func_args, func_kwargs or {}))
    submit_payload = {
        "function": payload,
        "gpu_required": gpu_required,
        "min_memory_gb": min_memory_gb,
        "priority": priority,
    }

    try:
        response = requests.post(f"{url}/submit", json=submit_payload, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
    except RequestException as exc:
        raise HandoffError(
            f"failed to submit job for {job.model_name!r} to {url}: {exc}"
        ) from exc

    if not isinstance(data, dict) or "job_id" not in data:
        raise HandoffError(f"bad response from {url}/submit: {data!r}")

    job_id = data["job_id"]

    def status_fn(jid: str) -> dict:
        try:
            resp = requests.get(f"{url}/status/{jid}", timeout=5)
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - surfaced through MagicFuture.status()
            return {"status": "error", "error": str(exc)}

    return MagicFuture(job_id, status_fn, is_local=False)
