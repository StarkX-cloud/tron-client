import functools
import requests
import threading
import time
import uuid
from requests.exceptions import RequestException
from .serializer import serialize
from .config import get_config
from .magic_future import MagicFuture


def _local_hardware_can_satisfy(requires_gpu: bool, required_memory_gb) -> tuple[bool, str | None]:
    """Can *this* literal process's real detected hardware actually meet
    a caller's declared gpu/memory_gb hint? Returns (True, None) when
    satisfiable or nothing was declared to check -- "zero unless known,"
    same contract as the required_memory_gb forwarding below.

    Without this, `@remote(gpu=True)` on a machine with no GPU would
    still just run the function locally if it didn't happen to crash --
    silently violating the one thing the caller actually asked for,
    the same class of gap `required_memory_gb` forwarding closed for the
    remote path. This is deliberately the only consumer of
    `tron_efficiency.hardware.detect` from this file -- reusing the real
    probe already built for the advisor, not inventing a second one.
    """
    if not requires_gpu and required_memory_gb is None:
        return True, None

    from tron_efficiency.hardware.detect import detect_hardware

    hardware = detect_hardware()

    if requires_gpu and not hardware.gpus:
        return False, "gpu=True requested but this machine has no detected GPU"

    if required_memory_gb is not None:
        if requires_gpu and hardware.gpus:
            available_bytes = max(gpu.total_memory_bytes for gpu in hardware.gpus)
        else:
            available_bytes = hardware.cpu_total_memory_bytes
        if available_bytes is not None:
            available_gb = available_bytes / 1_000_000_000.0
            if required_memory_gb > available_gb:
                return False, (
                    f"memory_gb={required_memory_gb} requested but this machine reports only "
                    f"{available_gb:.1f}GB available"
                )

    return True, None


def remote(fn=None, local_first: bool = True, **default_resources):
    """
    @remote decorator - Make any function distributed with zero changes.

    Usage:
        @remote
        def my_func(x):
            return x * 2

        result = my_func(5).get()  # Transparent execution
        # or
        result = await my_func(5)  # Async support

    Args:
        local_first: Try local execution first, fallback to remote
        **default_resources: GPU, memory hints (gpu=True, memory_gb=8)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract call-time execution hints
            execution_kwargs = {}
            for key in list(kwargs.keys()):
                if key in ["local_only", "remote_only", "gpu", "memory_gb", "priority"]:
                    execution_kwargs[key] = kwargs.pop(key)

            # Merge decorator defaults with call-time execution hints
            resources = {**default_resources, **execution_kwargs}

            local_only = resources.get("local_only", False)
            remote_only = resources.get("remote_only", False)

            # Determine execution strategy
            should_try_local = local_only or (local_first and not remote_only)

            # A declared gpu/memory_gb hint gates the *default* local
            # attempt against this machine's real hardware -- an explicit
            # local_only still always tries (and still raises clearly
            # below if it genuinely fails), since that's the caller
            # overriding this decision on purpose, not asking for it.
            if should_try_local and not local_only:
                requires_gpu = bool(resources.get("gpu", False))
                required_memory_gb = resources.get("memory_gb") if "memory_gb" in resources else None
                can_run_locally, infeasible_reason = _local_hardware_can_satisfy(
                    requires_gpu, required_memory_gb
                )
                if not can_run_locally:
                    should_try_local = False
                    print(f"[TRON] Skipping local execution for {func.__name__}: {infeasible_reason}")

            # Try local execution if enabled
            if should_try_local:
                try:
                    result = func(*args, **kwargs)
                    job_id = f"local-{uuid.uuid4()}"
                    return MagicFuture(
                        job_id, 
                        lambda _: {"status": "completed"},
                        is_local=True,
                        local_result=result
                    )
                except Exception as e:
                    if local_only:
                        raise RuntimeError(
                            f"[TRON] local_only=True but local execution failed for {func.__name__}: {e}"
                        ) from e
                    # Fallthrough to remote
                    pass

            # Ensure a TRON server is available before submitting
            config = get_config()
            try:
                queue_url = config.ensure_server()
            except Exception as exc:
                raise RuntimeError(
                    f"[TRON] Failed to ensure a local TRON server for {func.__name__}: {exc}"
                ) from exc

            payload = serialize((func, args, kwargs))

            # Merge default resources with call-time hints
            resources = {**default_resources, **execution_kwargs}
            submit_payload = {
                "function": payload,
                "gpu_required": resources.get("gpu", False),
                "min_memory_gb": resources.get("memory_gb", 1),
                "priority": resources.get("priority", 1),
            }
            # A caller writing `memory_gb=64` reads, by every reasonable
            # expectation, as "this job needs 64GB" -- but queue_server.py's
            # `min_memory_gb` (what the line above sets) is documented as a
            # soft scheduling weight only, never checked against a worker's
            # real capacity (see queue_server.py's /submit, "memory_gb ...
            # is a soft scheduling weight, not a correctness constraint").
            # The real hard filter -- tron/spine/matcher.py's memory_fits,
            # the one this project built and benchmarked to actually prevent
            # OOM placements -- only ever gets reached through tron_router's
            # model-name-based estimate (decide_placement). @remote wraps
            # arbitrary functions with no model name, so that path can never
            # apply here -- but a caller-supplied number is real information
            # this decorator was silently discarding. Forward it as
            # required_memory_gb too, so the one resource hint a developer
            # actually gives this decorator is an enforced guarantee, not
            # just a nudge.
            if "memory_gb" in resources:
                submit_payload["required_memory_gb"] = resources["memory_gb"]

            try:
                r = requests.post(
                    f"{queue_url}/submit",
                    json=submit_payload,
                    timeout=10
                )
                r.raise_for_status()
                data = r.json()
            except RequestException as exc:
                raise RuntimeError(
                    f"[TRON] Failed to submit {func.__name__} to {queue_url}: {exc}"
                ) from exc

            if not isinstance(data, dict) or "job_id" not in data:
                raise RuntimeError(f"[TRON] Bad server response: {data}")

            job_id = data["job_id"]

            # Create status function for MagicFuture
            def status_fn(jid):
                try:
                    resp = requests.get(f"{queue_url}/status/{jid}", timeout=5)
                    return resp.json()
                except Exception as e:
                    return {"status": "error", "error": str(e)}

            return MagicFuture(job_id, status_fn, is_local=False)

        return wrapper

    # Handle both @remote and @remote(...) syntaxes
    if fn is not None:
        return decorator(fn)
    return decorator
