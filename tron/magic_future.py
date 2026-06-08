"""
TRON Magic Future - Clean, transparent execution abstraction
Feels like calling normal functions, works with async/await natively.
"""

import asyncio
import threading
import time
import uuid
from typing import Any, Optional


class MagicFuture:
    """
    Unified future that works whether execution is local or remote.
    Supports blocking .get(), async/await, and transparent polling.
    """

    def __init__(
        self,
        job_id: str,
        status_fn,
        is_local: bool = False,
        local_result: Optional[Any] = None,
        local_error: Optional[Exception] = None,
    ):
        self.job_id = job_id
        self._status_fn = status_fn
        self._is_local = is_local
        self._local_result = local_result
        self._local_error = local_error
        self._cache = None
        self._cached_at = 0

    def status(self) -> dict:
        """Get job status with caching."""
        now = time.time()
        if self._cache and now - self._cached_at < 0.5:
            return self._cache

        try:
            status = self._status_fn(self.job_id)
            self._cache = status
            self._cached_at = now
            return status
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def ready(self) -> bool:
        """Is execution complete?"""
        if self._is_local:
            return self._local_result is not None or self._local_error is not None
        return self.status().get("status") in ["completed", "failed"]

    def done(self) -> bool:
        """Alias for ready() - matches concurrent.futures API."""
        return self.ready()

    def result(self, timeout: Optional[float] = None) -> Any:
        """
        Get result, blocking until ready.
        Supports timeout for async-friendly behavior.
        """
        return self.get(timeout=timeout)

    def get(self, timeout: Optional[float] = None, poll: float = 0.1) -> Any:
        """
        Blocking get with optional timeout.
        Perfect for sync code - no manual polling needed.
        """
        if self._is_local:
            if self._local_error:
                raise self._local_error
            return self._local_result

        start = time.time()

        while True:
            status = self.status()
            job_status = status.get("status")

            if job_status == "completed":
                result = status.get("result", {})
                output = result.get("output") or result.get("result")
                return output

            if job_status == "failed":
                error = status.get("error", "Job failed")
                raise RuntimeError(f"[TRON] Job {self.job_id} failed: {error}")

            if timeout:
                elapsed = time.time() - start
                if elapsed > timeout:
                    raise TimeoutError(
                        f"[TRON] Job {self.job_id} timed out after {timeout}s"
                    )

            time.sleep(poll)

    def __await__(self):
        """Support async/await: await future"""
        result = self.get()
        if False:
            yield  # pragma: no cover
        return result

    def __repr__(self) -> str:
        status = self.status().get("status", "unknown")
        return f"<MagicFuture job_id={self.job_id} status={status}>"
