"""The ``trace`` context manager — wraps one LLM call, records one CallRecord.

Deliberately not a proxy server: the caller keeps making their own SDK call
exactly as before and just hands the response back in. Lower integration
cost than re-pointing a base URL, and nothing here ever sees the prompt
content unless the caller's response object happens to echo it back (it
isn't read even if present).

Duck-types both OpenAI- and Anthropic-shaped response objects rather than
importing either SDK, so this package has zero hard dependency on either
provider's client library.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

from tron_costtrace import pricing
from tron_costtrace.schema import CallRecord
from tron_costtrace.store import JsonlStore

DEFAULT_LOG_PATH_ENV = "TRON_COSTTRACE_PATH"
DEFAULT_LOG_PATH = "tron_costtrace_log.jsonl"


def default_store() -> JsonlStore:
    return JsonlStore(os.environ.get(DEFAULT_LOG_PATH_ENV, DEFAULT_LOG_PATH))


class _TraceHandle:
    def __init__(self, task: str):
        self.task = task
        self.model: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.retry_count = 0
        self._success: bool | None = None

    def record(self, response) -> None:
        """Pull model + token usage out of an OpenAI- or Anthropic-shaped
        response. Call once per attempt; the last call before the context
        exits wins (matches "did the final attempt succeed")."""
        model = getattr(response, "model", None)
        if model is not None:
            self.model = model

        usage = getattr(response, "usage", None)
        if usage is None:
            return

        input_tokens = getattr(usage, "prompt_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "output_tokens", None)

        if input_tokens is not None:
            self.input_tokens = input_tokens
        if output_tokens is not None:
            self.output_tokens = output_tokens

    def retry(self) -> None:
        self.retry_count += 1

    def mark_success(self, success: bool) -> None:
        self._success = success


@contextmanager
def trace(task: str, store: JsonlStore | None = None):
    """Wrap one LLM call (or one logical attempt-with-retries) and record
    cost/latency/outcome for it.

    >>> with trace("support_ticket_classify") as t:
    ...     resp = client.chat.completions.create(...)
    ...     t.record(resp)
    ...     t.mark_success(True)
    """
    store = store if store is not None else default_store()
    handle = _TraceHandle(task)
    start = time.monotonic()
    error: str | None = None
    try:
        yield handle
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.monotonic() - start
        success = handle._success if handle._success is not None else (error is None)
        model = handle.model or "unknown"
        cost = pricing.cost_usd(model, handle.input_tokens, handle.output_tokens)
        store.append(
            CallRecord(
                task=task,
                model=model,
                input_tokens=handle.input_tokens,
                output_tokens=handle.output_tokens,
                cost_usd=cost,
                latency_seconds=elapsed,
                retry_count=handle.retry_count,
                success=success,
                error=error,
                timestamp=time.time(),
            )
        )
