"""One shard's side of an over-the-wire distributed training run.

`run_shard` is the whole loop: fetch this shard's data, then for each
round — pull the round's starting vector from the master, train
`local_steps` steps on it locally with zero communication, post the
trained vector back, wait for the barrier to clear. It reuses
`local_sgd._sample_batch` and `TinyMLP.train_step` unchanged, so a shard
process runs the identical computation a shard object runs in
`local_sgd.train_local_sgd` — the only difference is that the vectors
cross a socket.

Transport is injected (`Transport` protocol below) so the same function
runs against a real `requests` session pointed at a remote master, or
against a `fastapi.testclient.TestClient` in-process for the parity test.
"""
from __future__ import annotations

import random
import time
from typing import Optional, Protocol

from ..local_sgd import _sample_batch
from ..model import TinyMLP
from .protocol import decode_dataset, decode_vector, encode_vector


class Transport(Protocol):
    def get_bytes(self, path: str) -> Optional[bytes]:
        """GET `path`; return the body bytes, or None on HTTP 202 (barrier
        not ready yet)."""

    def get_json(self, path: str) -> dict: ...

    def post_bytes(self, path: str, blob: bytes) -> dict: ...


class RequestsTransport:
    """Default transport: a real HTTP client for pointing a shard process
    at a master on another host.

    Retries transient network failures with exponential backoff + jitter.
    This is not defensive boilerplate — a real WAN test against a live
    remote master (see tests/test_wan_transport.py) hit exactly this:
    a shard's very first request came back as a mid-handshake connection
    reset (no fault of the master; the shape any flaky/congested real
    link produces), and with zero retry logic that permanently killed the
    shard — a training run relying on one lucky handshake per shard is
    not something you can hand a telecom. Only failures that are plausibly
    transient are retried (connection errors, timeouts, and 5xx); a 4xx
    is a real application error (bad session id, malformed request) and
    is raised immediately — retrying it would just repeat the same
    mistake `max_retries` times.
    """

    def __init__(
        self,
        base_url: str,
        auth_token: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 5,
        backoff_base: float = 0.5,
        backoff_max: float = 20.0,
    ):
        import requests

        self._requests = requests
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._headers = {"X-TRON-AUTH": auth_token} if auth_token else {}
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max

    def _sleep_for_retry(self, attempt: int) -> None:
        # Exponential backoff (attempt 0 -> backoff_base, 1 -> 2x, ...)
        # with full jitter, capped at backoff_max, so N shards that all
        # just failed at once don't all retry in lockstep and hammer the
        # master with a synchronized second wave.
        delay = min(self._backoff_max, self._backoff_base * (2 ** attempt))
        time.sleep(random.uniform(0, delay))

    def _with_retries(self, description: str, call):
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = call()
            except self._requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                self._sleep_for_retry(attempt)
                continue

            if resp.status_code == 202:
                return resp
            if 500 <= resp.status_code < 600 and attempt < self._max_retries:
                last_error = self._requests.exceptions.HTTPError(
                    f"{resp.status_code} from {description}", response=resp
                )
                self._sleep_for_retry(attempt)
                continue

            resp.raise_for_status()  # a 4xx (or an un-retried 5xx) raises here, immediately
            return resp

        raise last_error

    def get_bytes(self, path: str) -> Optional[bytes]:
        resp = self._with_retries(
            path, lambda: self._requests.get(self._base + path, headers=self._headers, timeout=self._timeout)
        )
        if resp.status_code == 202:
            return None
        return resp.content

    def get_json(self, path: str) -> dict:
        resp = self._with_retries(
            path, lambda: self._requests.get(self._base + path, headers=self._headers, timeout=self._timeout)
        )
        return resp.json()

    def post_bytes(self, path: str, blob: bytes) -> dict:
        resp = self._with_retries(
            path,
            lambda: self._requests.post(
                self._base + path, data=blob,
                headers={**self._headers, "Content-Type": "application/octet-stream"},
                timeout=self._timeout,
            ),
        )
        return resp.json()


def run_shard(
    transport: Transport,
    session_id: str,
    shard_idx: int,
    *,
    poll_interval: float = 0.05,
    max_wait_seconds: float = 120.0,
    node_id: Optional[str] = None,
) -> dict:
    """Drive shard `shard_idx` through the whole run. Returns the master's
    final status for the session (as seen from this shard)."""
    meta = transport.get_json(f"/training/session/{session_id}")
    num_rounds = int(meta["num_rounds"])
    local_steps = int(meta["local_steps"])
    lr = float(meta["lr"])
    model_config = meta["model_config"]

    x, y = decode_dataset(transport.get_bytes(f"/training/session/{session_id}/shard/{shard_idx}/data"))

    model = TinyMLP(**model_config)

    for round_idx in range(num_rounds):
        init_blob = _await_bytes(
            transport, f"/training/session/{session_id}/round/{round_idx}/init?shard={shard_idx}",
            poll_interval, max_wait_seconds,
        )
        model.set_flat_params(decode_vector(init_blob))

        for local_step in range(local_steps):
            global_step = round_idx * local_steps + local_step
            batch_x, batch_y = _sample_batch(x, y, global_step)
            model.train_step(batch_x, batch_y, lr)

        path = f"/training/session/{session_id}/round/{round_idx}/shard/{shard_idx}"
        if node_id:
            path += f"?node_id={node_id}"
        transport.post_bytes(path, encode_vector(model.get_flat_params()))

    return transport.get_json(f"/training/session/{session_id}")


def _await_bytes(transport: Transport, path: str, poll_interval: float, max_wait: float) -> bytes:
    waited = 0.0
    while True:
        blob = transport.get_bytes(path)
        if blob is not None:
            return blob
        if waited >= max_wait:
            raise TimeoutError(f"barrier at {path} did not clear within {max_wait}s")
        time.sleep(poll_interval)
        waited += poll_interval
