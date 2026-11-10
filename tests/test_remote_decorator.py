"""Tests for tron.remote's @remote decorator -- previously untested
entirely, despite being this project's flagship "zero changes, just
decorate" entrypoint. Focused on the real gap closed here: a caller's
memory_gb resource hint must become required_memory_gb in the /submit
payload, so tron/spine/matcher.py's real memory_fits hard filter can
actually enforce it, not just nudge scheduling weight. No real server is
started -- config.ensure_server() and requests.post are mocked so this
exercises exactly the resource-merging/payload logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tron.remote import remote


def _fake_post_response(job_id="fake-job-id"):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"job_id": job_id})
    return response


def test_memory_gb_hint_becomes_required_memory_gb_in_submit_payload():
    @remote(remote_only=True, memory_gb=64)
    def heavy(x):
        return x

    with patch("tron.remote.get_config") as mock_get_config, \
         patch("tron.remote.requests.post") as mock_post:
        mock_get_config.return_value.ensure_server.return_value = "http://fake-master:9000"
        mock_post.return_value = _fake_post_response()

        heavy(1)

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["min_memory_gb"] == 64
        assert payload["required_memory_gb"] == 64  # the real fix: now enforced, not just a weight


def test_no_memory_gb_hint_means_no_required_memory_gb_key():
    # "Zero unless known" -- a caller that never mentions memory must get
    # exactly the pre-existing behavior, not a fabricated requirement.
    @remote(remote_only=True)
    def light(x):
        return x

    with patch("tron.remote.get_config") as mock_get_config, \
         patch("tron.remote.requests.post") as mock_post:
        mock_get_config.return_value.ensure_server.return_value = "http://fake-master:9000"
        mock_post.return_value = _fake_post_response()

        light(1)

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert "required_memory_gb" not in payload
        assert payload["min_memory_gb"] == 1  # unchanged default


def test_call_time_memory_gb_hint_also_becomes_required_memory_gb():
    # memory_gb can be given at call time, not just as a decorator default
    # -- the merge logic (default_resources + execution_kwargs) must carry
    # either source through to required_memory_gb the same way.
    @remote(remote_only=True)
    def variable(x):
        return x

    with patch("tron.remote.get_config") as mock_get_config, \
         patch("tron.remote.requests.post") as mock_post:
        mock_get_config.return_value.ensure_server.return_value = "http://fake-master:9000"
        mock_post.return_value = _fake_post_response()

        variable(1, memory_gb=32)

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["required_memory_gb"] == 32


def test_local_only_never_submits_remotely():
    @remote(local_only=True, memory_gb=64)
    def runs_here(x):
        return x * 2

    with patch("tron.remote.requests.post") as mock_post:
        future = runs_here(5)
        assert mock_post.call_count == 0
        assert future.get() == 10
