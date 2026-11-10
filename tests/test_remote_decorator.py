"""Tests for tron.remote's @remote decorator -- previously untested
entirely, despite being this project's flagship "zero changes, just
decorate" entrypoint. Covers two real gaps closed here:

1. A caller's memory_gb resource hint must become required_memory_gb in
   the /submit payload, so tron/spine/matcher.py's real memory_fits hard
   filter can actually enforce it, not just nudge scheduling weight.
2. A declared gpu/memory_gb hint must gate whether this decorator even
   *tries* local execution -- previously it would just run locally
   regardless, silently ignoring the hint unless the function happened
   to crash on mismatched hardware.

No real server is started and no real hardware is probed -- config.
ensure_server(), requests.post, and detect_hardware() are mocked so
these exercise exactly the decision/payload logic, deterministically,
regardless of what machine runs the test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tron.remote import _local_hardware_can_satisfy, remote
from tron_efficiency.hardware.detect import GPUInfo, HardwareInfo


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


def _fake_hardware(gpus=(), cpu_total_memory_bytes=None):
    return HardwareInfo(
        platform="test",
        python_version="3.x",
        cpu_count=4,
        cpu_total_memory_bytes=cpu_total_memory_bytes,
        cuda_available=bool(gpus),
        torch_version=None,
        gpus=list(gpus),
    )


def test_local_hardware_can_satisfy_true_when_nothing_declared():
    assert _local_hardware_can_satisfy(False, None) == (True, None)


def test_local_hardware_cannot_satisfy_gpu_hint_with_no_gpu():
    with patch(
        "tron_efficiency.hardware.detect.detect_hardware",
        return_value=_fake_hardware(gpus=[]),
    ):
        ok, reason = _local_hardware_can_satisfy(True, None)
        assert ok is False
        assert "no detected GPU" in reason


def test_local_hardware_can_satisfy_gpu_hint_with_a_real_gpu():
    gpu = GPUInfo(index=0, name="fake-gpu", total_memory_bytes=8_000_000_000)
    with patch(
        "tron_efficiency.hardware.detect.detect_hardware",
        return_value=_fake_hardware(gpus=[gpu]),
    ):
        ok, reason = _local_hardware_can_satisfy(True, None)
        assert ok is True
        assert reason is None


def test_local_hardware_cannot_satisfy_memory_hint_beyond_cpu_capacity():
    with patch(
        "tron_efficiency.hardware.detect.detect_hardware",
        return_value=_fake_hardware(cpu_total_memory_bytes=4_000_000_000),  # 4GB
    ):
        ok, reason = _local_hardware_can_satisfy(False, 64)  # needs 64GB
        assert ok is False
        assert "64" in reason


def test_local_hardware_can_satisfy_memory_hint_within_cpu_capacity():
    with patch(
        "tron_efficiency.hardware.detect.detect_hardware",
        return_value=_fake_hardware(cpu_total_memory_bytes=64_000_000_000),  # 64GB
    ):
        ok, reason = _local_hardware_can_satisfy(False, 4)  # needs 4GB
        assert ok is True
        assert reason is None


def test_remote_decorator_skips_local_when_gpu_required_but_absent():
    # The actual regression this closes: before this fix, this would have
    # just run `doubles` locally anyway and silently ignored gpu=True.
    @remote(gpu=True)
    def doubles(x):
        return x * 2

    with patch(
        "tron_efficiency.hardware.detect.detect_hardware",
        return_value=_fake_hardware(gpus=[]),
    ), patch("tron.remote.get_config") as mock_get_config, \
         patch("tron.remote.requests.post") as mock_post:
        mock_get_config.return_value.ensure_server.return_value = "http://fake-master:9000"
        mock_post.return_value = _fake_post_response()

        doubles(5)

        assert mock_post.call_count == 1  # fell through to remote, never ran locally


def test_remote_decorator_still_runs_locally_when_hints_are_satisfiable():
    @remote(gpu=False, memory_gb=1)
    def doubles(x):
        return x * 2

    with patch(
        "tron_efficiency.hardware.detect.detect_hardware",
        return_value=_fake_hardware(cpu_total_memory_bytes=64_000_000_000),
    ), patch("tron.remote.requests.post") as mock_post:
        future = doubles(5)

        assert mock_post.call_count == 0
        assert future.get() == 10
