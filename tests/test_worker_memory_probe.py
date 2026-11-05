"""Tests for worker.py's measure_free_memory_gb — the real-memory-aware-
placement counterpart to detect_gpu/bandwidth_probe's pattern: a real
measurement reported every heartbeat, not a hard-coded constant. Mocked at
the subprocess/psutil boundary, same as detect_gpu's own tests would,
rather than requiring a real GPU or needing psutil installed to run.
"""

from __future__ import annotations

import subprocess

import worker


def test_gpu_path_parses_nvidia_smi_free_memory_mib_to_gib(monkeypatch):
    def fake_check_output(cmd, text=True, timeout=5):
        assert any("memory.free" in part for part in cmd)
        return "8192\n"  # MiB

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    gb = worker.measure_free_memory_gb(gpu_available=True)
    assert gb == 8192 / 1024.0


def test_gpu_path_falls_back_to_ram_when_nvidia_smi_fails(monkeypatch):
    def fake_check_output(cmd, text=True, timeout=5):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    class _FakeVirtualMemory:
        available = 4_000_000_000  # bytes

    class _FakePsutil:
        @staticmethod
        def virtual_memory():
            return _FakeVirtualMemory()

    monkeypatch.setitem(__import__("sys").modules, "psutil", _FakePsutil())
    gb = worker.measure_free_memory_gb(gpu_available=True)
    assert gb == 4.0


def test_cpu_only_path_reads_ram_via_psutil(monkeypatch):
    class _FakeVirtualMemory:
        available = 2_000_000_000

    class _FakePsutil:
        @staticmethod
        def virtual_memory():
            return _FakeVirtualMemory()

    monkeypatch.setitem(__import__("sys").modules, "psutil", _FakePsutil())
    gb = worker.measure_free_memory_gb(gpu_available=False)
    assert gb == 2.0


def test_returns_none_when_nothing_is_measurable(monkeypatch):
    def fake_check_output(cmd, text=True, timeout=5):
        raise FileNotFoundError("no nvidia-smi")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    monkeypatch.setitem(__import__("sys").modules, "psutil", None)

    # psutil set to None in sys.modules makes `import psutil` raise
    # ImportError/AttributeError depending on Python version — either way
    # measure_free_memory_gb must degrade to None, not crash the worker's
    # heartbeat loop.
    gb = worker.measure_free_memory_gb(gpu_available=False)
    assert gb is None
