"""Run one inference workload under a given memory policy and measure it.

Every number here is a real measurement taken during this process's own
execution — no estimates, no values copied from another run. Failures
(missing CUDA, missing dependency, out-of-memory, model not found) are
caught and reported as a failed ``RunOutcome`` rather than crashing the
caller, per the spec's "handle ... errors gracefully" requirement.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from tron_efficiency.workloads.config import WorkloadConfig

try:
    import torch
except ImportError:
    torch = None  # handled per-use below; reported as a warning/failure, not a crash

if TYPE_CHECKING:
    # Only needed for the type hint below; importing it at module load time
    # would create a cycle (memory.policies imports workloads.config, which
    # triggers workloads/__init__.py, which imports this module).
    from tron_efficiency.memory.policies import MemoryPolicy


@dataclass
class RunOutcome:
    success: bool
    policy_name: str
    error: str | None = None
    load_time_seconds: float | None = None
    time_to_first_token_seconds: float | None = None
    end_to_end_latency_seconds: float | None = None
    tokens_generated: int | None = None
    tokens_per_second: float | None = None
    peak_cpu_memory_bytes: int | None = None
    peak_gpu_memory_allocated_bytes: int | None = None
    peak_gpu_memory_reserved_bytes: int | None = None
    device_map: dict = field(default_factory=dict)
    generated_text: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class _RSSSampler:
    """Polls this process's resident set size on a background thread.

    A single reading after the fact only shows RSS at that instant, not the
    peak during loading+generation — sampling periodically on a thread is
    the simplest real way to catch the peak without native OS support
    (Windows has no ``resource.getrusage`` peak-RSS equivalent).
    """

    def __init__(self, interval_seconds: float = 0.05):
        self._interval = interval_seconds
        self._peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = None

    def start(self) -> None:
        try:
            import psutil

            self._process = psutil.Process()
        except ImportError:
            self._process = None
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss
                self._peak = max(self._peak, rss)
            except Exception:
                pass
            self._stop.wait(self._interval)

    def stop(self) -> int | None:
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join(timeout=1.0)
        return self._peak


def run_workload(config: WorkloadConfig, policy: MemoryPolicy) -> RunOutcome:
    sampler = _RSSSampler()
    sampler.start()
    warnings: list[str] = []

    if torch is not None and config.device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        load_start = time.perf_counter()
        loaded = policy.load_model(config)
        load_time = time.perf_counter() - load_start

        tokenizer = loaded.tokenizer
        model = loaded.model

        inputs = tokenizer(
            [config.prompt] * config.batch_size,
            return_tensors="pt",
            padding=True,
        )
        if torch is None:
            raise ImportError("torch")
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.no_grad():
            ttft_start = time.perf_counter()
            model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            time_to_first_token = time.perf_counter() - ttft_start

            e2e_start = time.perf_counter()
            output = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            end_to_end_latency = time.perf_counter() - e2e_start

        input_length = inputs["input_ids"].shape[-1]
        tokens_generated = (output.shape[-1] - input_length) * config.batch_size
        tokens_per_second = (
            tokens_generated / end_to_end_latency if end_to_end_latency > 0 else None
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        peak_gpu_allocated = None
        peak_gpu_reserved = None
        if torch is not None and config.device == "cuda" and torch.cuda.is_available():
            peak_gpu_allocated = torch.cuda.max_memory_allocated()
            peak_gpu_reserved = torch.cuda.max_memory_reserved()

        peak_cpu_memory_bytes = sampler.stop()

        return RunOutcome(
            success=True,
            policy_name=policy.name,
            load_time_seconds=load_time,
            time_to_first_token_seconds=time_to_first_token,
            end_to_end_latency_seconds=end_to_end_latency,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            peak_cpu_memory_bytes=peak_cpu_memory_bytes,
            peak_gpu_memory_allocated_bytes=peak_gpu_allocated,
            peak_gpu_memory_reserved_bytes=peak_gpu_reserved,
            device_map=dict(loaded.device_map) if isinstance(loaded.device_map, dict) else {"device": str(loaded.device_map)},
            generated_text=generated_text,
            warnings=warnings,
        )
    except ImportError as exc:
        sampler.stop()
        return RunOutcome(
            success=False,
            policy_name=policy.name,
            error=f"missing dependency: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - deliberately broad: must never crash the caller
        sampler.stop()
        message = str(exc)
        if "out of memory" in message.lower() or "cuda error" in message.lower():
            error = f"out of memory under this policy's budget: {message}"
        else:
            error = f"{type(exc).__name__}: {message}"
        return RunOutcome(success=False, policy_name=policy.name, error=error)
