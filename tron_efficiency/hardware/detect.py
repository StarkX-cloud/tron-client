"""Real hardware discovery — no simulation, no reported-not-measured specs.

Unlike ``tron/gpu/*`` (which aggregates *reported* worker specs for the job
scheduler), this module calls ``torch.cuda`` and ``psutil`` directly and
records exactly what it measured. If a value can't be measured, the field is
``None`` rather than a guess.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, field


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_bytes: int


@dataclass
class HardwareInfo:
    platform: str
    python_version: str
    cpu_count: int | None
    cpu_total_memory_bytes: int | None
    cuda_available: bool
    torch_version: str | None
    gpus: list[GPUInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def detect_hardware() -> HardwareInfo:
    """Probe the machine this process is actually running on.

    Never raises on a missing optional dependency — records a warning
    instead, so callers (CLI/API/dashboard) always get a usable result.
    """
    warnings: list[str] = []

    cpu_count: int | None = None
    cpu_total_memory_bytes: int | None = None
    try:
        import psutil

        cpu_count = psutil.cpu_count(logical=True)
        cpu_total_memory_bytes = psutil.virtual_memory().total
    except ImportError:
        warnings.append(
            "psutil not installed — CPU core count and RAM size could not be "
            "measured. Install requirements-training.txt."
        )

    cuda_available = False
    torch_version: str | None = None
    gpus: list[GPUInfo] = []
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    GPUInfo(
                        index=i,
                        name=props.name,
                        total_memory_bytes=props.total_memory,
                    )
                )
        else:
            warnings.append(
                "torch.cuda.is_available() is False — no CUDA GPU detected on "
                "this machine. GPU-memory-budget experiments cannot be run "
                "here; only the CPU execution path is available."
            )
    except ImportError:
        warnings.append(
            "torch not installed — no inference workload can run. Install "
            "requirements-training.txt."
        )

    return HardwareInfo(
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu_count=cpu_count,
        cpu_total_memory_bytes=cpu_total_memory_bytes,
        cuda_available=cuda_available,
        torch_version=torch_version,
        gpus=gpus,
        warnings=warnings,
    )
