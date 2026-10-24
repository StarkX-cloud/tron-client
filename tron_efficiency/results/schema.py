"""The structured result produced by every single run.

One ``BenchmarkResult`` = one hardware snapshot + one workload config + one
memory policy + one ``RunOutcome``. Nothing here is an aggregate or an
estimate — aggregation and comparison happen in ``report.py``, over a list
of these.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from tron_efficiency.hardware.detect import HardwareInfo
from tron_efficiency.workloads.config import WorkloadConfig
from tron_efficiency.workloads.runner import RunOutcome


@dataclass
class BenchmarkResult:
    run_id: str
    timestamp: float
    hardware: HardwareInfo
    config: WorkloadConfig
    policy: dict
    outcome: RunOutcome
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        hardware: HardwareInfo,
        config: WorkloadConfig,
        policy_description: dict,
        outcome: RunOutcome,
    ) -> "BenchmarkResult":
        warnings = list(hardware.warnings) + list(outcome.warnings)
        if config.device == "cuda" and not hardware.cuda_available:
            warnings.append(
                "config requested device=cuda but no CUDA GPU was detected; "
                "this result cannot be a real GPU measurement."
            )
        return cls(
            run_id=str(uuid.uuid4()),
            timestamp=time.time(),
            hardware=hardware,
            config=config,
            policy=policy_description,
            outcome=outcome,
            warnings=warnings,
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "hardware": self.hardware.to_dict(),
            "config": self.config.to_dict(),
            "policy": self.policy,
            "outcome": self.outcome.to_dict(),
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkResult":
        from tron_efficiency.hardware.detect import GPUInfo

        hw = data["hardware"]
        hardware = HardwareInfo(
            platform=hw["platform"],
            python_version=hw["python_version"],
            cpu_count=hw["cpu_count"],
            cpu_total_memory_bytes=hw["cpu_total_memory_bytes"],
            cuda_available=hw["cuda_available"],
            torch_version=hw["torch_version"],
            gpus=[GPUInfo(**g) for g in hw.get("gpus", [])],
            warnings=hw.get("warnings", []),
        )
        config = WorkloadConfig(**data["config"])
        outcome = RunOutcome(**data["outcome"])
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            hardware=hardware,
            config=config,
            policy=data["policy"],
            outcome=outcome,
            warnings=data.get("warnings", []),
        )
