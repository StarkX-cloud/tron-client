"""Choose optimization levers from measured outcomes + advisor signals.

Levers (only when technically justified by hardware/history):
  - memory policy (baseline vs offload)
  - dtype / quantization when supported
  - CPU/GPU device when CUDA is real
  - batch size from memory headroom / history
  - distributed partitioning when cost model says so (via Autopilot)

Never claims independent GPUs form one VRAM pool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from tron_efficiency.advisor.recommend import Recommendation
from tron_efficiency.hardware.detect import HardwareInfo

from tron_adaptive.batching import choose_batch_size
from tron_adaptive.performance_store import PerformanceStore
from tron_autopilot.types import WorkloadSpec


@dataclass
class OptimizationProfile:
    """Concrete levers applied before Autopilot plan/execute."""

    mode: str  # "unoptimized" | "optimized"
    policy_override: str | None = None  # "baseline" | "offload" | None=advisor
    dtype_override: str | None = None
    device_override: str | None = None
    batch_size: int = 1
    enable_distributed: bool = False
    max_parallel_workers: int | None = None
    memory_budget_bytes: int | None = None
    rationale: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class AdaptiveOptimizer:
    def __init__(self, store: PerformanceStore | None = None):
        self.store = store or PerformanceStore()

    def unoptimized_profile(self, spec: WorkloadSpec) -> OptimizationProfile:
        """Fixed naive path: baseline, fp32, batch=1, no distributed."""
        return OptimizationProfile(
            mode="unoptimized",
            policy_override="baseline",
            dtype_override="fp32",
            device_override="cpu",  # may be upgraded only if CUDA proven below
            batch_size=1,
            enable_distributed=False,
            rationale=[
                "unoptimized baseline: baseline policy, fp32, batch_size=1, "
                "no distributed partitioning"
            ],
            tradeoffs=[
                "maximum simplicity; may OOM on tight memory or under-use GPU",
                "no quantization — higher memory, often higher CPU quality",
            ],
        )

    def optimized_profile(
        self,
        spec: WorkloadSpec,
        *,
        hardware: HardwareInfo,
        recommendation: Recommendation | None = None,
    ) -> OptimizationProfile:
        """Build levers from advisor + measured history + hardware reality."""
        rationale: list[str] = []
        tradeoffs: list[str] = []
        rec = recommendation

        # Device: only claim CUDA when detect_hardware says so.
        device = "cpu"
        if hardware.cuda_available and hardware.gpus:
            device = "cuda"
            rationale.append(
                f"CUDA available ({hardware.gpus[0].name}) — prefer GPU execution"
            )
            tradeoffs.append(
                "GPU path uses discrete VRAM on this device only — "
                "not aggregated across machines"
            )
        else:
            rationale.append("no CUDA — CPU execution (honest hetero fallback)")

        policy = "baseline"
        dtype = "fp32"
        budget = None
        if rec is not None:
            policy = rec.policy or "baseline"
            dtype = rec.dtype or "fp32"
            budget = rec.memory_budget_bytes
            device = rec.device or device
            rationale.append(
                f"advisor recommends policy={policy} dtype={dtype} device={device}"
            )
            if policy == "offload":
                tradeoffs.append(
                    "offload reduces peak device memory via CPU/disk transfer; "
                    "usually slower than baseline when memory already fits"
                )
            if dtype in ("int8", "int4"):
                tradeoffs.append(
                    f"{dtype} quantization saves memory; on CPU often slower "
                    "(measured in dtype-sweep writeup) — used only when needed"
                )

        # History override: if a config succeeded faster/higher tok/s, prefer it.
        best = self.store.best_successful_key(spec.model_name)
        hist_batch = None
        if best:
            rationale.append(f"performance history best key: {best}")
            parts = best.split("|")
            # model|target|policy|device|dtype|bN
            if len(parts) >= 6:
                if parts[2] != "-":
                    policy = parts[2]
                if parts[3] != "-":
                    device = parts[3]
                if parts[4] != "-":
                    dtype = parts[4]
                try:
                    hist_batch = int(parts[5].lstrip("b"))
                except ValueError:
                    hist_batch = None
                rationale.append(
                    "adopting measured best policy/device/dtype/batch from history"
                )

        # Batch size from memory headroom / history.
        n_items = len(spec.partition_items) if spec.partition_items else 1
        # For efficiency workloads without partition_items, treat as 1 "prompt".
        mem_budget = budget
        if mem_budget is None and hardware.cuda_available and hardware.gpus:
            mem_budget = hardware.gpus[0].total_memory_bytes
        elif mem_budget is None:
            mem_budget = hardware.cpu_total_memory_bytes

        # Look up last measured peak for this model under current policy.
        measured_peak = None
        for s in reversed(self.store.samples_for(spec.model_name)):
            if s.success and s.peak_memory_bytes:
                measured_peak = s.peak_memory_bytes
                break

        # For efficiency (no partition list), treat as a single logical workload
        # whose generate batch_size can still grow from measured history.
        item_count = (
            len(spec.partition_items)
            if spec.partition_items
            else max(hist_batch or 1, int(getattr(spec, "batch_size", 1) or 1))
        )
        batch_plan = choose_batch_size(
            max(1, item_count),
            measured_peak_memory_bytes=measured_peak,
            memory_budget_bytes=int(mem_budget) if mem_budget else None,
            max_batch_size=int(getattr(spec, "max_batch_size", 8) or 8),
            history_best_batch=hist_batch,
        )
        if hist_batch:
            bs = batch_plan.batch_size
        elif not spec.partition_items:
            bs = 1
            rationale.append(
                "efficiency workload without history — keep batch_size=1 "
                "until measured"
            )
        else:
            bs = batch_plan.batch_size
        rationale.extend(batch_plan.rationale)

        # Distributed: only enable when caller marked parallelizable; Autopilot
        # cost model still has final say at plan time.
        enable_dist = bool(spec.parallelizable and spec.partition_items)
        if enable_dist:
            rationale.append(
                "parallelizable workload — Autopilot may choose distributed "
                "if communication cost model agrees"
            )
            tradeoffs.append(
                "distributed adds network/queue overhead; refused when "
                "predicted speedup < 1.15x"
            )

        # Force CUDA device override only when hardware has it; never invent.
        if device == "cuda" and not hardware.cuda_available:
            device = "cpu"
            rationale.append("downgraded cuda→cpu because hardware.cuda_available=False")

        return OptimizationProfile(
            mode="optimized",
            policy_override=policy,
            dtype_override=dtype,
            device_override=device,
            batch_size=max(1, int(bs)),
            enable_distributed=enable_dist,
            max_parallel_workers=spec.max_parallel_workers,
            memory_budget_bytes=budget,
            rationale=rationale,
            tradeoffs=tradeoffs,
        )

    def apply_to_spec(
        self, spec: WorkloadSpec, profile: OptimizationProfile
    ) -> WorkloadSpec:
        """Return a copy of spec with adaptive knobs applied."""
        kwargs = {
            "batch_size": profile.batch_size,
            "adaptive_mode": profile.mode,
        }
        # WorkloadSpec fields we add: batch_size, adaptive_mode, force_* 
        # Applied via replace when fields exist.
        updates = {}
        if hasattr(spec, "batch_size"):
            updates["batch_size"] = profile.batch_size
        if hasattr(spec, "force_policy"):
            updates["force_policy"] = profile.policy_override
        if hasattr(spec, "force_dtype"):
            updates["force_dtype"] = profile.dtype_override
        if hasattr(spec, "force_device"):
            updates["force_device"] = profile.device_override
        if hasattr(spec, "adaptive_mode"):
            updates["adaptive_mode"] = profile.mode
        if profile.enable_distributed:
            updates["parallelizable"] = True
        else:
            # Unoptimized: ensure distributed stays off even if caller set it.
            if profile.mode == "unoptimized":
                updates["parallelizable"] = False
        if profile.max_parallel_workers is not None:
            updates["max_parallel_workers"] = profile.max_parallel_workers
        return replace(spec, **updates) if updates else spec
