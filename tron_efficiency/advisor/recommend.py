"""Automatic policy selection: given a model and hardware, pick a config.

This is the layer that turns the raw mechanism (``BaselinePolicy`` /
``OffloadPolicy`` from ``tron_efficiency.memory.policies``) into something a
caller doesn't have to hand-tune. It decides: which dtype, which policy,
and — when offload is needed — what nominal ``memory_budget_bytes`` to pass
to ``OffloadPolicy`` (which itself reserves activation headroom; this layer
does not duplicate that logic, it feeds it).

The thresholds here are informed by exactly two confirmed real data points
from real CUDA hardware (see writeup/compute-efficiency-gpu-validation.md):
a 600MB total GPU budget failed (for every split tried) running
pythia-410m on a T4; a 1.2GB total budget succeeded. The true minimum
viable floor for *that* model+GPU pair is somewhere in between — unknown
precisely, and almost certainly different for other models/GPUs (more
layers/attention heads need more activation memory; different GPUs have
different fixed CUDA context overhead). This module does not pretend to
know that floor exactly: below the confirmed-failing point, it recommends
against GPU offload outright; between the two confirmed points, it still
allows a recommendation but marks it explicitly low-confidence; above the
confirmed-working point, confidence rises only gradually with distance
from that single validated data point — it is never reported as "proven"
for a case that hasn't been run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tron_efficiency.advisor.model_info import BYTES_PER_PARAM, estimate_model_size
from tron_efficiency.hardware.detect import HardwareInfo, detect_hardware

# The two confirmed real data points from GPU validation (pythia-410m, T4):
CONFIRMED_FAILING_GPU_BUDGET_BYTES = 600 * 1024 * 1024
CONFIRMED_WORKING_GPU_BUDGET_BYTES = 1200 * 1024 * 1024

# dtypes tried for GPU inference, smallest-footprint last. fp32 is excluded
# here deliberately -- real GPU inference essentially never uses it, and
# including it would make the "does it fit without offload" check overly
# pessimistic for the case that actually matters.
GPU_DTYPE_CANDIDATES = ["fp16", "int8", "int4"]
CPU_DTYPE_CANDIDATES = ["fp32", "fp16"]


@dataclass
class Recommendation:
    model_name: str
    estimated_param_count: int
    param_count_is_estimate: bool
    device: str  # "cuda" or "cpu"
    dtype: str
    policy: str  # "baseline" or "offload"
    memory_budget_bytes: int | None  # for "offload": nominal budget to pass to OffloadPolicy
    confidence: str  # "high", "medium", "low"
    rationale: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "estimated_param_count": self.estimated_param_count,
            "param_count_is_estimate": self.param_count_is_estimate,
            "device": self.device,
            "dtype": self.dtype,
            "policy": self.policy,
            "memory_budget_bytes": self.memory_budget_bytes,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "warnings": self.warnings,
        }


def _check_quantization_executability(rec: Recommendation) -> None:
    """int8/int4 quantized loading via bitsandbytes (BitsAndBytesConfig) is
    real and wired into both BaselinePolicy and OffloadPolicy -- not
    advisory-only. The one real constraint bitsandbytes itself imposes:
    4-bit quantization cannot combine with device_map offload (no
    fp32-cpu-offload escape hatch for int4, unlike int8), so recommend()
    never recommends policy="offload" with dtype="int4" (see
    _recommend_gpu) -- this check exists as a safety net in case that
    invariant is ever broken by a future edit, not because it's expected
    to fire in normal operation."""
    if rec.dtype == "int4" and rec.policy == "offload":
        rec.warnings.append(
            "int4 + offload is not a supported bitsandbytes combination -- this "
            "recommendation would fail if executed. This should not happen; "
            "recommend() is supposed to use int8 for the offload case."
        )


def recommend(
    model_name: str,
    hardware: HardwareInfo | None = None,
    gpu_budget_bytes_override: int | None = None,
    size_estimate=None,
) -> Recommendation:
    """``size_estimate`` lets callers (tests, or a caller that already has
    one) skip the network call ``estimate_model_size`` makes to fetch the
    model's config; production callers normally omit it."""
    hardware = hardware or detect_hardware()
    size = size_estimate or estimate_model_size(model_name)
    rationale: list = []
    warnings: list = list(size.warnings)

    if size.param_count == 0:
        warnings.append(
            "no usable parameter-count estimate for this model -- recommending "
            "the safest fallback (CPU baseline) rather than guessing."
        )
        return Recommendation(
            model_name=model_name,
            estimated_param_count=0,
            param_count_is_estimate=True,
            device="cpu",
            dtype="fp32",
            policy="baseline",
            memory_budget_bytes=None,
            confidence="low",
            rationale=["could not estimate model size; defaulting to the safest option"],
            warnings=warnings,
        )

    if hardware.cuda_available and hardware.gpus:
        rec = _recommend_gpu(model_name, size, hardware, gpu_budget_bytes_override, rationale, warnings)
    else:
        rec = _recommend_cpu(model_name, size, hardware, rationale, warnings)
    _check_quantization_executability(rec)
    return rec


def _recommend_gpu(model_name, size, hardware, gpu_budget_bytes_override, rationale, warnings) -> Recommendation:
    available_bytes = gpu_budget_bytes_override or hardware.gpus[0].total_memory_bytes
    rationale.append(f"CUDA available: {hardware.gpus[0].name}, {available_bytes:,} bytes considered")

    # Try the smallest footprint dtype first that still fits WITHOUT offload
    # at all, using OffloadPolicy's own default headroom reservation as the
    # bar (a baseline load has the same activation-overhead need offload
    # does -- there's no free lunch just by not offloading).
    from tron_efficiency.memory.policies import DEFAULT_GPU_ACTIVATION_HEADROOM_FRACTION

    usable_without_offload = available_bytes * (1 - DEFAULT_GPU_ACTIVATION_HEADROOM_FRACTION)
    for dtype in GPU_DTYPE_CANDIDATES:
        weight_bytes = size.bytes_by_dtype[dtype]
        if weight_bytes <= usable_without_offload:
            rationale.append(
                f"{dtype} weights ({weight_bytes:,} bytes) fit within the GPU's usable "
                f"budget ({usable_without_offload:,.0f} bytes after reserving headroom) "
                "without needing to offload anything."
            )
            return Recommendation(
                model_name=model_name,
                estimated_param_count=size.param_count,
                param_count_is_estimate=size.param_count_is_estimate,
                device="cuda",
                dtype=dtype,
                policy="baseline",
                memory_budget_bytes=None,
                confidence="medium" if dtype != "fp16" else "high",
                rationale=rationale,
                warnings=warnings,
            )

    # Nothing fits without offload -- recommend OffloadPolicy. Dtype here is
    # deliberately "int8", not int4: bitsandbytes' CPU-offload escape hatch
    # (llm_int8_enable_fp32_cpu_offload) only exists for 8-bit quantization.
    # 4-bit does not support combining with device_map offload at all --
    # OffloadPolicy raises UnsupportedDtypeCombination if asked to, so
    # recommending int4 here would hand back a config guaranteed to fail.
    dtype = "int8"
    rationale.append(
        f"even int4 weights ({size.bytes_by_dtype['int4']:,} bytes) don't fit without "
        "offload -- recommending OffloadPolicy with int8 (int4 doesn't support "
        "combining with offload in bitsandbytes)."
    )

    if available_bytes < CONFIRMED_FAILING_GPU_BUDGET_BYTES:
        confidence = "low"
        warnings.append(
            f"GPU budget ({available_bytes:,} bytes) is below the smallest total budget "
            f"ever confirmed to work ({CONFIRMED_WORKING_GPU_BUDGET_BYTES:,} bytes) and at "
            f"or below a budget already confirmed to FAIL for a similarly-sized model "
            f"({CONFIRMED_FAILING_GPU_BUDGET_BYTES:,} bytes) -- this is very likely to OOM "
            "regardless of dtype or split. CPU execution is probably the safer call."
        )
    elif available_bytes < CONFIRMED_WORKING_GPU_BUDGET_BYTES:
        confidence = "low"
        warnings.append(
            f"GPU budget ({available_bytes:,} bytes) falls between the one confirmed-"
            f"failing budget ({CONFIRMED_FAILING_GPU_BUDGET_BYTES:,} bytes) and the one "
            f"confirmed-working budget ({CONFIRMED_WORKING_GPU_BUDGET_BYTES:,} bytes) for "
            "this mechanism -- the real minimum floor is unknown, this could go either way."
        )
    elif available_bytes <= CONFIRMED_WORKING_GPU_BUDGET_BYTES * 3:
        confidence = "medium"
        rationale.append(
            "budget is above the one confirmed-working data point, within a few multiples "
            "of it -- reasonable extrapolation, not independently validated at this size."
        )
    else:
        confidence = "low"
        warnings.append(
            f"GPU budget ({available_bytes:,} bytes) is far beyond the only validated data "
            f"point ({CONFIRMED_WORKING_GPU_BUDGET_BYTES:,} bytes) -- this far from the one "
            "real measurement, confidence is low even though the mechanism should scale."
        )

    return Recommendation(
        model_name=model_name,
        estimated_param_count=size.param_count,
        param_count_is_estimate=size.param_count_is_estimate,
        device="cuda",
        dtype=dtype,
        policy="offload",
        memory_budget_bytes=int(available_bytes),
        confidence=confidence,
        rationale=rationale,
        warnings=warnings,
    )


def _recommend_cpu(model_name, size, hardware, rationale, warnings) -> Recommendation:
    rationale.append("no CUDA GPU detected -- recommending CPU execution")
    if not hardware.cpu_total_memory_bytes:
        warnings.append("CPU RAM could not be measured (psutil missing?) -- defaulting conservatively")
        return Recommendation(
            model_name=model_name,
            estimated_param_count=size.param_count,
            param_count_is_estimate=size.param_count_is_estimate,
            device="cpu",
            dtype="fp32",
            policy="baseline",
            memory_budget_bytes=None,
            confidence="low",
            rationale=rationale,
            warnings=warnings,
        )

    for dtype in CPU_DTYPE_CANDIDATES:
        weight_bytes = size.bytes_by_dtype[dtype]
        # Generous margin: CPU RAM isn't hard-capped the way VRAM is in our
        # runner, but a model that doesn't comfortably fit still risks
        # starving the OS/other processes, so this stays conservative.
        if weight_bytes * 1.5 <= hardware.cpu_total_memory_bytes:
            rationale.append(
                f"{dtype} weights ({weight_bytes:,} bytes) fit comfortably in "
                f"{hardware.cpu_total_memory_bytes:,} bytes of system RAM."
            )
            return Recommendation(
                model_name=model_name,
                estimated_param_count=size.param_count,
                param_count_is_estimate=size.param_count_is_estimate,
                device="cpu",
                dtype=dtype,
                policy="baseline",
                memory_budget_bytes=None,
                confidence="medium",
                rationale=rationale,
                warnings=warnings,
            )

    rationale.append("model too large for comfortable in-RAM baseline -- recommending CPU disk offload")
    warnings.append(
        "recommending CPU+disk offload for a model this large is untested at scale "
        "(validated only with pythia-70m-sized models on CPU, see "
        "writeup/compute-efficiency-sample-report.md) -- expect it to be slow."
    )
    return Recommendation(
        model_name=model_name,
        estimated_param_count=size.param_count,
        param_count_is_estimate=size.param_count_is_estimate,
        device="cpu",
        dtype="fp32",
        policy="offload",
        memory_budget_bytes=int(hardware.cpu_total_memory_bytes * 0.7),
        confidence="low",
        rationale=rationale,
        warnings=warnings,
    )
