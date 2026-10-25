"""Advisor (auto policy selection) tests — pure logic, no network/model load.

``param_count_from_config`` is tested against real architecture shapes
(GPT-NeoX-style and LLaMA-style configs) with known real parameter counts,
so the formula's accuracy is pinned, not just "it returns a number."
``recommend`` is tested via dependency injection (``size_estimate``, a fake
``HardwareInfo``) so none of this touches the network or loads a model.
"""

from tron_efficiency.advisor.model_info import (
    BYTES_PER_PARAM,
    ModelSizeEstimate,
    param_count_from_config,
)
from tron_efficiency.advisor.recommend import (
    CONFIRMED_FAILING_GPU_BUDGET_BYTES,
    CONFIRMED_WORKING_GPU_BUDGET_BYTES,
    recommend,
)
from tron_efficiency.hardware.detect import GPUInfo, HardwareInfo

# Real EleutherAI/pythia-70m config fields (GPT-NeoX architecture, untied
# embeddings) -- real measured param count is 70,426,624 (verified directly
# against the loaded model during development, see
# writeup/compute-efficiency-gpu-validation.md's sibling work).
PYTHIA_70M_CONFIG = {
    "hidden_size": 512,
    "num_hidden_layers": 6,
    "vocab_size": 50304,
    "intermediate_size": 2048,
    "tie_word_embeddings": False,
    "hidden_act": "gelu",
    "architectures": ["GPTNeoXForCausalLM"],
}
PYTHIA_70M_REAL_PARAM_COUNT = 70_426_624


def _fake_cpu_hardware(ram_gb=16):
    return HardwareInfo(
        platform="fake", python_version="3.11", cpu_count=8,
        cpu_total_memory_bytes=ram_gb * 1024**3, cuda_available=False,
        torch_version="2.x", gpus=[], warnings=[],
    )


def _fake_gpu_hardware(vram_mb):
    return HardwareInfo(
        platform="fake", python_version="3.11", cpu_count=8,
        cpu_total_memory_bytes=16 * 1024**3, cuda_available=True,
        torch_version="2.x",
        gpus=[GPUInfo(index=0, name="Fake GPU", total_memory_bytes=vram_mb * 1024 * 1024)],
        warnings=[],
    )


def _fake_size_estimate(param_count, model_name="fake/model"):
    bytes_by_dtype = {dtype: int(param_count * bpp) for dtype, bpp in BYTES_PER_PARAM.items()}
    return ModelSizeEstimate(
        model_name=model_name, param_count=param_count, param_count_is_estimate=True,
        architecture="FakeArch", bytes_by_dtype=bytes_by_dtype, warnings=[],
    )


def test_param_count_formula_matches_real_pythia_70m_within_one_percent():
    count, is_estimate = param_count_from_config(PYTHIA_70M_CONFIG)
    assert is_estimate is True
    assert count is not None
    error = abs(count - PYTHIA_70M_REAL_PARAM_COUNT) / PYTHIA_70M_REAL_PARAM_COUNT
    assert error < 0.01, f"estimate {count} vs real {PYTHIA_70M_REAL_PARAM_COUNT}, {error:.2%} off"


def test_param_count_formula_handles_gated_mlp_llama_style():
    llama_like_config = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "intermediate_size": 11008,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "architectures": ["LlamaForCausalLM"],
    }
    count, _ = param_count_from_config(llama_like_config)
    # Llama-7B's real count is ~6.7B; this formula should land in the right
    # ballpark (same order of magnitude, not exact -- it's an estimate).
    assert 5_000_000_000 < count < 8_000_000_000


def test_param_count_formula_returns_none_for_unrecognized_config():
    count, _ = param_count_from_config({"some_unrelated_field": 123})
    assert count is None


def test_recommend_cpu_baseline_for_small_model_no_gpu():
    rec = recommend(
        "fake/small-model",
        hardware=_fake_cpu_hardware(ram_gb=16),
        size_estimate=_fake_size_estimate(70_000_000),
    )
    assert rec.device == "cpu"
    assert rec.policy == "baseline"
    assert rec.memory_budget_bytes is None
    assert not any("quantized" in w for w in rec.warnings)


def test_recommend_cpu_offload_for_model_too_big_for_ram():
    rec = recommend(
        "fake/huge-model",
        hardware=_fake_cpu_hardware(ram_gb=4),
        size_estimate=_fake_size_estimate(50_000_000_000),  # 50B params, way over 4GB RAM
    )
    assert rec.device == "cpu"
    assert rec.policy == "offload"
    assert rec.memory_budget_bytes is not None
    assert rec.confidence == "low"


def test_recommend_gpu_baseline_when_it_fits_without_offload():
    # 500M params -> ~1GB in fp16, comfortably under 16GB*(1-0.6)=6.4GB usable.
    rec = recommend(
        "fake/model",
        hardware=_fake_gpu_hardware(vram_mb=16000),
        size_estimate=_fake_size_estimate(500_000_000),
    )
    assert rec.device == "cuda"
    assert rec.policy == "baseline"
    assert rec.dtype == "fp16"
    assert rec.confidence == "high"


def test_recommend_gpu_offload_below_confirmed_failing_budget_is_low_confidence():
    rec = recommend(
        "fake/model",
        hardware=_fake_gpu_hardware(vram_mb=150),
        size_estimate=_fake_size_estimate(400_000_000),
    )
    assert rec.device == "cuda"
    assert rec.policy == "offload"
    assert rec.confidence == "low"
    assert any("OOM" in w or "FAIL" in w for w in rec.warnings)


def test_recommend_gpu_offload_at_confirmed_working_budget_is_not_low_confidence():
    rec = recommend(
        "fake/model",
        hardware=_fake_gpu_hardware(vram_mb=CONFIRMED_WORKING_GPU_BUDGET_BYTES // (1024 * 1024)),
        size_estimate=_fake_size_estimate(2_000_000_000),  # too big to fit even int4 at this budget
    )
    assert rec.policy == "offload"
    assert rec.confidence in ("medium", "high")


def test_recommend_warns_when_quantized_dtype_recommended():
    rec = recommend(
        "fake/model",
        hardware=_fake_gpu_hardware(vram_mb=700),
        size_estimate=_fake_size_estimate(400_000_000),
    )
    assert rec.dtype in ("int8", "int4")
    assert any("not yet implemented" in w for w in rec.warnings)


def test_recommend_handles_unestimatable_model_gracefully():
    unknown = ModelSizeEstimate(
        model_name="fake/weird-model", param_count=0, param_count_is_estimate=True,
        architecture=None, bytes_by_dtype={}, warnings=["could not estimate"],
    )
    rec = recommend("fake/weird-model", hardware=_fake_cpu_hardware(), size_estimate=unknown)
    assert rec.device == "cpu"
    assert rec.policy == "baseline"
    assert rec.confidence == "low"


def test_confirmed_budget_constants_are_consistent():
    assert CONFIRMED_FAILING_GPU_BUDGET_BYTES < CONFIRMED_WORKING_GPU_BUDGET_BYTES
