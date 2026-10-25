"""Estimate a model's real size without downloading its weights.

Only the model's ``config.json`` (a few KB) is fetched — never the weight
files — so this is fast and gives a real answer before committing to a
multi-GB download. The parameter count is an analytical estimate from the
architecture's shape (standard transformer decoder formulas), not measured
from actual weights, so it's reported as an *estimate* with the formula
used, never presented as an exact measured count.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bytes per parameter for each dtype a caller might load the model as.
# int4 is the real minimum granularity bitsandbytes/accelerate quantized
# loading supports today; anything finer isn't a thing yet.
BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


@dataclass
class ModelSizeEstimate:
    model_name: str
    param_count: int
    param_count_is_estimate: bool
    architecture: str | None
    bytes_by_dtype: dict  # {"fp32": bytes, "fp16": bytes, ...}
    warnings: list


def param_count_from_config(config: dict) -> tuple[int | None, bool]:
    """Analytically estimate parameter count from a HF config's fields.

    Returns ``(count, is_estimate)``. ``count`` is ``None`` when the config
    doesn't expose enough of the standard fields to even estimate (an
    unusual architecture) — callers must handle that, not assume a number.

    Formula (standard transformer decoder, covers GPT-NeoX/LLaMA/Mistral/
    GPT-2-family and close variants via field-name fallbacks):
      embeddings:  vocab_size * hidden_size  (x2 if embeddings aren't tied)
      per layer:   attention ~= 4 * hidden_size^2
                   mlp       ~= (3 if gated else 2) * hidden_size * intermediate_size
                   norms     ~= 4 * hidden_size (negligible, included for completeness)
      final norm:  hidden_size
    This is an approximation (ignores biases, exact attention head-count
    effects like grouped-query attention's smaller K/V projections) —
    typically within ~5-10% of the true count for common architectures,
    not exact. ``is_estimate`` is always True when a count is returned;
    there is no code path here that claims an exact measured count.
    """
    hidden_size = _first(config, ["hidden_size", "n_embd", "d_model"])
    num_layers = _first(config, ["num_hidden_layers", "n_layer", "num_layers"])
    vocab_size = _first(config, ["vocab_size"])
    if hidden_size is None or num_layers is None or vocab_size is None:
        return None, True

    intermediate_size = _first(
        config, ["intermediate_size", "n_inner", "ffn_dim"], default=hidden_size * 4
    )
    tie_embeddings = bool(_first(config, ["tie_word_embeddings"], default=True))
    is_gated_mlp = _first(
        config, ["hidden_act"], default=""
    ) in ("silu", "swish") or "gate_proj" in str(config.get("architectures", ""))

    embedding_params = vocab_size * hidden_size * (1 if tie_embeddings else 2)
    attn_params_per_layer = 4 * hidden_size * hidden_size
    mlp_multiplier = 3 if is_gated_mlp else 2
    mlp_params_per_layer = mlp_multiplier * hidden_size * intermediate_size
    norm_params_per_layer = 4 * hidden_size

    per_layer = attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer
    total = embedding_params + num_layers * per_layer + hidden_size

    return int(total), True


def _first(config: dict, keys: list, default=None):
    for key in keys:
        value = config.get(key) if isinstance(config, dict) else getattr(config, key, None)
        if value is not None:
            return value
    return default


def estimate_model_size(model_name: str) -> ModelSizeEstimate:
    """Fetch ``model_name``'s config (not its weights) and estimate its size
    at every dtype a caller might load it as."""
    warnings: list = []
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    config_dict = config.to_dict()

    param_count, is_estimate = param_count_from_config(config_dict)
    architecture = None
    archs = config_dict.get("architectures")
    if archs:
        architecture = archs[0]

    if param_count is None:
        warnings.append(
            f"could not estimate parameter count for '{model_name}' from its config "
            "fields (unrecognized architecture shape) — falling back is not possible; "
            "caller must handle a missing estimate."
        )
        param_count = 0

    bytes_by_dtype = {
        dtype: int(param_count * bytes_per_param)
        for dtype, bytes_per_param in BYTES_PER_PARAM.items()
    }

    return ModelSizeEstimate(
        model_name=model_name,
        param_count=param_count,
        param_count_is_estimate=is_estimate,
        architecture=architecture,
        bytes_by_dtype=bytes_by_dtype,
        warnings=warnings,
    )
