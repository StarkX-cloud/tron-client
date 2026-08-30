"""Run this to reproduce the Phase 3 scale-up benchmark:

    python -m tron.training.benchmark_lora

Same comparison as benchmark.py (local SGD vs. a sync-every-step-shaped
baseline, plus zero-communication weight merging), applied to a real
small open-weight model (EleutherAI's Pythia-70M) via LoRA on the
tiny-shakespeare corpus, instead of a hand-rolled numpy MLP on synthetic
data. See tron/training/lora_demo.py for the mechanism and
ARCHITECTURE.md for what this does and doesn't claim.
"""
from __future__ import annotations

import torch

from .lora_demo import (
    MODEL_NAME,
    adapter_state_bytes,
    full_model_bytes,
    load_base_model_and_tokenizer,
    load_corpus_token_ids,
    make_shards,
    run_independent_lora_and_merge,
    run_local_sgd_lora,
)

NUM_SHARDS = 3
NUM_ROUNDS = 4
LOCAL_STEPS = 5
LR = 5e-4


def run() -> dict:
    base_model, tokenizer = load_base_model_and_tokenizer()
    token_ids = load_corpus_token_ids(tokenizer)

    # Reserve the final slice of the corpus as held-out eval data, never
    # seen by any shard.
    held_out_fraction = 0.1
    split_at = int(len(token_ids) * (1 - held_out_fraction))
    train_ids, held_out_ids = token_ids[:split_at], token_ids[split_at:]

    shards = make_shards(train_ids, num_shards=NUM_SHARDS)
    held_out_blocks = make_shards(held_out_ids, num_shards=1)[0]

    local_sgd_result = run_local_sgd_lora(
        base_model, shards, held_out_blocks,
        num_rounds=NUM_ROUNDS, local_steps=LOCAL_STEPS, lr=LR,
    )

    merge_result = run_independent_lora_and_merge(
        base_model, shards, held_out_blocks,
        num_steps=NUM_ROUNDS * LOCAL_STEPS, lr=LR,
    )

    adapter_bytes = adapter_state_bytes(local_sgd_result.final_model)
    full_bytes = full_model_bytes(local_sgd_result.final_model)
    hypothetical_full_sync_bytes = NUM_SHARDS * full_bytes * NUM_ROUNDS

    return {
        "model": MODEL_NAME,
        "adapter_bytes_per_sync_per_shard": adapter_bytes,
        "full_model_bytes": full_bytes,
        "local_sgd": {
            "eval_loss_before": local_sgd_result.eval_loss_before,
            "eval_loss_after": local_sgd_result.eval_loss_after,
            "comm_bytes": local_sgd_result.comm_bytes,
            "num_syncs": local_sgd_result.num_syncs,
            "wall_clock_seconds": local_sgd_result.wall_clock_seconds,
        },
        "hypothetical_full_model_sync_bytes": hypothetical_full_sync_bytes,
        "merge": {
            "eval_loss_before": merge_result.eval_loss_before,
            "solo_eval_losses": merge_result.solo_eval_losses,
            "merged_eval_loss": merge_result.merged_eval_loss,
            "wall_clock_seconds": merge_result.wall_clock_seconds,
        },
    }


def _print_report(r: dict) -> None:
    print("=" * 72)
    print(f"TRON Phase 3 scale-up benchmark - {r['model']} + LoRA on tiny-shakespeare")
    print(f"({NUM_SHARDS} shards, {NUM_ROUNDS} rounds x {LOCAL_STEPS} local steps)")
    print("=" * 72)

    ls = r["local_sgd"]
    reduction = r["hypothetical_full_model_sync_bytes"] / ls["comm_bytes"]
    print(f"Adapter size: {r['adapter_bytes_per_sync_per_shard']:,} bytes  "
          f"(full model: {r['full_model_bytes']:,} bytes)")
    print(f"Local SGD: loss {ls['eval_loss_before']:.4f} -> {ls['eval_loss_after']:.4f}, "
          f"{ls['comm_bytes']:,} bytes over {ls['num_syncs']} syncs")
    print(f"  vs. a hypothetical full-model sync at the same cadence: "
          f"{r['hypothetical_full_model_sync_bytes']:,} bytes ({reduction:.0f}x more)")

    m = r["merge"]
    avg_solo = sum(m["solo_eval_losses"]) / len(m["solo_eval_losses"])
    print(f"\nMerge: solo shard losses {[round(x, 4) for x in m['solo_eval_losses']]} "
          f"(avg {avg_solo:.4f}) -> merged {m['merged_eval_loss']:.4f}, zero comm during training")


if __name__ == "__main__":
    _print_report(run())
