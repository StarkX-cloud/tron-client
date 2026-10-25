# GPU validation: the critical experiment, confirmed on real CUDA hardware

This fills the one gap the v0.1 write-up (`compute-efficiency-sample-report.md`)
could not close: the development machine that built `tron_efficiency/` has
no CUDA GPU, so the spec's actual "critical experiment" — *does a baseline
fail under a GPU memory budget that a memory-constrained path survives?* —
had only been tested via its CPU/disk-offload analog.

This run used a real NVIDIA T4 (Google Colab's free tier, ~14.56 GiB visible
to the process; CUDA available: true) to test that question directly.

## Setup

Same two mechanisms `tron_efficiency/memory/policies.py` implements
(`BaselinePolicy`: load the full model, no budget; `OffloadPolicy`:
`accelerate`'s real `device_map="auto"` + `max_memory`), run as a standalone
validation script against `EleutherAI/pythia-410m` (~1.64GB in fp32 — bigger
than the `pythia-70m` used for the CPU runs, so there's real pressure to
test against). This was **not** a literal execution of the committed
`tron_efficiency` package (no GPU box has it installed yet) — it's the same
mechanism, reproduced directly, specifically to validate the untested code
path before trusting it.

To simulate "this process only has a small GPU" on a T4 that actually has
~14.56GiB free, `torch.cuda.set_per_process_memory_fraction()` was used to
impose a real, hard, OS-enforced VRAM ceiling on the process — not a
reported/simulated number, an actual allocator-level cap that real CUDA
allocations fail against.

## What was found (in order — each run informed the next)

**Run 1 — 600MB hard cap, offload given the same 600MB as its weight
budget:** both policies failed. Baseline: `OutOfMemoryError: CUDA out of
memory. Tried to allocate 20.00 MiB` (fails after nearly filling the cap).
Offload: also `OutOfMemoryError`, failing even earlier, during weight
dispatch. **Real finding:** handing `accelerate`'s `max_memory` a weight
budget equal to the hard process cap leaves zero headroom for CUDA's own
context/cuBLAS overhead and activation tensors — a naive "memory budget ==
weight budget" design OOMs too, even with a much smaller actual weight
footprint than the baseline. This is a genuine design bug worth fixing in
`OffloadPolicy`, not an artifact of the test.

**Run 2 — reserving headroom (200MB weight budget inside the same 600MB
cap):** still both failed. 600MB is simply too tight once CUDA's own
context overhead is accounted for — confirming the overhead itself, not
just the lack of headroom, was the binding constraint at that scale.

**Run 3 — 1.2GB hard cap, 500MB weight budget (leaving ~700MB headroom):**

```
--- BASELINE ---
{
  "success": false,
  "error": "OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB"
}

--- OFFLOAD ---
{
  "success": true,
  "load_time_s": 0.3896,
  "e2e_s": 4.9422,
  "tok_per_s": 6.4748,
  "peak_gpu_bytes": 1029573632,
  "text": "The quick brown fox jumps over the lazy dog, and the dog jumps ov[er...]",
  "device_map": {
    "gpt_neox.embed_in": "0",
    "gpt_neox.layers.0": "0",
    "...": "...",
    "gpt_neox.layers.12": "cpu",
    "...": "cpu"
  }
}

=== SUMMARY ===
baseline success: False | offload success: True
```

**This is the spec's critical experiment, confirmed for real:**
- The baseline genuinely fails under the budget (real `CUDA out of memory`,
  not a simulated or predicted failure).
- The memory-constrained path genuinely succeeds under the *same* budget,
  by actually splitting the model across GPU and CPU (`device_map` shows
  early layers on GPU `"0"`, later layers on `"cpu"` — real placement, not
  a relabeled baseline).
- It produces coherent, correct generated text.
- Peak GPU memory (1.03GB) stayed under the 1.2GB cap, confirming the
  budget was actually respected, not just nominally requested.
- There is no baseline number to compare latency/throughput against here
  (baseline didn't complete), so this run cannot say *how much slower* the
  constrained path is relative to the baseline it replaced — only that it
  worked where the baseline could not. That comparison is what the CPU runs
  in `compute-efficiency-sample-report.md` showed (constrained ~5.8x slower
  there), on a different model/budget — the two results are about different
  questions and shouldn't be merged into one number.

## Re-verifying the fix (same session, after the headroom fix landed)

After `resolve_max_memory`/`OffloadPolicy` were changed to reserve
`gpu_activation_headroom_fraction` (default 0.6) of the nominal budget
before handing the rest to `accelerate`, Run 1's exact failing scenario was
replayed on the same T4: a 600MB hard process-wide VRAM cap, with
`OffloadPolicy` given 600MB as `memory_budget_bytes` (so accelerate now
only receives `600MB * 0.4 = 240MB` as the weight budget, instead of the
full 600MB that caused the original failure).

**Result: still fails.**

```
--- BASELINE (unchanged) ---
{ "success": false, "error": "OutOfMemoryError: CUDA out of memory. Tried to allocate 100.00 MiB" }

--- OFFLOAD (FIXED: only 251658240 bytes handed to accelerate, not the full 600MB) ---
{ "success": false, "error": "OutOfMemoryError: CUDA out of memory. Tried to allocate 102.00 MiB" }

=== VERIFICATION RESULT ===
baseline success: False (expected: False, same as original Run 1)
offload success: False (expected: True -- this is what the fix should change)
FIX DID NOT RESOLVE IT: offload still fails at 600MB even with headroom reserved.
```

**What this actually means, precisely:** the headroom-reservation fix is
real and does what it's supposed to — it stopped handing accelerate a
weight budget with zero margin, and that is exactly why Run 3 (1.2GB cap,
500MB weight budget, ~700MB headroom) succeeded. But reserving headroom
only helps when the *total* budget is large enough that some valid split
exists. At 600MB, even the most aggressive, correctly-proportioned split
tried here (240MB weights / 360MB headroom) still isn't enough — this
specific GPU + model + framework combination has a fixed overhead floor
somewhere above 600MB and at or below 1.2GB, and no re-proportioning of a
budget below that floor can succeed. This matches the original Run 2
finding (200MB weight budget inside a 600MB cap also failed) — it wasn't
really about the *split*, it was that 600MB itself was never a viable
total budget for this model on this GPU.

**So, honestly: the fix is confirmed to do what it claims (reserve real
headroom, and that headroom is what made the 1.2GB case succeed), but it
is not a fix for "every budget now works" — it never claimed to be, but
it's worth stating plainly since it would be easy to over-read "fix
confirmed" as "offload can now survive any budget baseline can't."**

## Running the real installed package end-to-end on GPU (closing the last gap)

Everything above used a standalone script that reproduced the mechanism —
not the committed `tron_efficiency` package itself. That gap is closed now:
the `feat/compute-efficiency-engine` branch was pushed to GitHub, cloned
into a fresh Colab T4 runtime, installed via `pip install -r
requirements-training.txt`, and run through its real CLI — no mirrored
code, the actual package.

```bash
python -m tron_efficiency.cli recommend --model EleutherAI/pythia-1.4b
python -m tron_efficiency.cli compare --model EleutherAI/pythia-1.4b \
    --device cuda --memory-budget-mb 1500 --max-new-tokens 16 --hourly-rate 0.50
```

`recommend` (real advisor, real hardware detection) correctly judged this
1.4B-param model fits on a full T4 without offload: `device=cuda,
dtype=fp16, policy=baseline, confidence=high` — fp16 weights (2,828,406,784
bytes estimated) comfortably under the GPU's usable budget. A constrained
`--memory-budget-mb 1500` was used for `compare` specifically to force the
offload path too, so both policies could be tested for real on a bigger,
commercially-relevant model size (1.4B vs. the 70M/410M used earlier).

**Real results (both succeeded, output verified identical by the engine's
own correctness check — no divergence warning was raised):**

| Metric | baseline | offload | Winner |
|---|---|---|---|
| Model load time (s) | 34.63 | 1.085 | offload |
| Time to first token (s) | 1.402 | 0.7267 | offload |
| End-to-end latency, 16 tokens (s) | 0.3305 | 10.09 | baseline |
| Throughput (tokens/sec) | 48.42 | 1.585 | baseline |
| Peak CPU memory (bytes) | 3.715e9 (~3.46GB) | 4.338e9 (~4.04GB) | baseline |
| Peak GPU memory allocated (bytes) | 2.844e9 (~2.65GB) | 6.285e8 (~586MB) | offload |

Estimated cost at a configured $0.50/hour: baseline $0.000046/run, offload
$0.001402/run (estimates, not measured cloud prices).

**What this confirms, precisely:**
- The headroom fix works correctly in the *real installed package*, not
  just the standalone script: offload's actual peak GPU memory (586MB)
  lands almost exactly on the computed weight budget
  (`1500MB * (1 - 0.6) = 600MB`), confirming the reservation is doing what
  it's supposed to at a new model size, end to end through the CLI.
- At this bigger, real-GPU-throughput scale, the offload penalty is
  **steeper** than the CPU case: ~30.5x slower throughput here (48.42 vs.
  1.585 tok/s) versus ~5.8x on CPU — because the baseline here runs at real
  GPU speed (fast), so the relative cost of falling back to per-layer disk
  reads during generation is much more visible than it was against an
  already-slow CPU baseline.
- Offload's real GPU memory win is genuine and large here: **586MB vs.
  2.65GB, a ~4.8x reduction** — unlike the CPU run in
  `compute-efficiency-sample-report.md` where the memory win was marginal.
  This is the clearest positive memory-efficiency result in this project so
  far: real GPU memory saved, at a real, honestly-reported throughput cost.
- Output correctness held: the benchmark engine's own cross-policy text
  comparison found no divergence between baseline and offload generations.

**A real gap this surfaced, not previously documented:** `BaselinePolicy`
doesn't pass a `torch_dtype` to `from_pretrained` — it loads whatever dtype
the checkpoint happens to be stored in on the Hub. `pythia-1.4b`'s
checkpoint happens to be stored in fp16 (its `model.safetensors` download
was 2.93GB, matching fp16's ~2.83GB estimate, not fp32's ~5.66GB), which is
why the advisor's fp16 estimate (2,828,406,784 bytes) landed almost exactly
on the measured peak (2.844e9 bytes, 0.6% off) — a happy coincidence of
this particular checkpoint's storage format, not because `recommend()`'s
dtype choice is actually wired into how `BaselinePolicy` loads the model.
For a model whose checkpoint is stored in fp32, `BaselinePolicy` would
silently load fp32 regardless of what `recommend()` suggested. There is no
code path connecting the advisor's dtype recommendation to an actual
`torch_dtype=` argument anywhere in `tron_efficiency/memory/policies.py` —
this is a real, unimplemented piece of wiring, not covered by the
`int8`/`int4` "not yet implemented" warning (which is about quantized
loading specifically; this is about plain dtype selection, a smaller gap).

## What this changes in the project's "not yet done" list

- No real end-to-end confidentiality, no defense against a malicious
  participant, single-process master with no replication — all unrelated,
  unchanged (see ROADMAP.md's distributed-training section).
- "No real GPU validation" is no longer accurate as an unqualified
  statement — both the mechanism and the installed package have now run,
  for real, on CUDA hardware, for several model/budget pairs.
- `OffloadPolicy`'s real design gap found in Run 1 (no activation
  headroom reserved) — **fixed and re-verified** (see above), with an
  honestly-reported floor it doesn't eliminate.
- **New gap found during the real-package run:** `BaselinePolicy` doesn't
  control or report the dtype it actually loads a model in — it's whatever
  the checkpoint happens to be stored as. `recommend()`'s dtype
  recommendation is therefore advisory even for `baseline`, not just for
  the already-flagged `int8`/`int4` quantized case.
- Still only a handful of model sizes tested (70M, 410M, 1.4B); no
  systematic sweep across sizes/budgets/GPU types.
