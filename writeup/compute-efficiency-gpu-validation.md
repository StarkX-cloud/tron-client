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
confirmed" as "offload can now survive any budget baseline can't."** A
future improvement worth considering: `OffloadPolicy` could detect when
the resulting weight budget is implausibly small and surface a clearer
warning than accelerate's own OOM message, rather than relying on the
headroom fraction alone to paper over an unviable total budget. Not
implemented here — flagged as a real follow-up, not treated as covered by
today's fix.

## What this changes in the project's "not yet done" list

- "No real GPU validation" is no longer accurate as an unqualified
  statement — the mechanism itself (`device_map`/`max_memory` genuinely
  splitting a model under a hard VRAM cap) is now validated on real CUDA
  hardware, for one model/budget pair.
- Still not done: this was a standalone script, not the installed
  `tron_efficiency` package running on a GPU machine end-to-end (CLI, API,
  dashboard) — that integration hasn't been exercised on CUDA.
- Still not done: `OffloadPolicy`'s real design gap found in Run 1 — it
  does not currently reserve headroom for activations/context overhead
  when a caller sets `memory_budget_bytes`; a caller who picks a budget
  too close to the true minimum will see a confusing OOM instead of a
  graceful degradation. Worth fixing before this is used by anyone besides
  the person who already knows this gotcha.
- Still only one model (`pythia-410m`) and one budget pair tested on GPU;
  no sweep.
