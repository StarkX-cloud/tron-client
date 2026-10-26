# TRON

TRON is a distributed execution runtime for Python, being rebuilt around a
single goal: **run real compute and training workloads across
heterogeneous, unreliable machines with near-zero communication overhead —
with a live, causal 3D replay of what the cluster is doing.**

See [ARCHITECTURE.md](ARCHITECTURE.md) for the technical design and
[ROADMAP.md](ROADMAP.md) for what's built vs. planned. Short version: all
four phases have a real first version built and tested — the remaining
work per phase (a bandwidth prober, real multi-machine training, 3D
interaction, etc.) is tracked in ROADMAP.md rather than glossed over.

This is not a rental-compute marketplace — an earlier billing/royalty layer
explored that idea and has been removed. See ARCHITECTURE.md's "What was
cut" section for why.

## What's here today

```python
import tron

tron.ensure_server()  # connects to an existing server, or starts a local one

@tron.remote
def expensive_task(x):
    return x * 2

result = expensive_task(10).get()
```

- **SDK** (`tron/`): `@tron.remote` + `MagicFuture` make remote execution
  look like a normal function call. Functions are shipped via cloudpickle.
- **Server** (`queue_server.py`): FastAPI job queue, worker registration
  and heartbeat, a watchdog that recovers a dead worker's in-flight work.
- **Execution spine** (`tron/spine/`): content-addressed artifacts,
  append-only replayable event log, real latency-aware placement
  (`topology.py`), and a periodic optimal-assignment match step
  (`matcher.py`, via `scipy.optimize.linear_sum_assignment`) for actual
  cross-worker job placement. `/spine/events` and `/spine/task/{id}`
  expose the raw log — this is the substrate Phase 4's Grid builds on.
- **Training demo** (`tron/training/`): DiLoCo-style local SGD vs. a
  sync-every-step baseline, and weight-space merging (task arithmetic +
  TIES) — the small-scale, benchmarked proof of the "train across
  unreliable heterogeneous nodes with near-zero communication" claim.
  Run `python -m tron.training.benchmark` for the report.
  `python -m tron.training.benchmark_lora` runs the same comparison
  against a real pretrained model (EleutherAI's Pythia-70M) via LoRA —
  a 717x smaller unit of communication than the full model, verified
  with a real recorded run (`requirements-training.txt`; ~20 minutes on
  CPU). See ARCHITECTURE.md for the numbers and what's honestly still
  missing (real multi-machine execution).
- **TRON-II** (`tron/orchestrator/`): training orchestration with
  pluggable adapters (Ray, SB3, scikit-learn, Transformers) and an
  outcome-tracking loop that scores adapters by how close their predicted
  cost/capability gain was to reality. Not yet wired to a real training
  workload — see ROADMAP.md.
- **vGPU** (`tron/gpu/`): a simulation layer that aggregates multiple
  GPUs' reported specs into one synthetic profile, plus an
  OpenAI-compatible API gateway on top. Explicitly a simulation, not GPU
  virtualization.
- **The Grid** (`tron/grid/`, served at `/grid/`): a 3D, time-scrubbable
  replay of the execution spine's event log. Every position is derived
  from real data — worker distance from measured latency, worker height
  from load, task position from which worker the log says it's actually
  on — not a decorative layout. Passive replay only for now; see
  ARCHITECTURE.md for what's deferred.

## Install

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # or: source .venv/bin/activate
pip install -r requirements.txt
```

For running tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## Run the server

```bash
python queue_server.py
```

Starts on `http://0.0.0.0:9000` by default (`TRON_PORT` / `TRON_HOST` to
change). Point the SDK at it:

```python
import tron
tron.set_config_url("http://localhost:9000")
```

Or with Docker:

```bash
docker compose up --build
```

## Worker bootstrap

```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/starkgenex-alt/TRON/main/install_tron.sh \
  | TRON_MASTER_URL=http://127.0.0.1:9000 START_WORKER=true bash
```

```powershell
# Windows
$env:TRON_MASTER_URL='http://127.0.0.1:9000'
$env:START_WORKER='true'
python install_tron.py
```

## Compute Efficiency Engine (v0.1)

A separate, standalone initiative living in `tron_efficiency/` — not part of
the distributed-training substrate above, and with zero coupling to
`queue_server.py`. It answers one question honestly: **can a
memory-constrained execution path run an LLM inference workload under a
tighter memory budget than a plain baseline, and what does that cost in
performance?** It is instrumentation to find out, not a claim that TRON has
already solved this.

Install the same optional deps used by the training demo above
(`torch`, `transformers`, `accelerate`, `psutil`, already in
`requirements-training.txt`), then:

```bash
python -m tron_efficiency.cli hardware                 # what this machine actually has
python -m tron_efficiency.cli recommend --model EleutherAI/pythia-70m
python -m tron_efficiency.cli run --policy baseline     # plain load + generate
python -m tron_efficiency.cli run --policy offload --memory-budget-mb 100
python -m tron_efficiency.cli compare --memory-budget-mb 100 --hourly-rate 0.50
python -m tron_efficiency.cli sweep --memory-budget-mb-list 50,100,200,400
python -m tron_efficiency.cli serve                      # API + dashboard on :9100 (TRON_EFFICIENCY_PORT)
```

The dashboard is served at `http://localhost:9100/dashboard/`, mounted the
same way `/grid` is mounted on `queue_server.py` — a separate static
directory, no shared frontend framework.

### Automatic policy selection (`tron_efficiency/advisor/`)

`recommend --model <name>` answers "what config should I even use?" without
hand-tuning a memory budget: it fetches the model's `config.json` (not its
weights — fast, no multi-GB download) and analytically estimates its real
parameter count from the architecture's shape, validated to within 0.04%
against `pythia-70m`'s actual measured size (70,398,464 estimated vs.
70,426,624 real). From that, plus detected (or overridden) hardware, it
picks a device, dtype, and policy, and — for `OffloadPolicy` — the nominal
`memory_budget_bytes` to pass it.

Every recommendation carries a `confidence` (`high`/`medium`/`low`) and
explicit `warnings`, calibrated against the only two real GPU data points
this project has (see the GPU validation write-up): a 600MB total budget
confirmed to fail, a 1.2GB total budget confirmed to work, for
`pythia-410m` on a T4. Below the confirmed-failing point: low confidence,
explicit "probably won't work" warning. Between the two: low confidence,
"unknown, could go either way." Near the confirmed-working point: medium.
Far beyond it (bigger GPUs, bigger models): still only medium-to-low,
because confidence shouldn't rise just because a number is larger — it's
still extrapolation past the one thing that was actually measured.

**Quantized loading is real, not advisory.** `int8`/`int4` recommendations
are backed by actual `bitsandbytes` quantization (`BitsAndBytesConfig`,
the same mechanism anyone using `transformers` has access to) — both
`BaselinePolicy` and `OffloadPolicy` load real quantized weights when
`WorkloadConfig.dtype` requests it (`--dtype int8`/`int4` on the CLI). One
real constraint carries through honestly: bitsandbytes doesn't support
combining 4-bit quantization with device-map offload (only 8-bit has the
`llm_int8_enable_fp32_cpu_offload` escape hatch), so `recommend()` never
pairs `int4` with the offload policy, and `OffloadPolicy` itself raises a
clear `UnsupportedDtypeCombination` (not a confusing crash) if asked to.
Verified locally end-to-end: a real int8-quantized `pythia-70m` loads and
generates correctly via `run_workload`, and the int4+offload rejection
fails with the intended clear message rather than a stack trace.

**Two real execution paths**, not one baseline relabeled with different
numbers: `BaselinePolicy` loads the full model onto the device with no
budget; `OffloadPolicy` uses Hugging Face Accelerate's real
`device_map="auto"` + `max_memory` + `offload_folder` mechanism to keep
resident memory under a stated budget, spilling the rest to disk. Every
run is checked for output correctness across policies (identical greedy
decoding output, not just similar timing) before it's reported as a
performance comparison.

**Honest limitation:** the development machine this was built on has no
CUDA GPU (`torch.cuda.is_available()` is `False`). The spec's
GPU-memory-budget experiment is implemented verbatim for a CUDA machine
(`OffloadPolicy` targets GPU VRAM via `max_memory={0: budget, ...}` when
`device="cuda"`), and could only be *exercised* locally via its real CPU
analog — constraining CPU RAM and forcing genuine disk offload. Every
hardware report, result JSON, and generated report says this explicitly;
nothing here claims a GPU result that wasn't actually measured on a GPU.

**GPU path since validated on real CUDA hardware** (a free-tier NVIDIA T4,
not this dev machine — see
[writeup/compute-efficiency-gpu-validation.md](writeup/compute-efficiency-gpu-validation.md)):
with a real, OS-enforced 1.2GB VRAM cap on the process and
`EleutherAI/pythia-410m` (~1.64GB in fp32), the baseline genuinely fails
with a real `CUDA out of memory` error, while the offload mechanism
(`device_map`/`max_memory`) genuinely succeeds under the identical cap by
splitting the model across GPU and CPU — the spec's actual "critical
experiment," confirmed, not simulated. That write-up also documents a real
design gap it surfaced: handing `OffloadPolicy` a weight budget with no
headroom reserved for activations/CUDA context overhead can OOM it too
(first attempt did, at a 600MB cap). **Fixed and re-verified on the real
T4**: `OffloadPolicy` now reserves `gpu_activation_headroom_fraction`
(default 0.6, configurable) of the nominal `memory_budget_bytes` before
handing the rest to accelerate as the GPU weight budget. Re-running the
original 1.2GB-cap scenario confirms the fix does what it claims — real
headroom is what made that case succeed. **But precision matters here**:
replaying the exact *failing* 600MB-cap scenario from the first attempt,
even with the fix applied (only 240MB handed to accelerate instead of the
full 600MB), **it still fails** — same `CUDA out of memory` error. That
600MB total budget is simply below a fixed-overhead floor for this
model+GPU combination; no re-proportioning of a too-small total budget can
rescue it. The fix is real and necessary (it's why 1.2GB works), but it is
not "every budget now works," and the write-up says so explicitly rather
than letting a single success imply more than it showed. See
[writeup/compute-efficiency-gpu-validation.md](writeup/compute-efficiency-gpu-validation.md)'s
"Re-verifying the fix" section for the full before/after numbers.

**The real installed package has now run end-to-end on GPU, closing that
specific gap.** `feat/compute-efficiency-engine` was pushed to GitHub,
cloned into a fresh Colab T4, installed normally, and run through its own
CLI (`recommend` + `compare`) against `EleutherAI/pythia-1.4b` — a bigger,
more commercially-relevant size than the 70M/410M used so far. The real
advisor correctly judged it fits without offload on a full T4; a
constrained `--memory-budget-mb 1500` forced the offload path too, for a
genuine baseline-vs-offload comparison at this scale: baseline hit
48.42 tok/s using ~2.65GB VRAM, offload hit 1.585 tok/s (~30.5x slower)
using only ~586MB VRAM (~4.8x less memory — the clearest real memory win
in this project so far, honestly paired with its real throughput cost).
Output correctness held (no divergence between the two policies' text).
Full numbers and a new gap this run surfaced (`BaselinePolicy` doesn't
control which dtype it actually loads — it's whatever the checkpoint
happens to be stored as, not something `recommend()`'s dtype choice
actually drives) are in the write-up's "Running the real installed package
end-to-end on GPU" section.

**Actual measured result** (`EleutherAI/pythia-70m`, CPU, 32 generated
tokens, 100MB CPU budget vs. the model's ~134MB footprint — see
[writeup/compute-efficiency-sample-report.md](writeup/compute-efficiency-sample-report.md)
for the full report and
[writeup/compute-efficiency-sample-run.csv](writeup/compute-efficiency-sample-run.csv)
for the raw data): the constrained path loads faster (accelerate defers
most weight materialization) but is **~5.8x slower end-to-end** (10.2s vs
1.8s; 3.1 vs 18.0 tokens/sec) because those deferred reads happen during
generation instead of up front, and peak CPU memory is only marginally
lower (464MB vs 487MB) — this tiny model's weights (~134MB) are dwarfed by
the Python/torch process's own overhead, so there isn't much headroom for
offloading to show a memory win. **This is a negative/mixed result,
reported as such** — not a case where TRON's approach "wins."

See `tests/test_efficiency_*.py` for pure-unit coverage (hardware, results,
memory policy logic — no model load, fast) and
`tests/test_efficiency_workload_real.py` for the real-model tests, gated
behind `TRON_EFFICIENCY_RUN_MODEL_TESTS=1` (not part of the default suite,
same convention as `tron/training/benchmark_lora.py`'s ~20-minute real run).

**Known gaps:**
- **Fixed since the gap above was found:** `WorkloadConfig.dtype` now
  flows into an actual `torch_dtype=`/`quantization_config=` argument in
  both policies (`tron_efficiency/memory/policies.py`'s
  `_loading_kwargs_for_dtype`) — `recommend()`'s dtype choice is no longer
  disconnected from how a model actually loads. `dtype=None` (the default)
  preserves the old pre-fix behavior exactly for backward compatibility.
- `int8` quantization combined with `OffloadPolicy` on CPU-only hardware
  is fragile in a way that isn't fully characterized: 100MB and
  110-120MB CPU budgets both fail, but with *different* errors (100MB:
  accelerate's "whole model to disk" refusal; 110-120MB:
  `RuntimeError: Cannot access accelerator device when none is
  available`, a CUDA-shaped error on a machine with no CUDA at all), while
  130MB succeeds. The 110-120MB failure mode specifically suggests
  bitsandbytes' `llm_int8_enable_fp32_cpu_offload` escape hatch may
  internally assume a CUDA accelerator is reachable even when only
  splitting between CPU and disk — a real quirk of a CUDA-oriented
  library being used in a CPU-only configuration, not something this
  project invented. The 130MB "success" could not be confirmed as a
  genuine disk-split placement either way (`hf_device_map` isn't
  populated the same way for quantized models as it is for plain-dtype
  offload, so the usual placement evidence isn't available). Properly
  characterizing this needs testing on real GPU hardware rather than
  further guessing on CPU-only hardware — flagged for the next GPU
  session, not resolved here.
- **Addressed, deliberately not with a numeric floor:** `OffloadPolicy`
  still accepts a budget too small to ever succeed under (there is no
  honest universal minimum byte count to check against — the real floor
  is device/model-dependent and only loosely bounded by the two confirmed
  data points above). What changed: `workloads/runner.py`'s
  `_clarify_error` now recognizes the specific confusing failures this
  actually produces (accelerate's "whole model to disk" refusal, CUDA OOM
  under offload) and rewrites them into actionable messages citing what
  was actually measured, instead of leaving the raw underlying error as
  the only information. A pre-flight numeric check would have meant
  guessing a number this project doesn't have evidence for.
- **Addressed for the budget dimension:** `cli sweep --memory-budget-mb-list
  100,150,250` (API: `POST /sweep`) runs the same workload across several
  budgets in one pass — a real tradeoff curve, not two isolated points.
  Real example in
  [writeup/compute-efficiency-budget-sweep-example.md](writeup/compute-efficiency-budget-sweep-example.md):
  `pythia-70m` on CPU holds steady at ~33 tok/s from 150-250MB, then hits
  a real cliff at 100MB — throughput collapses 8.6x (32.7→3.8 tok/s) for
  only an 8.6% memory saving. Sweeping *model sizes* still isn't
  automated (each model is its own multi-GB download with its own real
  cost — a deliberate choice to leave that explicit, not an oversight).
- **Also addressed for the quantization dimension:** `cli dtype-sweep
  --dtypes fp32,fp16,int8,int4` (API: `POST /dtype-sweep`) measures what
  quantization actually costs/saves, since it had only been confirmed to
  load and generate correctly before this, never benchmarked. Real result
  in
  [writeup/compute-efficiency-dtype-sweep-example.md](writeup/compute-efficiency-dtype-sweep-example.md):
  memory drops monotonically and genuinely with precision (804MB fp32 →
  589MB int4, a real 27% reduction), but **throughput does not** —
  int8 is 2.6x *slower* than fp32 on this CPU machine (6.03 vs. 15.45
  tok/s), likely because bitsandbytes' CPU kernels aren't as optimized as
  native PyTorch ops (a GPU would likely tell a different, faster story —
  not yet measured). "Quantize for a speed win" turned out to be the GPU
  story, not a universal one; measuring this before claiming it avoided
  shipping a false claim.
- Single model family tested per environment (`pythia-70m` on CPU,
  `pythia-410m`/`pythia-1.4b` on the GPU runs); still only a handful of
  models total.
- Peak CPU memory is sampled via a polling thread (50ms interval), not a
  true OS-level peak — Windows has no `resource.getrusage` equivalent.
- On this CPU-only dev machine specifically, the "critical experiment"
  (baseline fails, offload succeeds, same budget) couldn't be demonstrated
  locally — `pythia-70m`'s ~134MB footprint fits trivially in available
  RAM. It *was* demonstrated on real CUDA hardware instead (see above).

The single most valuable next experiment: run the actual `tron_efficiency`
package (not just the mechanism) end-to-end on a CUDA machine — CLI,
benchmark engine, API, dashboard all pointed at a real GPU — to close the
integration gap the standalone validation script doesn't cover.

## Contributing / picking up a phase

See [ROADMAP.md](ROADMAP.md) for the current phase breakdown. Phase 2
(topology-aware scheduling) and Phase 3 (the distributed training demo) are
the highest-leverage places to contribute right now — both are unstarted
and both are described concretely enough to pick up directly.
