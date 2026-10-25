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
python -m tron_efficiency.cli run --policy baseline     # plain load + generate
python -m tron_efficiency.cli run --policy offload --memory-budget-mb 100
python -m tron_efficiency.cli compare --memory-budget-mb 100 --hourly-rate 0.50
python -m tron_efficiency.cli serve                      # API + dashboard on :9100 (TRON_EFFICIENCY_PORT)
```

The dashboard is served at `http://localhost:9100/dashboard/`, mounted the
same way `/grid` is mounted on `queue_server.py` — a separate static
directory, no shared frontend framework.

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
(first attempt did, at a 600MB cap). **Fixed**: `OffloadPolicy` now reserves
`gpu_activation_headroom_fraction` (default 0.6, configurable) of the
nominal `memory_budget_bytes` before handing the rest to accelerate as the
GPU weight budget — close to, and more conservative than, the ~58% headroom
that was empirically proven to work on the T4 run. The fix itself is
covered by unit tests (`resolve_max_memory` now reserves headroom on CUDA,
unchanged on CPU); it has **not** been re-run against the real GPU, since
that requires the same manual Colab session as the original validation —
that re-run is still open work, not yet done.

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
- The GPU validation above was a standalone script reproducing
  `OffloadPolicy`'s mechanism, not the installed `tron_efficiency` package
  (CLI/API/dashboard) actually running end-to-end on a CUDA machine — that
  integration is still unexercised on real GPU hardware.
- The activation-headroom fix above is unit-tested but not yet re-verified
  on real GPU hardware (would need another manual Colab session) — the
  default fraction is a conservative extrapolation from one working data
  point, not itself independently confirmed.
- Single model family tested per environment (`pythia-70m` on CPU,
  `pythia-410m` on the one GPU run); no sweep across model sizes or memory
  budgets yet (`benchmarks/engine.py` supports it — `cli.py` just doesn't
  expose a sweep flag yet).
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
