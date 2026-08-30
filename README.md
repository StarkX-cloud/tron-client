# TRON

TRON is a distributed execution runtime for Python, being rebuilt around a
single goal: **run real compute and training workloads across
heterogeneous, unreliable machines with near-zero communication overhead —
with a live, causal 3D replay of what the cluster is doing.**

See [ARCHITECTURE.md](ARCHITECTURE.md) for the technical design and
[ROADMAP.md](ROADMAP.md) for what's built vs. planned. Short version:
Phases 1–3 (execution spine, topology-aware scheduling, a small-scale
distributed training demo) are built and tested; Phase 4 (the 3D Grid) is
not yet built.

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
  Run `python -m tron.training.benchmark` for the report. See
  ARCHITECTURE.md for the numbers and what's honestly still missing
  (real multi-machine execution, a larger model).
- **TRON-II** (`tron/orchestrator/`): training orchestration with
  pluggable adapters (Ray, SB3, scikit-learn, Transformers) and an
  outcome-tracking loop that scores adapters by how close their predicted
  cost/capability gain was to reality. Not yet wired to a real training
  workload — see ROADMAP.md.
- **vGPU** (`tron/gpu/`): a simulation layer that aggregates multiple
  GPUs' reported specs into one synthetic profile, plus an
  OpenAI-compatible API gateway on top. Explicitly a simulation, not GPU
  virtualization.

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

## Contributing / picking up a phase

See [ROADMAP.md](ROADMAP.md) for the current phase breakdown. Phase 2
(topology-aware scheduling) and Phase 3 (the distributed training demo) are
the highest-leverage places to contribute right now — both are unstarted
and both are described concretely enough to pick up directly.
