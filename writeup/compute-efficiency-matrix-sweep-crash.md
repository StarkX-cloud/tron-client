# A real segfault, found by the matrix sweep, and how it was contained

Building the dtype x budget matrix sweep (to properly characterize the
int8+offload CPU quirk noted in ROADMAP.md, instead of probing one budget
at a time by hand) surfaced something more serious than expected: a real,
reproducible crash in `OffloadPolicy` itself — not a Python exception, a
segmentation fault.

## The crash

```bash
python -m tron_efficiency.cli matrix-sweep --model EleutherAI/pythia-70m \
    --max-new-tokens 8 --dtypes checkpoint-default --memory-budget-mb-list 130,120
```

Exited with `Segmentation fault`, process exit code 139. Isolated to the
smallest possible repro:

```python
config = WorkloadConfig(model_name="EleutherAI/pythia-70m", max_new_tokens=8, device="cpu", dtype=None)
outcome = run_workload(config, OffloadPolicy(memory_budget_bytes=130 * 1024 * 1024))
```

Reproduced **twice**, independently, with exit code 139 both times:
once via the CLI's `matrix-sweep`, once calling `run_workload` directly in
a plain `python -c` script. Not a fluke, not specific to the CLI or the
sweep orchestration — the crash lives in the real mechanism
(`OffloadPolicy.load_model` → `accelerate`'s `device_map="auto"` + disk
offload → `generate()`), most likely somewhere in `accelerate`'s
disk-offload / `safetensors` memory-mapped file handling or a native
extension it calls into, on this specific Windows/CPU combination at this
specific budget. Not root-caused further than that — proper native
debugging (minidumps, a debug build of the relevant extensions) is beyond
this project's scope right now.

## Why this matters more than a single failed run

Every other failure path in `tron_efficiency` is a Python exception,
caught by `run_workload`'s `except Exception` and reported as a graceful
`RunOutcome(success=False, error=...)` — `OffloadPolicy` running out of
memory, `UnsupportedDtypeCombination` for int4+offload, accelerate's
"whole model to disk" refusal, all of it. A segfault is categorically
different: it happens below the Python interpreter, in native code, and
**no amount of `try/except` in Python can catch it**. Before this was
found, a crash like this during a `sweep`/`dtype-sweep`/`matrix-sweep` run
would have killed the whole process — not just failed one cell, but lost
every result already collected in that run, with no graceful report, just
`Segmentation fault` on the terminal and nothing saved.

## The fix: containment, not a cure

`tron_efficiency/workloads/isolated_runner.py` runs the actual workload in
a subprocess (`multiprocessing`, `spawn` context explicitly — the only
real option on Windows anyway) instead of the calling process. If the
child crashes, the parent detects the nonzero exit code and reports a
normal failed `RunOutcome` citing the crash — the sweep continues to the
next cell instead of dying. If the child hangs, a timeout (default 600s)
terminates it instead of blocking forever.

This is containment, not a fix for the underlying crash — the segfault
itself is still unexplained and unfixed. `--isolated` is on by default for
`sweep`/`dtype-sweep`/`matrix-sweep` (opt out with `--no-isolated` to trade
the safety for speed — isolation means paying a full Python/torch import
cost per run, not shared across a sweep the way an in-process loop would).

## An honest complication: re-running the exact crash under isolation did NOT crash

Re-running the identical 130MB/120MB scenario through
`run_workload_isolated` succeeded both times — no crash. This does **not**
mean the bug is fixed or understood. Two explanations are equally
plausible and this project cannot currently distinguish between them:

1. The crash is non-deterministic (a real possibility for native memory
   corruption bugs) and simply didn't trigger on that run.
2. `spawn`'s clean per-process import state incidentally avoids whatever
   combination of library initialization order causes the crash in a
   process that's already done other work (import caching, thread pool
   state, etc. — all plausible triggers for this class of bug).

**What is proven, deterministically, independent of whether the original
crash reproduces on any given run:** the containment mechanism itself
works. `tests/test_efficiency_isolated_runner.py` verifies this against
small, fast, deliberately-crashing/hanging/succeeding target functions
(`os._exit(1)` to simulate an abnormal exit, `time.sleep()` past the
timeout, a normal return) — not by hoping the flaky real trigger fires on
schedule. A crash in the subprocess is correctly detected and reported;
the caller survives either way.

## What's still open

- The segfault itself: not root-caused, not fixed, possibly
  non-deterministic. Isolation contains the blast radius; it does not
  explain or prevent the crash.
- Whether this is Windows-specific, CPU-specific, specific to this exact
  budget/model pairing, or a broader `accelerate` disk-offload issue is
  unknown — not tested on Linux or on a GPU.
- `run_workload`/`run_suite`/`compare`/`run` (the non-sweep CLI commands)
  still run in-process by default, uninsulated from this same class of
  crash — isolation is currently only wired into the three sweep commands,
  where the cost of losing an entire multi-run sweep to one crash was the
  motivating case. A single `run`/`compare` invocation crashing just
  crashes that one invocation, same as before this work.
