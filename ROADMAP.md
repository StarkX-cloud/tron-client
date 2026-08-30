# Roadmap

Status as of this rebuild. See [ARCHITECTURE.md](ARCHITECTURE.md) for the
technical design behind each phase.

- [x] **Phase 1 — Execution spine.** Content-addressed artifacts +
      append-only replayable event log (`tron/spine/`). Wired into
      `queue_server.py`. 19 passing tests in `tests/test_spine.py`.
- [x] **Phase 2a — Latency-aware placement (partial).**
  - [x] `tron/spine/topology.py`: `TopologyMap` records real, worker-
        self-reported heartbeat round-trip latency (EWMA-smoothed, ages
        out if the worker stops reporting).
  - [x] `tron_runtime/global_brain.py` rewritten from a stub into real
        scoring driven by measured latency.
  - [x] Deleted the 11 other `tron_runtime`/root stub modules that were
        instantiated but never called by anything — decoration, not
        behavior.
  - **Real architectural limit found while wiring this up, not yet
    solved:** `/next_job` is worker-*pull* — a worker asks "give me my
    best job" and the master answers from that worker's own queue view in
    isolation. There is no point where two different workers' scores for
    the *same* job are ever compared, so a naive per-worker latency
    penalty (constant across every job in one call) provably cannot
    change which job gets picked — it shifts every candidate's score
    equally, which leaves argmax untouched. Current fix: the penalty
    scales with the job's `compute_weight`, so a slow-link worker is
    steered toward light jobs and away from heavy ones — a real,
    testable effect (see `tests/test_queue_server_topology.py`), but it
    is *not* the same thing as "the closest worker gets the job."
- [x] **Phase 2b — Cross-worker arbitration.** `tron/spine/matcher.py`:
      a periodic (`MATCH_INTERVAL_SECONDS`, default 1s) master-side match
      step using `scipy.optimize.linear_sum_assignment` — the real
      Hungarian algorithm, not a greedy heuristic (a greedy "take the
      best pair first" approach is provably suboptimal here; see the
      worked example in matcher.py's docstring and
      `test_optimal_assignment_beats_the_greedy_trap`). Results land in
      `pending_assignment`; `/next_job` serves from there first, falling
      back to its own per-worker argmax only for jobs the match cycle
      hasn't reached yet. 10 tests in `tests/test_matcher.py`, 2
      end-to-end in `tests/test_queue_server_topology.py` proving the
      exact cross-worker case Phase 2a's per-worker scoring alone
      couldn't solve. New runtime dependency: `scipy`.
- [ ] Either implement or delete `tron_runtime/load_shaper.py`'s
      `reshape()`, which today is a pass-through (`delay: 0` for every
      job) — same "no module ships as an empty interface" rule.
- [x] **Phase 3 — Distributed training demo (small-scale, real).**
      `tron/training/`: a hand-written numpy MLP (backprop verified
      against numerical gradients, `tests/test_training_model.py`),
      synthetic non-IID sharded data (`data.py`), DiLoCo-style local SGD
      vs. a sync-every-step baseline (`local_sgd.py`), and weight-space
      merging via task arithmetic + TIES (`merge.py`). Run
      `python -m tron.training.benchmark` for the report. Locked,
      reproducible numbers on the benchmark's fixed problem (class_sep=1.0,
      skew=0.95 — deliberately hard, not cherry-picked):
      - Local SGD: **10x less communication** than sync-every-step
        (135,680 vs. 1,356,800 bytes) for **1.6 points less accuracy**
        (85.6% vs. 87.2%).
      - Merging: **zero communication during training** recovers 81.0%
        accuracy (TIES: 81.6%) vs. 58.8% average for an unmerged solo
        shard — each shard only ever saw ~95% one class.
      - All numbers pinned as regression tests in `tests/test_local_sgd.py`
        and `tests/test_training_merge.py`.
  - **What this does not claim:** this is a few-hundred-parameter model
    on synthetic data run in seconds on one CPU core, not a
    frontier-scale training run across real heterogeneous machines. The
    goal was to implement and verify the *algorithms* correctly at a
    size anyone can rerun and check in seconds — scaling to a real model
    size and an actual multi-machine network is substantial future work,
    not a claim this phase makes.
  - [ ] Not yet done: running this across genuinely separate physical
        machines over a real network (today all "shards" are in-process
        Python objects on one machine — communication bytes are counted,
        not actually transmitted over a wire yet).
  - [ ] Not yet done: a public writeup / standalone repo extraction — see
        the "getting noticed" discussion this phase was scoped around.
- [x] **Phase 4 v1 — The 3D Grid: passive replay.** `tron/grid/index.html`,
      served at `/grid/` by `queue_server.py`. Worker distance from the
      master = real measured heartbeat latency (the same number
      `GlobalDecisionBrain` uses); worker height = reported load; task
      position = whichever worker the event log actually assigned it to.
      Time-scrubber replays `/spine/events` to any point — verified in a
      real browser against a live server: scrubbing to right after 6 jobs
      were submitted (before any assignment) showed exactly 6 grey
      "queued" markers at the master, matching the raw log. "Live" mode
      tails new events. three.js is vendored locally
      (`tron/grid/three.min.js`, `OrbitControls.js`) rather than loaded
      from a CDN, so the page works offline.
  - [ ] Not yet done: pipe width from bandwidth (no bandwidth prober
        exists yet — only latency is measured; see the note under Phase 2b).
  - [ ] Not yet done: smooth motion between states (currently discrete
        per scrub step).
  - [ ] Not yet done: interaction (click a task to inspect its recorded
        inputs; drag a worker and watch the scheduler react). Deliberately
        deferred until the passive replay is solid — see ARCHITECTURE.md's
        "why 3D observation, not 3D authoring."
- [x] **Phase 3 -> Phase 1 wiring.** `tron/training/spine_integration.py`:
      `run_local_sgd_with_spine` records each shard's per-round training
      as a real Task in the spine log (queued -> assigned -> started ->
      completed, output = the shard's actual weight vector as a
      content-addressed Artifact), reusing local_sgd.py's exact private
      helpers rather than reimplementing the algorithm — a parity test
      proves the instrumented and uninstrumented runs produce
      bit-for-bit identical final models. `POST /training/run_demo`
      triggers this against `queue_server.py`'s own spine, registering
      synthetic `shard-N` workers. Verified live: triggered a 4-shard,
      6-round run against a running server (85.75% accuracy — consistent
      with the standalone benchmark), then confirmed in the Grid that
      all 96 events (4 shards x 6 rounds x 4 lifecycle events) rendered
      correctly with all 24 tasks completed and correctly attributed to
      their shard.
  - Caught and fixed during that verification: task identity was
    `(fn_hash, input_hashes, attempt)` with the same inputs for every
    shard in a round, so all shards in a round collapsed onto the same
    task id (a test — not a visual check — caught this: task count came
    back as 4 instead of 8 for a 4-round, 2-shard run). Fixed by hashing
    each shard's actual data into its task's input_hashes, which is also
    the more correct semantics (different data is a different input).
  - Also caught and fixed: `hashAngle()` (the Grid's stable-layout hash)
    used a plain polynomial hash, which barely disperses sequential
    names like `shard-0`..`shard-3` (89°/90°/91°/92° — stacked almost on
    top of each other). Replaced with FNV-1a plus a murmur3-style
    finalizer for real avalanche behavior; confirmed the 4 shard nodes
    now spread clearly around the master in a live screenshot.
  - Also caught and fixed: `queue_server.py`'s `event_log`/`artifact_store`
    defaulted to a fixed on-disk path, so test suites that reload the
    module (to reset in-memory state) were silently accumulating state
    across test runs in the same file — invisible until a test asserted
    on the *total* log contents instead of filtering by a known id. Fixed
    with a `TRON_SPINE_DIR` env var tests now set to an isolated temp dir.

## Known follow-ups (not blocking, tracked here so they aren't lost)

- `tron/gpu/` OpenAI-compatible gateway (`openai_bridge.py`, `scheduler.py`)
  is functional but its standalone demo docs/scripts were removed along
  with the duplicate `vgpu/` package — reviving the gateway demo needs new
  docs pointing at `tron.gpu.*` instead of the old `vgpu.*` paths.
- `TRON-II`'s outcome-tracking idea (score adapters by capability-gained
  per compute spent, adjust future decisions from actual outcomes) is
  worth pointing at Phase 3's training runs once those exist — right now
  `tron/orchestrator/outcomes.py` has no real workload feeding it.
