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
- [ ] **Phase 3 — Distributed training demo.**
  - [ ] DiLoCo-style local SGD (local steps between infrequent outer sync).
  - [ ] Weight-space merging (TIES / task arithmetic) as the
        zero-communication path.
  - [ ] Run across genuinely heterogeneous nodes (not all-identical
        hardware) and publish one number: communication bytes + wall-clock
        vs. a naive all-reduce baseline on the same hardware.
  - [ ] Reproducible standalone repo + writeup — this is the artifact meant
        to get attention, not the platform as a whole.
- [ ] **Phase 4 — The 3D Grid.**
  - [ ] Passive replay view over `EventLog.replay()` — node distance from
        measured topology, pipe width from throughput, node size from load.
  - [ ] Time-scrubbing through a real (not scripted) training run.
  - [ ] Interaction (inspect, re-place, compose) only after the above works.

## Known follow-ups (not blocking, tracked here so they aren't lost)

- `tron/gpu/` OpenAI-compatible gateway (`openai_bridge.py`, `scheduler.py`)
  is functional but its standalone demo docs/scripts were removed along
  with the duplicate `vgpu/` package — reviving the gateway demo needs new
  docs pointing at `tron.gpu.*` instead of the old `vgpu.*` paths.
- `TRON-II`'s outcome-tracking idea (score adapters by capability-gained
  per compute spent, adjust future decisions from actual outcomes) is
  worth pointing at Phase 3's training runs once those exist — right now
  `tron/orchestrator/outcomes.py` has no real workload feeding it.
