# Roadmap

Status as of this rebuild. See [ARCHITECTURE.md](ARCHITECTURE.md) for the
technical design behind each phase.

- [x] **Phase 1 — Execution spine.** Content-addressed artifacts +
      append-only replayable event log (`tron/spine/`). Wired into
      `queue_server.py`. 19 passing tests in `tests/test_spine.py`.
- [ ] **Phase 2 — Heterogeneous fault-tolerant fabric.**
  - [ ] Topology prober: measure real pairwise latency/bandwidth between
        registered workers.
  - [ ] Replace `tron_runtime/global_brain.py`'s stub scoring with
        placement decisions driven by measured topology.
  - [ ] Either implement or delete each remaining stub in `tron_runtime/`
        (`auto_scaler.py`, `market_engine.py`, `predictor_engine.py`,
        etc.) — no module ships as an empty interface long-term.
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
