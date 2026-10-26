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
- [x] **Phase 2c — Measured bandwidth, not just latency.** Phase 2a/2b
      scored placement on one network signal (heartbeat RTT). A worker
      now also measures real **throughput** to the master: `worker.py`'s
      `bandwidth_probe()` downloads 1MB from `GET /probe/blob` (random,
      so transport gzip can't inflate the rate) and uploads 256KB to
      `POST /probe/sink`, dividing bytes by wall-clock seconds — a
      genuine transfer measurement, replacing the hard-coded
      `network_bandwidth_gbps: 1.0` the old registration reported.
      Probed every 15th heartbeat (throughput drifts slower than
      latency), reported alongside `latency_ms`.
      `TopologyMap.record_bandwidth` / `.bandwidth()` store it with the
      identical EWMA + age-out machinery as latency.
      `matcher.score_pair` and `global_brain.decide` gain a transfer
      term: `transfer_seconds * BANDWIDTH_PENALTY_WEIGHT`, where
      `transfer_seconds = job.transfer_bytes * 8 / (mbps * 1e6)` — one
      shared helper (`matcher._transfer_penalty`) so the periodic match
      step and the per-worker `/next_job` fallback can never disagree
      about what a placement costs. **The term is provably inert until
      bandwidth is measured *and* the job declares `transfer_bytes`** —
      every pre-2c placement decision (and its pinned test) is
      byte-identical. New tests: 9 in `tests/test_topology.py`, 3 in
      `tests/test_matcher.py` (heavy transfer steered to the fat pipe
      where latency-only scoring couldn't tell the workers apart), 5
      end-to-end in `tests/test_queue_server_topology.py`. 120 passing.
- [x] **`tron_runtime/load_shaper.py` is real now** (was a `delay: 0`
      pass-through — the "implement it or delete it" rule). Backed by
      Phase 2c's measured bandwidth: a worker whose downlink is `B` Mbps
      absorbs ~`B * 1e6 / 8 * window_seconds` bytes of job input before
      further transfers queue on the wire. `LoadShaper` tracks bytes
      dispatched to each worker over a sliding window (a *link cooldown*
      that ages out on its own, not tied to job completion — a slow-pipe
      worker that just took a big transfer is still recovering even after
      it reports the job done) and won't pile a new transfer onto a
      worker already at budget. `_run_match_cycle` consults
      `can_accept()` before committing a pairing; `/next_job` runs the
      queue through `reshape()`. Inert by construction: unmeasured
      bandwidth = unlimited, no `transfer_bytes` = never held, an idle
      link accepts any single job (even one bigger than a window, else it
      could never schedule). `shape()` — genuinely unused — was deleted.
      12 unit tests in `tests/test_load_shaper.py`, 2 end-to-end in
      `tests/test_queue_server_topology.py`.
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
  - [x] **Over a real wire.** `tron/training/distributed/`: shards are
        now separate OS processes that serialize their parameter vectors
        and `POST` them to the master (`queue_server.py`'s
        `/training/session*` endpoints), which stores each as a spine
        Artifact, barrier-averages once every shard has reported for a
        round (`protocol.average_vectors`), and serves the merge back.
        `python -m tron.training.distributed.shard_worker --master <url>
        --session <id> --shard <k>` is the per-process entry point;
        pointing `--master` at another host is the *only* change needed
        to run across genuinely separate machines (commit 3d6939b already
        proved a real worker against a deployed Render instance works).
        `examples/distributed_training/` runs it across local subprocesses
        and against separate hosts. **The transport does not change the
        math:** `tests/test_distributed_training.py`'s parity test drives
        the full HTTP path through `fastapi.testclient` and asserts the
        final merged model is bit-for-bit equal to
        `local_sgd.train_local_sgd` in one process — same seeds, same
        schedule, same averaging op. 9 tests (protocol round-trips, the
        barrier's 202-until-ready behaviour, wire-byte accounting split
        into upload+download legs, the bit-for-bit parity run, spine
        recording). The run reports both `wire_bytes_transferred` (every
        leg that actually crossed a socket) and `algorithmic_comm_bytes`
        (the conceptual one-sync-per-shard figure the in-process metric
        counts) so the comparison stays honest.
    - [x] **LoRA over the same wire.** `tron/training/distributed/
          lora_wire.py` + `lora_param_server.LoraTrainingSession`: the
          unit crossing the socket each round is a LoRA **adapter state
          dict** (~400KB for Pythia-70M), not a numpy vector and never
          the ~282MB base model — the base is assumed present on every
          node (`shard_worker` loads it locally; only the adapter is
          transmitted). `POST /training/session` with
          `model_kind: "lora"` dispatches to this path;
          `run_local.py --lora` runs it across real subprocesses.
          Same parity bar as the numpy path:
          `tests/test_lora_over_wire.py` drives the full HTTP path with
          one thread per shard and asserts the final merged adapter is
          **tensor-for-tensor identical** to
          `lora_demo.run_local_sgd_lora` in one process (long-lived
          optimizer across rounds included). 5 tests. Verified end to end
          against **real Pythia-70M** via `run_local.py --lora` (2 shards,
          3 rounds x 3 steps): eval loss 4.3296 -> 4.2641; adapter
          393,216 B vs. the full model's 282,099,712 B (**717x**);
          **4,718,592 bytes actually sent over sockets** — the 282MB base
          never moved. (The example's launcher sets
          `KMP_DUPLICATE_LIB_OK` + single-digit BLAS threads so several
          torch processes loading a model on one Windows box don't crash
          on duplicate OpenMP runtimes — a local-orchestration detail, not
          something a real multi-host run needs.)
  - [x] **Public writeup + standalone extraction.**
        `writeup/distributed-training.md` is a self-contained account —
        the idea, the numpy result, the LoRA/Pythia result, the move to a
        real socket, reproduction steps, and an explicit "what this does
        not claim". `scripts/extract_training_repo.py` copies the spine +
        training + distributed code + the load-bearing tests into a
        minimal standalone tree (with its own README / requirements /
        minimal `tron/__init__.py`); verified — the extracted repo's
        suite runs 61 passed, 2 skipped (the outcome tests, which need
        `tron.orchestrator`, skip cleanly). Nothing is published by the
        script; it prepares the tree and prints the `git init` steps.
- [x] **Phase 3 scale-up — same story, a real pretrained model.**
      `tron/training/lora_demo.py` + `benchmark_lora.py`: the identical
      local-SGD / weight-merging comparison, applied to EleutherAI's
      **Pythia-70M** (a real, recognized small open-weight model — not
      a hand-rolled net) via **LoRA**, fine-tuned on the tiny-shakespeare
      corpus (`tron/training/data/tinyshakespeare.txt`, public domain,
      vendored locally). Run `python -m tron.training.benchmark_lora`.
      Recorded numbers from an actual run (3 shards, 4 rounds x 5 local
      steps):
      - LoRA adapter: **393,216 bytes** vs. the full model's
        **282,099,712 bytes** — a **717x** smaller unit of communication,
        before local-SGD's sync-frequency reduction is even counted.
      - Local SGD: eval loss **4.3296 -> 4.2236** (genuinely decreasing —
        the pipeline trains correctly), **4,718,592 bytes** over 4 syncs
        vs. a hypothetical full-model sync at the same cadence
        (**3,385,196,544 bytes** — the same 717x, compounding with
        whatever sync-frequency reduction is chosen).
      - Merge: solo shard losses `[4.283, 4.3703, 4.3146]` (avg 4.3226,
        zero communication during training) -> merged **4.2403** —
        recovers most of local SGD's benefit for no communication at all.
      - This is the concrete demonstration of the "low-rank delta, not a
        full checkpoint, is the unit of communication" idea from the
        project's original brief — now real, not conceptual.
      - **Honest cost:** a full run takes **~20 minutes** on this
        project's CPU-only dev machine (8 cores, no GPU) — too slow to
        be a live pytest. `tests/test_lora_demo.py` covers the pure
        logic fast (shard splitting, byte accounting, adapter averaging
        — one test hand-verifies the adapter byte count against the
        LoRA math directly) without downloading or training the real
        model; the full pipeline was verified manually, once, with the
        numbers above recorded here rather than re-asserted by CI.
      - Real bug hit and fixed during this, not a defensive guess: the
        `hf_xet` fast-transfer backend hung indefinitely downloading
        model weights on this machine. `HF_HUB_DISABLE_XET=1` (set
        automatically at the top of `lora_demo.py`) works around it.
      - New optional dependency group: `requirements-training.txt`
        (torch, transformers, peft, accelerate, safetensors) — not
        required for `queue_server.py` or the core test suite.
  - [x] **LoRA adapters wired into the spine.** `tron/training/lora_spine.py`:
        `run_local_sgd_lora_with_spine` is the adapter counterpart of
        `spine_integration.run_local_sgd_with_spine`. It reuses
        `lora_demo.py`'s own helpers (`make_lora_model`, `train_steps`,
        `average_adapter_states`, ...) in the identical order, adding
        recording as pure side effect — each shard's per-round adapter
        training is one Task (queued -> assigned -> started -> completed),
        with that shard's LoRA state dict stored as the completed event's
        output Artifact (`transfer_bytes` = adapter size, task metadata
        carries the ~280MB full-model size alongside for contrast). So the
        Grid renders a LoRA run with the same event vocabulary as
        everything else. `tests/test_lora_spine.py` (6 tests, tiny
        stand-in model — no Pythia download) asserts the instrumented run
        matches `lora_demo.run_local_sgd_lora` **tensor-for-tensor**.
        `python -m tron.training.benchmark_lora --spine-dir DIR` runs the
        real Pythia-70M local-SGD portion through this path so an actual
        LoRA run is Grid-replayable (`TRON_SPINE_DIR=DIR python
        queue_server.py`).
  - [x] **LoRA over the real wire — done.** `tron/training/distributed/
        lora_wire.py` + `lora_param_server.LoraTrainingSession` send LoRA
        adapter state dicts (~400KB) as the round's POST body over the
        same transport the numpy MLP uses; the ~282MB base model never
        moves (every node loads its own). `model_kind: "lora"` on
        `POST /training/session`; `run_local.py --lora`.
        `tests/test_lora_over_wire.py` pins the final merged adapter
        tensor-for-tensor against the single-process
        `lora_demo.run_local_sgd_lora`. This is the "real model over a
        real network" combination, now real.
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
- [x] **Phase 4 v2 — Grid interaction, pipe width, smooth motion.**
      Passive replay came first on purpose; it's solid, so v2 adds:
  - **Pipe width = measured bandwidth.** The master->worker link is a
    tube whose radius comes from `bandwidth_mbps_down` (Phase 2c);
    unmeasured links are drawn hair-thin, not assumed fat — same rule as
    latency distance. Verified in a real browser: fast-w (300 Mbps) pipe
    radius 0.12, mid-w (40) 0.042, slow-w (6) 0.032, unmeasured shards
    0.03 — matches `bandwidthRadius()` exactly.
  - **Click a task -> inspector.** Raycast onto the task marker, fold
    that task's events out of the already-loaded log (no new fetch), show
    id / status / node / attempt / fn+output hashes / lineage events, and
    for training tasks the round·shard, the bytes transferred, and the
    adapter-vs-full-model size ratio.
  - **Shift-drag a worker -> "what would the scheduler do?".** Dragging a
    worker radially maps its new distance back to a hypothetical latency
    (inverse of the placement formula) and `POST`s it to
    `/scheduler/whatif`, which re-runs `tron/spine/matcher.py` against a
    throwaway topology + copies of the live queue and idle workers —
    **touching no real state** — and returns the assignment it *would*
    produce next to the baseline. Reassigned jobs are drawn as bright
    links; the worker snaps back (it was a question, not a move). 5 tests
    in `tests/test_grid_whatif.py` (non-destructive, reflects the
    matcher's real latency and bandwidth scoring, empty-queue case).
  - **Smooth motion.** Task markers ease toward their target position
    each frame (`position.lerp`) instead of snapping on every scrub step.
  - The three.js interactions were verified in a real browser against a
    live server (worker picking, the click->inspector round trip, the
    drag->what-if round trip, zero console errors) — the same standard
    Grid v1 was held to.
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

- [x] `tron/gpu/` OpenAI-compatible gateway now has `tron/gpu/README.md`
  documenting the `tron.gpu.*` layout (cluster / runtime / scheduler /
  openai_bridge), how to run the bridge (with the mocked-proxy test
  path), **and** the honest gap: `TRONOpenAIProxy` still posts to
  `/submit_job` and polls `/jobs` — the older standalone scheduler's API,
  not the spine server's `/submit` + `/status/{id}`. Folding an OpenAI
  request into a real spine Task is the remaining wiring.
- [x] `TRON-II`'s outcome-tracking now has a real workload:
  `tron/orchestrator/outcomes.py`'s `OutcomeLog` is fed by every finished
  training run. `param_server.TrainingSession` (numpy, over the wire) and
  `lora_spine.run_local_sgd_lora_with_spine` (LoRA) each record one
  `TrainingOutcome` — capability gained (held-out accuracy delta vs. the
  untrained init, or eval-loss reduction) over compute spent (total local
  SGD steps). `GET /training/outcomes` on `queue_server.py` reports them
  with a `capability_per_compute` ratio; `benchmark_lora.py --spine-dir`
  persists an `outcomes.json` next to the run's spine. Tests:
  `tests/test_distributed_training.py` (recorded once, not per poll;
  cost == rounds·shards·local_steps), `tests/test_lora_spine.py`.
- [x] The feedback loop half of the TRON-II idea: outcomes are now read
  back, not just recorded. Three pieces:
  - **Estimate refinement** — `TrainingOrchestrator._estimate_capability_gain`
    / `_estimate_adapter_cost` check `OutcomeLog.estimate_capability_gain`
    / `.estimate_cost` first and only fall through to the original
    hardcoded per-adapter priors (`ray` -> 0.9, `sb3` -> 1.0, ...) when
    there's no history yet. The three real-workload recorders
    (`lora_spine.py`, `param_server.py`, `lora_param_server.py`) do the
    same for their own "expected" value at record time, so `accuracy()`
    stops being trivially 1.0 on every run for a repeated `run_name`/
    session id (e.g. `benchmark_lora.py`'s fixed run name across CI runs)
    and becomes a real prediction-vs-actual signal.
  - **Module specificity** — `OutcomeLog.pair_accuracy` /
    `.pair_success_rate` key by `(adapter_name, module_id)`, not just
    `adapter_name`, so an adapter that's reliable for one module and not
    another doesn't get flattened into one blended number.
    `TrainingPolicy.score_estimate` prefers the pair-specific numbers,
    falling back to the adapter-wide ones the pair has no history.
  - **Placement feedback** — `TrainingOutcome` now carries `node_ids` (the
    spine worker names that took part in that run); `OutcomeLog.node_quality`
    turns that into a per-node success rate. `matcher.score_pair` /
    `match_jobs_to_workers` take an optional `outcome_log` and fold in a
    `(node_quality - 0.5)` bonus/penalty on top of the existing
    latency/bandwidth terms — "zero unless known," same contract as the
    bandwidth term, so a node with no recorded outcomes scores exactly as
    before. `queue_server.py` passes `training_outcomes` into both real
    call sites (`/scheduler/whatif` and the periodic match cycle).
    Deliberately NOT added to `tron_runtime/global_brain.py`'s per-worker
    `/next_job` fallback: that call fixes `worker_name` and picks among
    jobs, so a worker-only term would shift every candidate's score by
    the same constant and never change the argmax — dead weight there,
    real weight in the matcher's cross-worker assignment.
  Tests: `tests/test_outcomes_feedback.py` (12 cases — pair stats staying
  scoped per module, the estimate/adapter-wide/hardcoded fallback chain,
  node_quality from participation, matcher steering toward the
  better-track-record node when latency ties, save/load round-trip
  including old outcomes.json files with no `node_ids` key).
- [x] Verified the distributed transport against a real WAN link, not
  just localhost — deployed the master to Render (free tier, Oregon) and
  ran shards against it from a genuinely separate network. This surfaced
  a real gap `shard_client.RequestsTransport` had zero retry logic, so
  the first run's intermittent connection-layer failures (TLS record
  corruption, connection resets — confirmed with both `requests` and
  `curl`, so a property of the link, not a client-library quirk)
  permanently killed a shard. Fixed with exponential-backoff + jitter
  retries on transient failures only (never on a 4xx). Re-run after the
  fix survived the same flaky link — one retry fired mid-run, training
  finished clean, `GET /training/outcomes` on the live master confirmed
  `node_ids` recorded correctly. See writeup/distributed-training.md's
  "Verified against a live public master" section for the real numbers
  (measured latency, failure rate, final accuracy). Explicitly not yet
  done at that point: auth between nodes, encryption beyond what HTTPS
  provides, and the master is a single process with no replication —
  real gaps before this could run multi-tenant.
- [x] Closed the auth gap above. Every `/training/session*` route and
  `/training/outcomes` had zero access control — anyone with the
  master's URL could create sessions (a LoRA one loads a real base
  model, a resource-exhaustion vector) or submit fabricated "trained"
  data as any shard. `TRON_TRAINING_AUTH_TOKEN`, set on the master, now
  makes every one of those routes require a matching `X-TRON-AUTH`
  header (`hmac.compare_digest`, fail-closed once set; open only when
  unset, the local-dev/test default — every existing test and the
  loopback example are unaffected). `RequestsTransport` and
  `shard_worker.py --auth-token` already had the client-side plumbing;
  `run_local.py` gained a matching `--auth-token` flag. Also fixed a
  related, same-class bug found while in this code: `/heartbeat` only
  checked the header *if the caller happened to send one*, so omitting
  it entirely was a silent bypass — now fail-closed there too (all real
  callers, i.e. `worker.py`, already always sent it). `render.yaml`
  declares `TRON_TRAINING_AUTH_TOKEN` as a dashboard-set secret, not
  committed. Tests: `tests/test_training_auth.py` (6 cases — open by
  default, 401 with no/wrong token once configured, every route
  individually guarded, and a full two-shard run driven end-to-end
  through an authenticated transport).
  Still not done at that point: encryption of the payload itself (HTTPS
  covers transport only), defense against an authenticated-but-malicious
  participant, and the master remains a single process with no
  replication.
- [x] Closed the at-rest encryption gap above.
  `TRON_ARTIFACT_ENCRYPTION_KEY` (a Fernet key) now encrypts every
  artifact — training vectors, adapter states, shard data — before it
  touches disk (`tron/spine/store.py`). Content-addressing-transparent by
  construction: `artifact_id` is always `content_hash` of the plaintext,
  computed before encryption and checked after decryption, so nothing
  that references an artifact by hash changes behavior whether encryption
  is on or off. Enabling it on a store that already has plaintext
  artifacts doesn't retroactively encrypt them — reading old data still
  works (the store can tell the two cases apart honestly: if the raw
  on-disk bytes already hash to the requested id, they're pre-encryption
  plaintext, not corrupted). Also fixed the matching gap in the auth
  token itself: `_require_training_auth` now refuses a request carrying
  `X-Forwarded-Proto: http` (the standard signal a TLS-terminating proxy
  sets) even with a correct `X-TRON-AUTH` token — a secret that can still
  cross the wire in cleartext isn't meaningfully protected. `render.yaml`
  declares `TRON_ARTIFACT_ENCRYPTION_KEY` as a dashboard-set secret, not
  committed, same as the auth token. Tests: 6 new cases in
  `tests/test_spine.py` (roundtrip, artifact_id unaffected by encryption,
  ciphertext really isn't plaintext on disk, wrong key can't decrypt,
  enabling encryption on an existing plaintext store still reads old
  data, unset key behaves exactly as before) + 3 new cases in
  `tests/test_training_auth.py` (plain-HTTP rejected even with the right
  token, no X-Forwarded-Proto header behaves as before, https is fine).
  Still not done: this does not hide data *from the master itself* — it
  genuinely has to read each shard's plaintext vector to average it, so
  true end-to-end confidentiality would need secure aggregation (a much
  larger, different feature) — and it does not defend against a caller
  who holds a valid token but sends bad data on purpose. The master also
  still remains a single process with no replication.
- [x] Built a concrete telecom demo on top of all of the above, instead
  of continuing general-purpose hardening — the honest gap wasn't more
  mechanism, it was that nothing here was pointed at a problem a telecom
  actually recognizes. `tron/training/telecom_data.py`: N cell towers,
  each a hand-built (synthetic — no real carrier data) traffic archetype
  (business-district, residential, entertainment-district, transit-hub),
  predicting next-interval congestion tier from recent signals — genuinely
  non-IID by construction, the same honest hard case this project has
  used throughout. Required zero new training mechanism: one
  `dataset_factory` hook added to `TrainingSession` (default path
  completely untouched, existing pinned tests unaffected) lets
  `queue_server.py`'s `/training/session` accept `problem:
  "telecom_congestion"` and everything downstream — the real wire
  transport, spine recording, Grid replay, outcome scoring, auth,
  at-rest encryption — is exactly what already existed.
  `examples/telecom_demo/run_cell_tower_demo.py` runs it end to end and
  reports held-out accuracy vs. a naive baseline plus the honest
  bandwidth comparison (see writeup/telecom-demo.md — reported per-round,
  not as a misleading cumulative-vs-one-time-dump number, after a real
  run showed those two can be comparable at this toy model's tiny
  parameter count). Tests: `tests/test_telecom_demo.py` (7 cases —
  dataset shape/determinism/genuine non-IID-ness, the wire path actually
  using the tower dataset not the default one, and the trained model
  meaningfully beating the naive majority-tier baseline, not just "the
  run completed"). Full suite: 197 passing.
  Still not done: real telecom data of any kind (explicitly synthetic,
  said twice in the writeup on purpose), validated at real tower counts
  (4, not hundreds), and everything already listed above as not done for
  the underlying distributed-training machinery.
- [x] **New, separate initiative: TRON Compute Efficiency Engine v0.1**
  (`tron_efficiency/`) — not part of the distributed-training substrate
  above, zero coupling to `queue_server.py`. Different question: can a
  memory-constrained inference execution path run under a tighter memory
  budget than a plain baseline, and what's the real performance cost?
  Two real paths (`BaselinePolicy` vs. `OffloadPolicy`, the latter using
  Hugging Face Accelerate's actual `device_map`/`max_memory`/
  `offload_folder` mechanism, not a relabeled baseline), a benchmark
  engine that flags output divergence across policies as a correctness
  issue rather than reporting a timing "win," JSON/CSV/markdown report
  output with per-metric winners and explicitly-labeled cost estimates, a
  CLI, a standalone FastAPI app (`TRON_EFFICIENCY_PORT`, default 9100),
  and a dashboard mounted the same way `/grid` is. See the README's
  "Compute Efficiency Engine (v0.1)" section for usage.
  **Real measured result** (`pythia-70m`, CPU, 32 tokens, 100MB CPU
  budget vs. ~134MB model footprint —
  `writeup/compute-efficiency-sample-report.md`): the constrained path is
  **~5.8x slower end-to-end** (10.2s vs 1.8s; 3.1 vs 18.0 tok/s) with
  only marginally lower peak memory (464MB vs 487MB) — a negative/mixed
  result, reported honestly as one, not a TRON win.
  **GPU validation follow-up (same day):** this dev machine has no CUDA,
  so the GPU-budget code path (`OffloadPolicy` targeting
  `max_memory={0: budget, ...}` when `device="cuda"`) was implemented per
  spec but unexercised. Validated it for real against a free-tier NVIDIA
  T4 (Google Colab) instead — see
  `writeup/compute-efficiency-gpu-validation.md`. With a real, OS-enforced
  1.2GB VRAM cap and `pythia-410m` (~1.64GB fp32), the baseline genuinely
  fails (real `CUDA out of memory`) while the offload mechanism genuinely
  succeeds under the identical cap, splitting the model across GPU/CPU —
  **the spec's actual "critical experiment," confirmed on real hardware,
  not simulated.** Took three iterations to get a valid setup: run 1 (600MB
  cap, offload given the same 600MB as weight budget) OOM'd on *both*
  policies, surfacing a real design gap — `OffloadPolicy` doesn't reserve
  headroom for activations/CUDA context overhead, so a budget with no
  margin fails even when the model would otherwise fit.
  **Fixed (same day):** `resolve_max_memory`/`OffloadPolicy` now reserve
  `gpu_activation_headroom_fraction` (default 0.6, constructor-overridable)
  of the nominal `memory_budget_bytes` before handing the remainder to
  accelerate as the GPU weight budget — more conservative than the ~58%
  headroom that was empirically proven to work in the Run 3 validation
  above. `CPU` path is deliberately untouched (no hard OS-level RAM cap in
  this codebase to collide with, unlike `set_per_process_memory_fraction`
  for VRAM). Covered by new unit tests in `tests/test_efficiency_memory_policy.py`
  (`test_resolve_max_memory_on_cuda_reserves_activation_headroom`,
  `test_resolve_max_memory_on_cuda_respects_custom_headroom_fraction`,
  `test_offload_policy_describe_reflects_custom_headroom_fraction`). 221
  passing (was 219).
  **Re-verified on the real T4 (same day):** replayed Run 1's exact
  failing scenario (600MB cap, `OffloadPolicy` given 600MB as
  `memory_budget_bytes`, so the fix hands accelerate only 240MB as the
  weight budget) — **it still OOMs**, identical error as before the fix.
  Replaying the 1.2GB-cap scenario the fix was modeled on confirms real
  headroom is what made *that* case succeed. Conclusion, precisely: the
  fix does what it claims (stop handing accelerate a zero-margin weight
  budget — that's why 1.2GB works), but it has a floor — when the *total*
  nominal budget itself is below the fixed CUDA/framework overhead for a
  given model+GPU, no re-proportioning of that budget can succeed. 600MB
  is below that floor for `pythia-410m` on a T4; 1.2GB is above it. This
  is not "every budget now works," and is documented as such rather than
  letting the 1.2GB success stand in for more than it showed. Full numbers
  (both the re-verification failure and the original success) in
  `writeup/compute-efficiency-gpu-validation.md`'s "Re-verifying the fix"
  section.
  **Real installed package run end-to-end on GPU (same day, closes a
  previously-open gap):** `feat/compute-efficiency-engine` pushed to
  GitHub, cloned into a fresh Colab T4, installed via
  `requirements-training.txt`, run through its own CLI — not a mirrored
  script. `recommend --model EleutherAI/pythia-1.4b` correctly judged this
  bigger, more commercially-relevant model (1.4B params vs. the
  70M/410M used so far) fits on a full T4 without offload; `compare
  --memory-budget-mb 1500` forced the offload path too, for a genuine
  baseline-vs-offload run at this scale: baseline 48.42 tok/s at ~2.65GB
  VRAM, offload 1.585 tok/s (~30.5x slower) at ~586MB VRAM (~4.8x less
  memory — the clearest real memory win so far, paired honestly with its
  throughput cost). Offload's measured peak GPU memory (586MB) landed
  almost exactly on the headroom fix's computed weight budget
  (`1500MB * 0.4 = 600MB`), confirming the fix works correctly in the real
  package, not just the standalone script. Output correctness held (no
  divergence between policies).
  **New gap found by this run:** `BaselinePolicy` doesn't pass
  `torch_dtype` to `from_pretrained` — it loads whatever dtype the
  checkpoint is stored in on the Hub. `pythia-1.4b`'s checkpoint happens to
  be fp16 (confirmed via download size: 2.93GB, matching fp16's ~2.83GB
  estimate not fp32's ~5.66GB), which is why the advisor's fp16 estimate
  landed within 0.6% of the measured peak — a coincidence of this
  checkpoint's storage format, not because `recommend()`'s dtype choice is
  actually wired into how `BaselinePolicy` loads a model. A model stored in
  fp32 would silently load fp32 regardless of what was recommended.
  **Not yet done at the time, both addressed in later entries below:**
  - ~~The dtype-wiring gap just found~~ — fixed in the "Real quantized
    loading" entry below.
  - ~~`OffloadPolicy` has no viable-budget floor check~~ — addressed (not
    with a numeric floor, deliberately — see the "Clarify confusing
    underlying errors" entry below for why) in a later entry.
  - Single model family tested per environment (`pythia-70m` CPU,
    `pythia-410m` the one GPU run); no sweep across model sizes/budgets
    yet (the engine supports it, the CLI doesn't expose a sweep flag).
  - Peak CPU memory is a 50ms-interval polling-thread approximation, not
    a true OS peak-RSS reading (no portable equivalent on Windows).
  - On this specific CPU-only dev machine, `pythia-70m`'s ~134MB footprint
    fits trivially in RAM regardless of a nominal CPU budget, so the
    baseline-fails/offload-succeeds result could not be reproduced
    *locally* — only on the GPU run above.
  Tests: `tests/test_efficiency_hardware.py`,
  `tests/test_efficiency_results.py`,
  `tests/test_efficiency_memory_policy.py`, `tests/test_efficiency_api.py`
  (22 fast unit/API tests, no model load) plus
  `tests/test_efficiency_workload_real.py` (3 real-model tests against
  `pythia-70m` on CPU — gated behind `TRON_EFFICIENCY_RUN_MODEL_TESTS=1`,
  kept out of the default suite the same way
  `tron/training/benchmark_lora.py`'s ~20-minute real run is). 219
  passing in the default suite (was 197) — nothing under `tron/` touched.
- [x] **Automatic policy selection — `tron_efficiency/advisor/`** (same
  day). The product layer on top of the raw mechanism: instead of a human
  hand-picking `memory_budget_bytes`, `recommend(model_name, hardware)`
  fetches the model's `config.json` (not its weights — no multi-GB
  download) and analytically estimates its real parameter count from the
  architecture's shape, then picks a device/dtype/policy and, for
  `OffloadPolicy`, the nominal budget to pass it. The param-count formula
  is validated to within 0.04% against `pythia-70m`'s real measured size
  (70,398,464 estimated vs. 70,426,624 real, confirmed directly against
  the loaded model). Confidence (`high`/`medium`/`low`) and warnings are
  explicitly calibrated against the only two real GPU data points this
  project has — the 600MB-fails / 1.2GB-works pair from the GPU
  validation above — never reported as higher confidence than one real
  measurement earns. CLI (`recommend --model <name>`), API
  (`GET /recommend`), and dashboard form all wired in.
  **Known gap, stated in every relevant recommendation:** the advisor can
  recommend `int8`/`int4` when that's what makes a model fit, but
  `BaselinePolicy`/`OffloadPolicy` don't implement quantized loading yet
  (no `bitsandbytes`/`load_in_8bit` wiring) — such a recommendation is
  explicitly flagged as advisory, not executable via `run_workload` as it
  stands. Implementing real quantized loading is the natural next step —
  it's also the other major lever (alongside offload) for the actual
  commercial pitch (cheaper hardware per model served).
  **Other not-yet-done:** no handling for architectures the analytical
  formula can't estimate (returns `param_count=0`, falls back to the
  safest CPU recommendation rather than guessing); the CPU-side viable-
  budget logic uses a flat 1.5x safety margin, not anything empirically
  tuned the way the GPU thresholds are (no CPU equivalent of the
  600MB/1.2GB data points exists yet).
  Tests: `tests/test_efficiency_advisor.py` (11 cases — formula accuracy
  against a real pinned architecture, CPU/GPU branch coverage, confidence
  calibration at and around both confirmed data points, the quantization-
  executability warning) plus `tests/test_efficiency_api.py`'s
  `test_recommend_endpoint`. 233 passing (was 221).
- [x] **Real quantized loading (`int8`/`int4` via bitsandbytes) — closes
  the gap the advisor entry above flagged** (same day).
  `WorkloadConfig.dtype` now flows into an actual `torch_dtype=`/
  `quantization_config=` argument in both `BaselinePolicy` and
  `OffloadPolicy` (`memory/policies.py`'s `_loading_kwargs_for_dtype`) —
  not advisory, real `BitsAndBytesConfig(load_in_8bit=True)` /
  `load_in_4bit=True`, the same mechanism any `transformers` user has.
  `dtype=None` (the default) preserves the exact old behavior (no kwargs
  added at all) for backward compatibility. Verified locally, end to end,
  on real CPU hardware (no GPU needed for this part): `pythia-70m` loads
  and generates correctly as real int8 and int4 quantized weights via
  `run_workload`.
  **Real bitsandbytes constraint, surfaced honestly rather than hidden:**
  4-bit quantization cannot combine with device-map offload (only 8-bit
  has the `llm_int8_enable_fp32_cpu_offload` escape hatch) —
  `recommend()` now always picks `int8`, never `int4`, when it needs to
  recommend the offload policy, and `OffloadPolicy` itself raises a clear
  `UnsupportedDtypeCombination` (caught the same way `run_workload`
  already catches `ImportError`) if asked for int4+offload anyway, rather
  than failing confusingly deep inside `from_pretrained`. Verified: the
  rejection fires with the intended message, and real int8+offload
  succeeds at a large enough CPU budget.
  **Also fixes the dtype-wiring gap found during the real-GPU-package
  run** (`pythia-1.4b` loading fp16 only by checkpoint-storage
  coincidence, not because anything requested it) — `BaselinePolicy`/
  `OffloadPolicy` now actually honor an explicit `dtype`, closing that gap
  too, not just the quantization one.
  **New gap found while testing this, more fragile than first thought:**
  `int8` quantization combined with `OffloadPolicy` on CPU-only hardware
  behaves inconsistently across budgets. Probed `pythia-70m` at six CPU
  budgets: 100MB fails with accelerate's "whole model to disk" refusal
  (the same error unquantized offload gives below its own viable floor);
  110MB and 120MB fail with a *different* error —
  `RuntimeError: Cannot access accelerator device when none is
  available` — a CUDA-shaped failure on a machine with no CUDA at all;
  130MB, 150MB, 200MB, 250MB all report success. The 110-120MB failure
  mode specifically suggests bitsandbytes' `llm_int8_enable_fp32_cpu_offload`
  escape hatch may internally assume a reachable CUDA accelerator even
  when only splitting between CPU and disk — plausibly a real quirk of a
  CUDA-oriented library used in a CPU-only configuration, not something
  this project introduced. The "successes" above could not be confirmed
  as genuine disk-split placement either way: `model.hf_device_map` isn't
  populated the same way for quantized models as it is for plain-dtype
  offload, so the usual placement evidence this project relies on
  elsewhere isn't available here. Properly characterizing this needs a
  real GPU, not more CPU-only guessing — flagged for the next GPU
  session, deliberately not resolved here.
  Tests: 8 new cases in `tests/test_efficiency_memory_policy.py`
  (`_loading_kwargs_for_dtype`/`_requires_device_map` behavior for every
  dtype, the int4+offload rejection, backward-compat of `dtype=None`)
  plus 2 updated in `tests/test_efficiency_advisor.py`
  (`test_recommend_picks_quantized_dtype_without_unsupported_warning`,
  `test_recommend_never_pairs_int4_with_offload`). `bitsandbytes` added
  to `requirements-training.txt`.
- [x] **Clarify confusing underlying errors instead of inventing a
  numeric budget floor** (same day). `OffloadPolicy`'s "no viable-budget
  floor check" gap, flagged repeatedly above, can't honestly be closed
  with a hard-coded minimum byte threshold — the real floor is
  device/model-dependent and this project only has two confirmed real
  data points loosely bounding it (600MB fails, 1.2GB works, for one
  model+GPU pair). A universal number would be exactly the kind of
  unearned confidence this project has been deliberately avoiding.
  Instead: `_clarify_error()` in `workloads/runner.py` recognizes three
  known-confusing failure patterns already hit for real during
  development and rewrites them into actionable messages that cite what
  was actually measured, while preserving the original error text:
  accelerate's "offload the whole model to the disk" refusal (explains
  the real cause and the fix: raise the budget); the CUDA-shaped
  "Cannot access accelerator device when none is available" error found
  testing int8+offload on CPU-only hardware (names the known quirk,
  points at this file's entry above instead of looking unrelated); CUDA
  OOM specifically under `OffloadPolicy` on a CUDA device (cites the two
  real confirmed data points, scoped so a baseline OOM doesn't get a
  misleading citation). Everything else passes through unchanged — this
  clarifies known patterns, it doesn't mask unknown ones. Verified
  against a real failure (the 110MB int8+offload CPU case) that the
  clarified message actually appears instead of the raw error.
  Tests: 6 new cases in `tests/test_efficiency_error_clarity.py`.
  248 passing (was 242).
- [x] **Budget sweep — real tradeoff curve instead of isolated two-point
  comparisons** (same day). `benchmarks/engine.py`'s `run_budget_sweep`
  runs the same workload once per memory budget (plus one unconstrained
  baseline), producing one `BenchmarkResult` per budget — each an
  independent real measurement, nothing interpolated between points.
  `results/report.py`'s `generate_sweep_report` labels rows by budget
  (`offload @ 100MB`, not just `offload` colliding across every budget
  the way `generate_report`'s column-per-policy-name layout would).
  Wired into `cli.py` (`sweep --memory-budget-mb-list 100,150,250`),
  `api/app.py` (`POST /sweep`), and a dashboard form.
  **Real example run** (`pythia-70m`, CPU, 3 budgets — full numbers in
  `writeup/compute-efficiency-budget-sweep-example.md`): throughput holds
  essentially flat (~33 tok/s) from 150MB to 250MB, matching baseline,
  then hits a real cliff at 100MB — **8.6x throughput collapse** (32.7 →
  3.8 tok/s) for only an **8.6% memory saving** (490MB → 448MB). This is
  the same "doesn't degrade gracefully, falls off a cliff" shape the
  bigger GPU result (1.4B params on a T4) showed, now demonstrated with
  three real points on one curve instead of two isolated numbers from
  different sessions.
  **Honest scope:** sweeping *model sizes* automatically is still not
  built — each model is its own multi-GB download with its own real
  cost, left as an explicit per-model decision rather than something
  this project downloads on your behalf. One run per budget point (no
  repeated trials for variance) — good enough to show the cliff shape is
  real, not a statistically rigorous curve.
  Tests: 5 new cases in `tests/test_efficiency_benchmarks_engine.py`
  (baseline inclusion, budget ordering, headroom-fraction passthrough,
  correctness-divergence check still firing across a sweep), 6 new in
  `tests/test_efficiency_results.py` (sweep report row labeling/ordering/
  edge cases), 2 new in `tests/test_efficiency_api.py` (`/sweep`
  endpoint, with and without baseline). 260 passing (was 248).
