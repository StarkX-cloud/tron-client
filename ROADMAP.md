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
