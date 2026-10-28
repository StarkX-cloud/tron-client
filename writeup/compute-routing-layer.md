# The routing layer: connecting single-node efficiency to the distributed pool

This is the first thing in this repo that connects two previously
separate systems: `tron_efficiency` (single-node memory virtualization —
offload, quantization, the advisor) and `tron`'s distributed spine/matcher
(`queue_server.py`, `tron/spine/topology.py`, `tron/spine/matcher.py` — real
bandwidth-aware placement across registered workers, built months earlier
as part of the original "run compute across heterogeneous unreliable
nodes" thesis).

## What it decides

`tron_router.decision.decide_placement(job)` picks one of three targets
for a job:

- **`local`** — this machine can run it without offload, using the real
  `tron_efficiency.advisor.recommend()` check (same real memory-threshold
  logic documented throughout `writeup/compute-efficiency-*.md`).
- **`pooled`** — local doesn't fit (or the job explicitly requires
  multiple nodes), and a real live `queue_server.py` instance is reachable
  with at least one worker whose heartbeat is fresh. This is a
  recommendation to route into the *existing* spine/matcher job path
  (`tron/spine/matcher.py` already does real Hungarian-algorithm optimal
  placement once a job is submitted there) — this module does not
  reimplement that logic, it decides a job should go there.
- **`dedicated`** — nothing else is a safe answer: local doesn't fit, no
  pool is reachable (or it's empty), or the job is marked intolerant of
  failure and the pool has no real reliability guarantee to offer it. This
  is a flag, not a provisioning action — this project does not call out to
  AWS/GCP/a GPU marketplace on a caller's behalf.

Every decision carries a `confidence` and explicit `rationale`/`warnings`,
same pattern as the memory advisor: never implying more certainty than
what was actually measured.

## Real, live verification (not just unit tests with fake data)

Started a real `queue_server.py` instance locally and drove
`tron_router` against it for real:

1. **No server reachable** (`master_url` pointing at an unused port):
   `query_pool_status` correctly reported `reachable=False` with the real
   `ConnectTimeout` error; `decide_placement` correctly recommended
   `dedicated`, confidence `low`.
2. **Server reachable, zero workers registered**: `query_pool_status`
   correctly reported `alive_worker_count=0`.
3. **Registered one real worker** via `POST /register_worker` against the
   live server: `query_pool_status` correctly picked up
   `alive_worker_count=1`, `total_reported_memory_gb=16.0` — the real
   number reported at registration, not invented.
4. **Full `decide_placement` against the live pool with that one real
   worker**, job marked `requires_multi_node=True`: correctly recommended
   `pooled`, confidence `low` (explicitly because one worker gives the
   matcher nothing to actually optimize across — stated as a warning, not
   silently upgraded to a confident recommendation).
5. **A real small model (`EleutherAI/pythia-70m`) with no multi-node
   requirement**: correctly recommended `local` without ever querying the
   pool at all — the local advisor's real memory-fit check short-circuits
   the whole pool-querying path when it isn't needed.

## Honest scope — what this is not

- Every "worker" tested against here was the project's own existing
  worker-registration mechanism, run locally. No real external fleet, no
  real untrusted third-party machine, has ever been tested against this
  router or the matcher it routes into. The project's existing honest gap
  ("no real second physical host," ROADMAP.md) is inherited here, not
  closed.
- "Dedicated" is a flag, not a mechanism. This project does not provision
  real cloud/dedicated compute on a caller's behalf, and `decide_placement`
  does not pretend otherwise.
- The pool-health signal here was coarse: worker count and reported
  memory, heartbeat freshness. It did not use the real measured
  bandwidth/latency data `tron/spine/topology.py` already tracks
  internally inside a running `queue_server.py` process, because that
  data wasn't exposed via a public endpoint.
  **(Closed below — see "Real network data" section.)**
- No automated test exercises the actual hand-off into
  `tron/spine/matcher.py`'s real job-submission path after a `pooled`
  decision — this module decides a job *should* go there; it doesn't yet
  verify that handing a job off from here into that path works end to end.
  **(Closed by `tron_router/handoff.py` and `tests/test_router_handoff.py`
  — see below.)**

## The hand-off: `tron_router/handoff.py`

`decide_placement` only ever decided. `submit_to_pool(job, decision,
func, func_args=..., func_kwargs=...)` actually executes a `pooled`
decision: it serializes `(func, args, kwargs)` with the exact same
`tron/serializer.py` cloudpickle+base64 encoding `@tron.remote` already
uses, POSTs it to the live master's real `/submit`, and returns a real
`tron.magic_future.MagicFuture` so the caller can block on
`.get()` for the real result once a worker picks the job up. It raises
`HandoffError` instead of silently proceeding if handed anything other
than a `pooled` decision — this function executes a decision, it does
not make one.

**A real bug was found while building this, not assumed away:**
`worker.py`'s `execute_job` was silently discarding every submitted
job's `function` payload and returning a fake echoed `{"status": "ok",
"message": prompt}` regardless of what was actually submitted. Since
`queue_server.py`'s `/submit` rejects any job that doesn't carry a
function payload, this meant *every* job ever submitted through
`@tron.remote` or this router — in this entire codebase, before this
fix — looked like it executed (real job ID, real `/status` polling,
a real-looking completed result) but the submitted function itself was
never actually called anywhere. Fixed: `execute_job` now deserializes
`job["function"]` via `tron.serializer.deserialize` and calls it for
real.

**Real, live, end-to-end verification** in `tests/test_router_handoff.py`
— the only test in this project that runs a real `uvicorn` server in a
background thread on a real socket (not FastAPI's `TestClient`, since
`submit_to_pool` and `worker.py` both use the real `requests` library
over real HTTP):

1. `submit_to_pool` with a `local`/`dedicated` decision raises
   `HandoffError` immediately — it never silently submits anyway.
2. A real `_add(3, 4)` function, submitted via `submit_to_pool` against
   the live master, is picked up by a real worker registration + one
   real `/next_job` → execute → `/complete` cycle. The worker's own
   returned result is the real computed `7`, not an echoed prompt.
   `future.get()` against `/status` also returns the real `7`.
3. A deliberately-failing function surfaces as a real
   `{"status": "error", "error": "ValueError: ..."}` job result — errors
   inside the submitted function aren't swallowed or turned into a fake
   success.

**Still honest scope, even after this:** `dedicated` is still just a
flag, not a provisioning mechanism. Every worker exercised by this test
is still this project's own machine, not a real external fleet.
**(The claim that used to sit here — "no retry/reliability handling if
a pooled worker drops mid-job" — turned out to be wrong when actually
checked; see "Dropped-worker reliability" below.)**

## Real network data: `/topology` and `measured_links`

`decide_placement`'s "pooled" recommendation used to look only at worker
count and self-reported memory. `tron/spine/topology.py` already tracks
the real signal that actually matters for "is this a cheap or expensive
pool to hand work to" — smoothed latency and bandwidth, measured from
real heartbeats and real payload transfers (`/probe/blob`,
`/probe/sink`) — but nothing outside `queue_server.py`'s own process
could read it.

`queue_server.py` now exposes `GET /topology`: per-registered-worker
`latency_ms`/`bandwidth_mbps`, `None` for either field when never
measured or aged out (never a fabricated 0). `query_pool_status` reads
it as a best-effort second call after `/workers` — if `/topology` isn't
reachable (an older master, or any other server answering `/workers`'
shape), `measured_links` is just `None`; the pool isn't marked
unreachable over it, since `/workers` already answered that question.

`decide_placement` now adds this to a `pooled` decision's rationale when
real data exists (`"2/2 alive worker(s) have real measured network
data: avg latency 20.0ms, avg bandwidth 75.0Mbps"`), and an honest
warning when it doesn't — either no alive worker has been measured yet,
or `/topology` itself couldn't be read. No invented latency/bandwidth
thresholds were added to the decision logic itself — there's no
calibration data yet for "how slow is too slow to pool," so this closes
the *visibility* gap (the real numbers are now surfaced, every time)
without pretending to close a *threshold* gap that would need real
measurement to calibrate honestly.

**Real verification:**
- `tests/test_queue_server_topology.py`: `GET /topology` against a real
  FastAPI `TestClient`, confirming a heartbeating worker's real latency
  shows up and a never-heartbeated worker shows `None` for both fields,
  not an error.
- `tests/test_router_topology.py`: the only test (alongside
  `test_router_handoff.py`) that runs a real `uvicorn` server on a real
  socket — proving `query_pool_status` reads real latency over real HTTP
  from a worker that actually heartbeated, and reads real `None`s for
  one that only registered.
- `tests/test_router_decision.py`: pure unit tests (injected
  `measured_links`, no network) for the three rationale/warning
  branches — data present, no alive worker measured yet, `/topology`
  unreachable.

**Honest scope, unchanged by this:** this is visibility, not a smarter
decision yet. `decide_placement` still picks `pooled` purely on worker
count/health, tolerant/intolerant, and multi-node — the real latency/
bandwidth numbers are reported alongside the decision, not yet used to
influence which target gets picked. Doing that for real would need
measured data on how much latency/bandwidth actually costs a given
workload, which this project doesn't have yet.

## Dropped-worker reliability: a claim that was wrong until it was checked

Several earlier sections of this document (and matching entries in
ROADMAP.md) said, flatly: "no retry/reliability handling if a pooled
worker drops mid-job — a pooled job with no worker to finish it just
never completes." That statement was never actually tested against
this project's own real submission path — it turns out to be false.

`queue_server.py` already has `_watchdog_loop`: every 5 seconds it
checks for a worker whose heartbeat has gone stale while it was marked
`busy`, and for each one, asks the spine's content-addressed event log
(`tron/spine/log.py`'s `recover_orphaned_tasks`) which jobs were last
assigned to that now-dead worker but never reached `completed`/`failed`.
Each orphaned job gets pushed back onto `job_queue` with its original
`job_id` and its original `function` payload intact, ready for another
worker's `/next_job` call to pick up.

Nothing had ever driven this *end to end through the actual path this
project uses to submit work* — `tron_router.handoff.submit_to_pool`'s
real `/submit` → `/next_job` → `/complete` → `/status` mechanics (the
same ones `@tron.remote` uses). `tests/test_pool_reliability.py` does,
for real, in one pass:

1. `submit_to_pool` a real function (`_multiply(6, 7)`) against a live
   master.
2. A real worker registers, claims the job via a real `/next_job` call
   (the master marks it `busy`, the job moves into `running_jobs`) —
   then goes completely silent. No heartbeat, no `/complete`. A real
   dead worker, not a simulated one.
3. The real watchdog thread (started manually in the test — it's
   normally only started under `queue_server.py`'s own `__main__`
   guard, which importing the ASGI app for a test server never
   triggers) notices the stale heartbeat and requeues the job — proven
   by polling the real `job_queue`/`job_store` state, not assumed.
4. A second real worker calls `/next_job`, receives the *same*
   `job_id` with its `function` payload still intact, actually
   deserializes and calls it, and reports completion.
5. The *original* `MagicFuture` returned by the first `submit_to_pool`
   call — never recreated — resolves to the real result (`42`), having
   survived the entire dead-worker-and-requeue cycle.

This passed on its first real run.

**What this does and does not prove:** it proves the requeue mechanism
works for one dead worker, one job. It does **not** prove anything
about a worker that's hung but still heartbeating (the watchdog only
catches a stale heartbeat, not a stuck-but-alive worker), or anything
at real untrusted-fleet scale (the project's standing "no real second
physical host" gap is unchanged by this). The honest lesson for future
work on this repo: a limitation written down once and repeated in
later entries without re-checking it can quietly go stale. This one did.

**The one real gap this test run *did* immediately turn up** — no
retry-count cap, so a job whose every assigned worker dies in turn
would requeue forever — was fixed right after, not left as a TODO.
`_watchdog_loop` now tracks `requeue_count` on the job dict itself
(added in `/submit`, default 0) and gives up once it exceeds
`MAX_JOB_REQUEUE_ATTEMPTS` (env `TRON_MAX_JOB_REQUEUE_ATTEMPTS`, default
3): the job resolves to a real `{"status": "error", "error": "gave up
after N requeue(s) -- ..."}` result — the same result shape a failing
submitted function already produces, not an invented new status — and
a real `"failed"` spine event is emitted so the task stops being
returned by `recover_orphaned_tasks` on a later watchdog pass. A second
new test drives this for real: `MAX_JOB_REQUEUE_ATTEMPTS` set to 2,
three real doomed workers in a row each claim and die on the same job,
and the job is confirmed (by reading real `job_store`/`job_queue`
state, not inferring it) to stop requeuing after the second real
requeue and resolve to a real error on the third.

## Testing on a real external fleet (2026-10-27/28)

Every test above, including the dead-worker one, ran against workers
on the *same machine* as the test process — this project's own
standing "no real second physical host" gap, repeated in this file
and `ROADMAP.md` for months. Closed it, cheaply: provisioned two real
Linode Nanode 1GB instances ($0.0075/hr each) in genuinely separate
data centers (Newark, Fremont), ran `queue_server.py` as the master on
one and `worker.py` as a real worker on the other, and submitted a
real job from a third machine (this dev box) through the actual
`decide_placement` → `submit_to_pool` path. It worked: real measured
network cost (136ms latency, ~15Mbps), a real `pooled` decision, real
execution on the Fremont node, and the real result back in 2.25s.

That one real test immediately surfaced two real bugs no mocked or
same-machine test had ever caught:

1. **`tron_router/decision.py`'s `query_pool_status` had a hardcoded
   2.0s timeout** across two sequential HTTP calls. Against real
   ~130ms WAN latency and a loaded single-vCPU master, this
   intermittently misreported a perfectly healthy pool as unreachable,
   silently downgrading a correct `pooled` decision to `dedicated`.
   Fixed: default raised to 5.0s, plus one retry.
2. **`worker.py`'s real polling loop read `job.get("job_id")`** to get
   the id for `/complete` — no job dict has ever had that key
   (`queue_server.py` uses `"id"` everywhere). Every job ever run
   through the *actual* loop reported completion to `/complete/None`
   instead of its real id, so the master's job never cleared and the
   submitter's `MagicFuture` waited forever for a result that had
   already been computed and silently discarded.
   `tests/test_router_handoff.py`'s existing "end-to-end" test never
   caught this because its helper called `report_complete(job["id"])`
   directly, bypassing the buggy line entirely. Fixed, and the test
   helper now uses `job.get("id")` the same way `main()` does, with an
   assertion that would have caught the original bug.

**A third, real, hard constraint found and documented rather than
fixed:** submitting a cloudpickled closure from a Python 3.11 process
to a worker running Python 3.12 fails with `SystemError: error return
without exception set` — cloudpickle pickles raw bytecode for
closures, which is not forward/backward compatible across CPython
minor versions. Both test nodes were pinned to 3.11 to match the
submitter for this session. This is not fixed and may not be cheaply
fixable without changing how jobs are serialized (e.g. moving away
from raw closures to importable, by-reference functions for anything
crossing a real heterogeneous fleet) — real production use of a
genuinely external pool needs either strict Python-version pinning
across every worker, or a different submission mechanism.

**A fourth anomaly, found but not root-caused:** testing the
dead-worker-across-real-machines path (kill the real worker process
over SSH mid-job, same assertions as the local test above, but with
the master on one real node and two real workers, one in Fremont and
one colocated with the master in Newark) showed the watchdog correctly
detecting the real dead worker (it was marked `offline` automatically)
and the system did not hang — the `MagicFuture` resolved rather than
timing out. But two re-runs produced results inconsistent with the
submitted job's own logic: a `slow_work(25)` closure that should block
for a real 25 seconds instead completed in 0.009s per the real spine
event log, and a separate run failed with `TypeError: 'module' object
does not support item assignment` on the second of two different
`__main__`-sourced closures deserialized in sequence by the same
long-lived worker process. Neither was root-caused this session.
Suspected cause for both: some interaction between cloudpickle's
handling of functions whose `__module__` is `"__main__"` and a
worker process that lives long enough to deserialize more than one
such closure over its lifetime — plausible given a real worker pool is
expected to run many jobs from many submitters over its life, but
unconfirmed. **Practical workaround that did work:** defining the
submitted function in a real importable module instead of a one-off
script's own `__main__` scope. Flagging this honestly as an open
finding rather than claiming the dead-worker path is proven at the
same confidence level as the pooling path above — it is not, yet.

**Security, addressed to a first layer, not solved:** every job
submission ships arbitrary code (a cloudpickled closure) that a worker
deserializes and runs with no sandbox — fine between processes one
company owns, a real problem the moment a second party's worker runs a
first party's job. Added a first real guardrail in `worker.py`:
`execute_job` now runs the deserialized function in a child process
(via `multiprocessing`'s `"fork"` context, Linux's default) with a
real wall-clock timeout (`TRON_JOB_TIMEOUT_SECONDS`, default 300s) and
a real address-space cap (`TRON_JOB_MEMORY_LIMIT_MB` via `RLIMIT_AS`,
default 2048MB), killing and reporting a real error instead of hanging
the worker forever or letting it OOM the host.
`tests/test_worker_job_guardrail.py` proves both for real — a job that
hangs forever, a job that allocates 500MB against a 100MB limit — and
was verified passing on a real Linode Linux node (Windows has no
`fork`, so worker.py falls back to the old unguarded behavior there
with a one-time warning; this project's local dev happens on Windows
but real workers run on Linux). **This is not a sandbox**: the job
still has the same filesystem and network access as the worker process
itself. No seccomp, no container, no restricted user, no network
policy. Real multi-tenant isolation is still open work.

**Honest bottom line on "is this ready for AI companies to use":**
closer, not there. The core external-pooling mechanism is now proven
on genuinely separate infrastructure, not just asserted. Two real bugs
that would have broken production use were found and fixed because of
that real test, not despite it. A first real resource guardrail
exists and is verified on real Linux hardware. But: the
cross-Python-version constraint is a real deployment requirement, not
a nuisance; the dead-worker-on-real-infra path has an unresolved
anomaly; and "a first layer" of execution safety is explicitly not the
same as the sandboxing a genuine multi-tenant marketplace would need.
Smaller teams running their own pool of machines they already trust
are a much closer fit today than "any AI company can plug untrusted
external workers into this."
