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
- The pool-health signal here is coarse: worker count and reported
  memory, heartbeat freshness. It does not yet use the real measured
  bandwidth/latency data `tron/spine/topology.py` already tracks
  internally inside a running `queue_server.py` process (that data isn't
  exposed via a public endpoint yet) — a real, worthwhile next
  improvement, not done here.
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
flag, not a provisioning mechanism. There is still no retry/reliability
handling if a pooled worker drops mid-job — a pooled job with no worker
available to finish it just never completes; nothing here adds the
reliability guarantee ROADMAP.md already names as missing for an
unreliable or untrusted participant. Every worker exercised by this
test is still this project's own machine, not a real external fleet.
