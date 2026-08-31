# `tron.gpu` — virtual GPU aggregation + OpenAI-compatible gateway

This package was one of the three duplicated trees the rebuild collapsed
(`vgpu/` + `tron/gpu/` → one copy under `tron/`). The old standalone demo
scripts and docs referenced the removed `vgpu.*` import paths; this file
replaces them, pointing at `tron.gpu.*`.

## What's here

| module | role |
|---|---|
| `cluster.py` | `VirtualGPUCluster` / `VirtualGPUNode` — register heterogeneous GPU nodes, read an **aggregated** profile (summed VRAM / CUDA cores across nodes). Pure dataclasses + logic, no I/O. |
| `runtime.py` | `VirtualGPURuntime` — **simulated** allocation and task routing onto an aggregated cluster. Explicitly a simulation (see its docstring), for exercising the routing logic without real hardware. |
| `scheduler.py` | `TRON vGPU Master Scheduler` — a standalone FastAPI app with its own worker registration + job queue + SQLite ledger (`tron_vgpu_master.db`). Separate from the main `queue_server.py` spine. |
| `openai_bridge.py` | An OpenAI-compatible HTTP surface (`POST /v1/chat/completions`, `/v1/embeddings`, batch, agent-run) that turns each request into a TRON job and waits for a worker result. |

## Running the OpenAI bridge

```bash
pip install -r requirements.txt          # fastapi, uvicorn, requests, pydantic
uvicorn tron.gpu.openai_bridge:app --port 9800
```

```bash
curl -s http://localhost:9800/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

Set `TRON_API_KEY` to require an `Authorization: Bearer <key>` header;
leave it unset for open access. `TRON_MASTER_URL` points the bridge's
`TRONOpenAIProxy` at the TRON server that will run the jobs.

`tests/test_openai_bridge.py` exercises the request → OpenAI-shaped
response path with the proxy's `submit_job_and_wait` monkeypatched, so it
runs with no server and no GPU.

## Known gap (honest status)

`TRONOpenAIProxy` currently posts to `POST /submit_job` and polls
`GET /jobs` — endpoint names from the older standalone scheduler, **not**
the ones `queue_server.py` (the execution spine) exposes today (`/submit`,
`/status/{id}`, `/spine/task/{id}`). So the bridge talks to
`tron/gpu/scheduler.py`'s app, or to a shim, but not yet to the main
spine server unmodified. Reconciling the two job APIs — so an OpenAI
request becomes a real spine Task, visible in the Grid like any other —
is the remaining work to fold this gateway into the rest of the system.
It is tracked in ROADMAP.md under "Known follow-ups".
