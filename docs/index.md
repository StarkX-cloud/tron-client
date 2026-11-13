# TRON Documentation

TRON is a lightweight AI compute orchestrator for distributed Python workloads.

- Install: `pip install tron-client-py` (or clone this repo)
- Integration: [integration.md](integration.md)
- External pilot: [`writeup/global-compute-pilot.md`](../writeup/global-compute-pilot.md)
- Examples: [`../examples/`](../examples/)

## Control planes

| Surface | How to run |
|---|---|
| Job pool | `uvicorn queue_server:app --port 9000` |
| Autopilot | `uvicorn tron_autopilot.api:app --port 9100` |
| Python SDK | `from tron_pilot import TronClient` |

## What people search for

TRON helps with distributed Python compute, self-hosted AI inference, remote function execution, and model batch scoring — without forcing a Ray/Celery ops stack.
