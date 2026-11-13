# TRON Global Compute Network — External Pilot (Prompt 4)

**Status:** production-readiness surface shipped as `tron_pilot/`.  
**Does not** implement financial custody, lending, or payment rails.

## Final objective (demonstrated)

An independent developer can:

1. Register compute (worker) and obtain a durable `provider_id`
2. Verify the provider (operator challenge attestation)
3. Submit a real workload via Python SDK (`TronClient`)
4. Let Autopilot decide local / pooled / distributed
5. Receive the correct result
6. Inspect cost + usage history (estimates for external billing)

Artifact: `artifacts/external_pilot.json` (from `scripts/external_pilot_harness.py`).

## Package map

| Module | Role |
|---|---|
| `tron_pilot/client.py` | SDK: `plan`, `run`, `fleet`, `submit_callable`, usage report |
| `tron_pilot/provider.py` | Provider registry + verify challenge |
| `tron_pilot/provider_stats.py` | Per-worker measured outcomes |
| `tron_pilot/usage.py` | Customer usage ledger + report (hooks only) |
| `tron_pilot/tenancy.py` | Soft `customer_id` / `tenant_id` tags |
| `tron_pilot/security.py` | Trust model + auth header helpers |

Autopilot HTTP additions: `GET /trust`, `/providers`, `/providers/{id}/stats`, `/usage`.

## Integrate (copy-paste)

```bash
# Master
export TRON_SUBMIT_AUTH_TOKEN=change-me
uvicorn queue_server:app --host 0.0.0.0 --port 9000

# Provider machine
export TRON_MASTER_URL=http://master:9000
export TRON_WORKER_NAME=my-box
export TRON_WORKER_SKU=g6-nanode-1
export TRON_WORKER_HOURLY_RATE=0.0075
python3 worker.py

# Control plane
export TRON_AUTOPILOT_API_TOKEN=change-me-api
uvicorn tron_autopilot.api:app --host 0.0.0.0 --port 9100
```

```python
from tron_pilot import TronClient

client = TronClient(
    autopilot_url="http://127.0.0.1:9100",
    master_url="http://127.0.0.1:9000",
    customer_id="acme-dev",
)

# Register + verify provider (operator attestation)
rec, token = client.register_provider(
    worker_name="my-box",
    master_url="http://127.0.0.1:9000",
    region="us-east",
    register_with_master=False,  # already running worker.py
)
ch = client.providers.issue_challenge(rec.provider_id)
client.providers.verify(rec.provider_id, challenge=ch, verified_by="ops")

# Submit callable through the pool
result = client.submit_callable(lambda x: x * 2, 21)

# Or Autopilot plan/run (budget / deadline / reliability supported)
out = client.run(
    model_name="acme/job",
    hourly_rate_usd=0.5,
    budget_usd=1.0,
    reliability="tolerant",
    expected_runtime_seconds=2.0,
    efficiency=False,  # set True for real-model efficiency path
)
print(out["outcome"]["actual_cost_usd"])
print(client.usage_report())
```

## Trust boundaries (honest)

- Submit + Autopilot tokens optional but **recommended** in production
- Worker tokens required for heartbeat / claim / complete
- Provider `verified` = operator attestation, not hardware root-of-trust
- Sandbox = process timeout + RLIMIT only — **not** multi-tenant isolation
- Cost = hourly rate × wall time (estimate for external billing systems)

## Commercial hooks (no custody)

`UsageLedger` writes JSONL events and optionally calls a webhook.  
`usage_report(customer_id=…)` aggregates successes, failures, and estimated $  
for an external invoice system to consume. TRON does not hold funds.

## Reproducible pilot

```bash
PYTHONPATH=. python3 scripts/external_pilot_harness.py
# → artifacts/external_pilot.json
```
