# TRON integration guide

See the full pilot write-up: [`writeup/global-compute-pilot.md`](../writeup/global-compute-pilot.md).

## Surfaces

| Surface | Default | Auth |
|---|---|---|
| Job pool (`queue_server`) | `:9000` | Optional `TRON_SUBMIT_AUTH_TOKEN` → `X-TRON-AUTH` |
| Autopilot API | `:9100` | Optional `TRON_AUTOPILOT_API_TOKEN` → `X-TRON-AUTH` |
| Worker | connects to master | Per-worker `auth_token` from `/register_worker` |
| Python SDK | `tron_pilot.TronClient` | Uses both tokens from env when set |

## Constraints on `/plan` and `/run`

- `budget_usd`, `deadline_seconds`, `reliability` (`tolerant`|`strict`)
- `hourly_rate_usd` / `pooled_hourly_rate_usd` (user-configured; not a live feed)
- Distributed: `parallelizable`, `partition_items`, `map_op`

## What this phase does **not** include

Financial custody, lending, payment processing, or regulated financial products.
