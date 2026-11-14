# TRON Experience Engine

**Status:** shipped as `tron_experience/` — learns from real Autopilot executions.  
**Does not** rewrite Autopilot, matcher, watchdog, or invent training rows.

## Mission

Make TRON progressively better at predicting and placing workloads using accumulated execution evidence — while hard budget / memory / security / reliability constraints stay authoritative.

## Architecture (thin layer)

```
DecisionLog / RuntimeHistory  ──ingest──►  ExperienceLedger
                                              │
                                    fingerprint (no prompts/code/I/O)
                                              │
                                    ExperienceLearner (train/holdout)
                                              │
                                    ModelRegistry (candidate→active only if beats baseline)
                                              │
                         TRON_EXPERIENCE_ENABLED=1
                                              ▼
                         EconomicAutopilot._predicted_runtime
                         (after history mean; before user hint)
```

## Components

| Module | Role |
|---|---|
| `fingerprint.py` | Structural workload class id |
| `ledger.py` | Workload Intelligence Ledger |
| `learn.py` | Clean → train → evaluate → version |
| `governance.py` | Tenant isolation, retention, shared opt-in, encrypt artifacts |
| `feedback.py` | Developer met-requirements / rating |
| `bridge.py` | Soft Autopilot wiring |
| `stats.py` + `dashboard/` | Intelligence dashboard |
| Autopilot API | `/experience/stats`, `/models`, `/feedback`, `/train`, `/experience/dashboard` |

## Hard constraint contract

Experience predictions only **fill** `predicted_runtime_seconds` when exact `RuntimeHistory` is missing. They never set `feasible=True`, skip `reliability_allows_pool`, or override matcher memory / efficiency / quarantine gates.

## Data governance

- Default: tenant-scoped ledger reads
- Cross-customer learning requires `TRON_EXPERIENCE_SHARED_LEARNING` + per-customer opt-in
- Fingerprints never include prompts, cloudpickle, or model I/O
- Optional Fernet encryption for model registry when `TRON_ARTIFACT_ENCRYPTION_KEY` is set
- Configurable retention via `TRON_EXPERIENCE_RETENTION_DAYS`

## Proof (real data)

```bash
PYTHONPATH=. python3 scripts/experience_engine_proof.py
# → artifacts/experience_engine_proof.json
```

Holdout on `artifacts/autopilot_decisions.jsonl` and transfer eval on `autopilot_real_decisions.jsonl` — learned MAE beats naive global-mean baseline; model promotes only when `beats_baseline`.
