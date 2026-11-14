# TRON Intelligence Flywheel

**Thesis:** The machines execute the work. The data teaches TRON how to execute it better.

## Company hierarchy

| # | Stage | What TRON does | Measured by |
|---|---|---|---|
| 1 | Acquire workloads | Autopilot `/run`, pool `/submit`, SDK, pilots | ExperienceLedger workload count |
| 2 | Accumulate intelligence | Fingerprints + predicted vs actual + recovery | Quality rate, class count, hardware profiles |
| 3 | Convert to algorithms | Train/evaluate/promote Experience models | Holdout MAE vs naive baseline |
| 4 | Improve economics | Better runtime → tighter cost + placement | Holdout improvement %, cost MAE |
| 5 | Attract more | Providers + customers return | Fleet alive, feedback satisfaction |
| 6 | Compound | Each cycle strengthens the next | Active model beats baseline + growing classes |

## Why this is the moat

Commoditized compute is rented. **Operational execution data** — what ran where, how long, what failed, which optimizations worked — is hard to copy and compounds with volume.

TRON deliberately does **not** train shared models on prompts, source code, or model I/O without explicit opt-in (`governance.py`). The moat is structural execution intelligence, collected under tenant rules.

## Product wiring

```
Workloads → DecisionLog → ExperienceLedger → Learner → ModelRegistry
                              ↓                         ↓
                     flywheel.score_flywheel    Autopilot soft predictions
                              ↓
              /experience/flywheel + dashboard
```

## Run

```bash
# After real Autopilot runs have been logged:
PYTHONPATH=. python3 -c "from tron_experience import score_flywheel; import json; print(json.dumps(score_flywheel(), indent=2))"

# Or via API:
# GET /experience/flywheel
```

See also: `writeup/experience-engine.md`, `artifacts/experience_engine_proof.json`.
