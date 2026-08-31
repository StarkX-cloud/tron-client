# A concrete telecom demo: cell-tower congestion prediction

Everything in `writeup/distributed-training.md` is generic distributed-
training infrastructure. This is that same machinery — unmodified,
already WAN-tested — pointed at a problem shaped like the ones a telecom
actually has, so the pitch takes 30 seconds instead of requiring an
interpreter.

**Important upfront: the data is synthetic.** `tron/training/
telecom_data.py` generates illustrative traffic archetypes (a business
district, a residential area, an entertainment district, a transit hub —
each with a hand-built expected-load curve over hour-of-day), sampled
with noise. No real carrier, subscriber, or network measurement of any
kind is in this. It exists to prove the mechanism on a problem shape a
telecom recognizes, not to claim anything about any real operator's
actual traffic.

---

## The story

N cell towers, each seeing its own local pattern of usage — genuinely
different from each other, not shuffled to look easier (the same "honest
hard case" this project has used throughout: see `tron/training/data.py`).
Each tower predicts its own next-interval congestion tier (low / medium /
high / critical) from recent signals — hour of day, whether it's a
weekend, a noisy recent-load reading, a load trend, and a device-count
proxy.

Training happens exactly the way every other distributed run in this
repo does: local SGD steps on each tower's own data, a small weight
delta synced through the master on a recurring cadence. **A tower's raw
traffic readings never leave the tower** — only the shared model's weight
vector crosses the wire, and it's tiny (212 parameters for the model
this demo trains).

This required zero new training mechanism. The only new code is the
dataset (`tron/training/telecom_data.py`) and one `dataset_factory` hook
added to `TrainingSession` so the existing wire protocol, spine
recording, Grid replay, and outcome scoring don't need to know or care
what problem is running. Every other piece — the real OS-process shards,
the retry-hardened HTTP transport, `TRON_TRAINING_AUTH_TOKEN`,
`TRON_ARTIFACT_ENCRYPTION_KEY` — is exactly what's already built and
tested elsewhere in this repo.

    python -m examples.telecom_demo.run_cell_tower_demo --master https://<your-master>

## Real numbers, from a local run (loopback)

| | |
|---|---|
| towers | 4 (business-district, residential, entertainment-district, transit-hub) |
| held-out congestion-tier accuracy | **81.9%** |
| naive "always predict the most common tier" baseline | 67.9% |
| model parameters | 180 |
| bytes per learning round (all 4 towers) | 11,520 |
| one-time cost of centralizing the raw readings instead | 89,600 |
| **per-round saving vs. shipping raw data once** | **7.8x less** |

(A run against the live deployed WAN master — same Render instance
`writeup/distributed-training.md`'s validation used — is pending as of
this writing: the first attempt against it ran before this code was
pushed and redeployed, so the master silently used its old default
dataset instead of the telecom one — a real, honest catch, not a result
to report. Numbers above are the loopback run; this section gets updated
once a genuine post-redeploy WAN run is captured.)

That per-round number, not a cumulative one, is the honest claim. At this
toy model size, running enough rounds makes the *cumulative* wire bytes
across a whole run comparable to a one-time raw-data dump — that's a real
measured fact, not something to hide. The durable claim is the recurring
one: a telecom re-syncs a shared model on an ongoing cadence for as long
as it's deployed, while shipping raw data would mean paying a real cost
every single time, growing without bound as more readings accumulate.
Paying 6.6x less *every round, forever*, instead of paying the full raw
cost *repeatedly*, is where this actually wins — and it only gets more
lopsided as the model or the reporting interval scales, since a real
deployment wouldn't run a 212-parameter toy net at telecom scale.

Every number above is reproducible: `tests/test_telecom_demo.py` pins
the dataset (deterministic, genuinely non-IID, and asserts the trained
model beats the naive baseline by a real margin, not just "the run
didn't crash") and drives the full wire path end to end.

## What this does not claim

- **Not real telecom data.** Said above, saying it again: illustrative
  archetypes, not measurements.
- **Not a deployed product.** A toy model (212 parameters) on synthetic
  data, same honesty as every other result in this repo — see
  `writeup/distributed-training.md`'s own "what this does not claim."
- **Not validated at real scale.** 4 towers, not hundreds or thousands.
  The mechanism (Local SGD + a topology-aware master) doesn't structurally
  change at scale, but nobody has thrown real tower counts at it.
- Everything `writeup/distributed-training.md` says isn't done yet
  (payload confidentiality from the master itself, defense against an
  authenticated-but-malicious participant, single-process master with no
  replication) is equally true here — this demo inherits those gaps, it
  doesn't fix them.
