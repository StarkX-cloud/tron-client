# Low-communication distributed training in TRON

A self-contained writeup of TRON's distributed-training work: the idea, the
two implementations (a hand-written numpy net and a real pretrained
transformer via LoRA), the move from in-process to over-a-real-socket, and
an explicit account of what is and isn't claimed. Everything here is
reproducible from this repository; every quantitative statement is pinned
by a regression test or recorded from a real run.

---

## The idea

Distributed training's dominant cost at scale is **communication**, not
compute: synchronizing model state across nodes every step. Two well-known
techniques attack that, and TRON implements both, end to end, at a size
anyone can rerun and verify:

1. **Local SGD (DiLoCo-style).** Each shard trains many steps entirely
   independently — zero communication — then the replicas are averaged
   once. `N` rounds of `K` local steps costs `N` syncs instead of `N·K`.

2. **Weight-space merging.** Train each shard's model to completion with
   *no* inter-node communication at all, then merge the weight deltas from
   a shared initialization (task arithmetic; TIES-Merging for the
   sign-conflict case).

On top of that sits the **low-rank-delta** thesis: when you fine-tune with
LoRA, the thing that has to cross the network each sync is a small
adapter (tens to hundreds of KB), not a full checkpoint (hundreds of MB).
Local SGD's sync-frequency reduction and LoRA's per-sync size reduction
compound.

---

## Result 1 — the algorithms, on a net small enough to audit

`tron/training/` contains a hand-written numpy MLP (one hidden layer,
ReLU, softmax + cross-entropy). Its backprop is checked against numerical
gradients (`tests/test_training_model.py`); its full parameter vector is a
few hundred floats, so every sync's byte count is exact.

The problem is deliberately hard, not cherry-picked: weakly separated
classes (`class_sep=1.0`) and a non-IID shard split where each of 4 shards
sees ~95% one class (`skew=0.95`).

    python -m tron.training.benchmark

| method | held-out accuracy | comm bytes | syncs |
|---|---|---|---|
| baseline (sync every step) | 87.2% | 1,356,800 | 200 |
| **local SGD** (10 steps / sync) | **85.6%** | **135,680** | 20 |
| solo shard (no sync, avg of 4) | 58.8% | 0 | — |
| merge: task arithmetic | 81.0% | 0 | 1 |
| merge: TIES | 81.6% | 0 | 1 |

- Local SGD: **10× less communication** for **1.6 points** of accuracy.
- Merging: **zero communication during training** recovers **81%**
  accuracy versus **58.8%** for an unmerged solo shard that only ever saw
  one class.

Pinned in `tests/test_local_sgd.py` and `tests/test_training_merge.py`.

---

## Result 2 — the same story on a real pretrained model (LoRA on Pythia-70M)

`tron/training/lora_demo.py` runs the identical local-SGD / merge
comparison against EleutherAI's **Pythia-70M**, fine-tuned with **LoRA**
(rank 8, on `query_key_value`) on the tiny-shakespeare corpus (public
domain, vendored). Numbers from one recorded real run (3 shards, 4 rounds
× 5 local steps):

    python -m tron.training.benchmark_lora

- **Adapter vs. full model:** 393,216 bytes vs. 282,099,712 bytes — a
  **717× smaller** unit of communication, before local SGD's
  sync-frequency reduction is even counted.
- **Local SGD:** eval loss **4.3296 → 4.2236** (genuinely decreasing —
  the pipeline trains), **4,718,592 bytes** over 4 syncs, versus
  **3,385,196,544 bytes** for a hypothetical full-model sync at the same
  cadence (the same 717×, compounding).
- **Merge:** solo-shard losses `[4.283, 4.3703, 4.3146]` (avg 4.3226,
  zero communication during training) → merged **4.2403**.

Honest cost: a full run is **~20 minutes** on this project's CPU-only dev
machine, so it is not a live pytest. `tests/test_lora_demo.py` covers the
pure logic (shard splitting, byte accounting, adapter averaging — one
test hand-verifies the adapter byte count against the LoRA math) fast and
offline, using a tiny synthetic module instead of downloading Pythia. The
full run was verified manually, once, with the numbers above recorded.

---

## From in-process to a real socket

Results 1 and 2 count the communication bytes a distributed run *would*
send, but the "shards" are Python objects in one process — nothing
crosses a wire. `tron/training/distributed/` closes that:

- Each shard is a **separate OS process**
  (`python -m tron.training.distributed.shard_worker --master <url>
  --session <id> --shard <k>`). It fetches its data, trains `local_steps`
  locally with zero communication, serializes its parameter vector, and
  `POST`s it to the master over HTTP.
- The master (`queue_server.py`'s `/training/session*` endpoints) stores
  each vector as a content-addressed spine Artifact, barrier-averages
  once every shard has reported for a round, and serves the merge back
  for the next round.
- Every `(round, shard)` is recorded as a spine Task, so a distributed
  run replays in the 3D Grid exactly like any other workload.

**The transport does not change the math.**
`tests/test_distributed_training.py` drives the full HTTP path and asserts
the final merged model is **bit-for-bit identical** to the single-process
`local_sgd.train_local_sgd` — same seeds, same schedule, same averaging
operation. If serialization ever corrupts a number, that test fails.

Running across genuinely separate physical machines is then a deployment
detail, not a code change: point `--master` at another host.

**Verified against a live public master, not just localhost.** A free
Render instance (`render.yaml`, Oregon region) was deployed as the master;
this repo's own machine, on an ordinary consumer network with no special
tuning, ran the shards against it over the public internet — a real WAN
link, not a loopback interface. Two things that only a real link surfaces:

- **Measured latency.** `time_connect` ~145ms, full TLS handshake ~300ms,
  a complete request round trip 430ms-930ms (`curl -w`, 5-sample spread).
  Nothing here was simulated or injected — it's what the actual path to
  Oregon costs.
- **A real fault, and a real fix.** The very first run hit intermittent
  connection-layer failures mid-run — TLS record corruption
  (`SSLV3_ALERT_BAD_RECORD_MAC`) and plain connection resets, roughly
  1-in-3 to 1-in-6 requests during the bad stretches, confirmed with both
  Python's `requests` and `curl` (so not a client-library quirk — a
  property of the path itself). `shard_client.RequestsTransport` had zero
  retry logic at that point: one bad handshake permanently killed a
  shard, full stop. That's exactly the failure mode a system aiming at
  real, imperfect telecom links cannot have. Fixed with exponential
  backoff + full jitter on transient failures (connection errors,
  timeouts, 5xx — never on a 4xx, which is a real application error, not
  a transient one), in both `RequestsTransport` (the code path shard
  processes actually use) and the `run_local.py` example harness's own
  calls. Re-run after the fix: same flaky link, one retry fired mid-run
  (visible in the run's own log output) and the run finished clean — 3
  shards, 4 rounds, 0.8375 held-out accuracy, 40,704 bytes over the wire.
  `GET /training/outcomes` on that live master confirms the outcome
  recorded `node_ids: ["shard-0","shard-1","shard-2"]` — the placement-
  feedback plumbing (see ROADMAP.md) working end to end against a real
  deployment, not just in a test process.

This is not a claim of production hardening — see "What this does not
claim" below — but it is a claim that the transport survives a real,
uncontrolled, intermittently-corrupting network path, because it was
handed one and did.

    python -m examples.distributed_training.run_local --shards 3 --rounds 4

reports the merged model's held-out accuracy plus two byte counts: the
bytes that **actually crossed a socket** (upload + download legs) and the
conceptual "one sync per shard per round" figure the in-process metric
counts.

`run_local.py --lora` runs the **same transport carrying LoRA adapters**:
each round's POST body is a ~400KB adapter state dict, the ~282MB
Pythia-70M base never moves (every shard process loads its own frozen
copy), and `tests/test_lora_over_wire.py` pins the merged adapter
tensor-for-tensor against the single-process run. That is the low-rank
delta thesis and the real-socket work in one demonstration.

Each finished run is also scored into `tron/orchestrator/outcomes.py` —
capability gained (held-out accuracy delta vs. the untrained init) per
unit of compute spent (total local SGD steps) — readable at
`GET /training/outcomes`.

---

## What this does *not* claim

- **Scale.** Result 1 is a few-hundred-parameter model on synthetic data.
  Result 2 is a 70M-parameter model fine-tuned with a rank-8 adapter on a
  ~1MB corpus for a few dozen steps. Neither is a frontier-scale training
  run. The goal was to implement and verify the *algorithms* correctly at
  a size anyone can rerun and check in minutes.
- ~~LoRA over the real wire.~~ **Done.** `distributed/lora_wire.py` +
  `LoraTrainingSession` send the LoRA adapter state dict (~400KB) as each
  round's POST body over the same transport; the ~282MB base model never
  moves. `run_local.py --lora` runs it against real Pythia-70M, and
  `tests/test_lora_over_wire.py` pins the merged adapter tensor-for-tensor
  against the single-process run. What's still not claimed here is
  *scale* — it's Pythia-70M with a rank-8 adapter, not a frontier model.
- **A production training system.** This is a demonstration of mechanism,
  with tests that pin every number, not a batteries-included framework.
- **Hardened for a hostile network, not just a flaky one.** The retry
  logic added above survives transient connection failures. It does not
  add encryption above what HTTPS already provides, or protection against
  an authenticated-but-malicious participant sending corrupt-but-
  well-formed data. The master is also a single process — one Render
  dyno, one point of failure, no replication. Real gaps before this is a
  system a telecom (or anyone) could run multi-tenant.
- **Authentication, since fixed.** Until this same push, every
  `/training/session*` route (and `/training/outcomes`) had zero access
  control — anyone with the master's URL could create sessions (a LoRA
  one loads a real base model — a resource-exhaustion vector) or submit
  fabricated "trained" data as any shard, silently poisoning the merge.
  `TRON_TRAINING_AUTH_TOKEN`, set on the master, now makes every one of
  those routes require a matching `X-TRON-AUTH` header — fail-closed
  when set, open only in the local-dev/test default with nothing
  configured. Same fix applied to a related, pre-existing bug in
  `/heartbeat`: it only checked the header *if the caller happened to
  send one*, so omitting it entirely was a silent bypass. Both are now
  fail-closed. `tests/test_training_auth.py` pins the guarded and
  open-by-default behavior, including a full two-shard run driven
  through an authenticated transport. What this is still not: defense
  against a caller who has the token but sends bad data on purpose.
- **At-rest encryption, since added.** Auth controls who can submit or
  fetch an artifact over the wire; it says nothing about someone who
  gets at the master's disk directly — a stolen backup, a misconfigured
  bucket, a curious host provider. Every training vector and adapter sat
  there in plaintext. `TRON_ARTIFACT_ENCRYPTION_KEY` now encrypts every
  artifact at rest (Fernet/AES via the `cryptography` package), invisible
  to content-addressing: `artifact_id` is always the hash of the
  *plaintext*, computed before encryption and checked after decryption,
  so nothing that references an artifact by hash has to know or care
  whether encryption is on. Enabling it on a deployment that already has
  artifacts on disk doesn't retroactively encrypt them, and old plaintext
  is still readable rather than treated as corrupted — the store can tell
  the difference honestly because content-addressing means it can check
  whether the raw bytes on disk already hash to the id being asked for.
  Also: a request presenting a valid `X-TRON-AUTH` token over plain HTTP
  (detected via `X-Forwarded-Proto: http`, the standard signal a
  TLS-terminating proxy sets) is now refused outright — the token itself
  is a secret, and no encryption story is complete while it can still
  cross the wire in cleartext. `tests/test_spine.py` (6 new cases) and
  `tests/test_training_auth.py` (3 new cases) pin both.
  Still not done: this does not hide data *from the master itself* — the
  master genuinely has to read each shard's plaintext vector to average
  it, so true end-to-end confidentiality would need secure aggregation
  (a much larger, different feature, not attempted here) — and it does
  not defend against a caller who holds a valid token but sends bad data
  on purpose.

---

## Reproduce everything

```bash
pip install -r requirements.txt -r requirements-training.txt
python -m pytest -q                                   # the whole suite, incl. the parity tests
python -m tron.training.benchmark                     # Result 1, seconds
python -m tron.training.benchmark_lora                # Result 2, ~20 min CPU-only
python -m examples.distributed_training.run_local --shards 3 --rounds 4   # over real sockets
```
