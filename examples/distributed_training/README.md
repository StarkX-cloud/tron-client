# Distributed training over a real wire

This is Phase 3's local-SGD demo with the shards moved out of the process.
Each shard is its own OS process; parameter vectors are serialized and
sent to the master over HTTP; the master stores each as a content-addressed
spine Artifact, barrier-averages once every shard has reported for a round,
and serves the merge back for the next round.

The math is unchanged from `tron/training/local_sgd.py` — same seeds, same
schedule, same averaging op — so the final merged model is **bit-for-bit
identical** to the single-process run. `tests/test_distributed_training.py`
pins that.

## Run it on one machine (separate processes)

```bash
python -m examples.distributed_training.run_local --shards 3 --rounds 4
```

Starts a `queue_server.py` on a free port, opens a training session, spawns
three `shard_worker` subprocesses, and prints the merged accuracy plus the
bytes that actually crossed a socket.

## Run it across genuinely separate machines

Nothing in the code changes — only the `--master` URL.

1. On the host that will be the master:

   ```bash
   PORT=9000 python queue_server.py
   ```

2. From any machine, define the run:

   ```bash
   curl -s -X POST http://<master-host>:9000/training/session \
     -H 'content-type: application/json' \
     -d '{"num_shards": 3, "num_rounds": 4, "local_steps": 5}'
   # -> {"session_id": "trainsess-abc123...", ...}
   ```

3. On each of the three shard machines:

   ```bash
   python -m tron.training.distributed.shard_worker \
     --master http://<master-host>:9000 \
     --session trainsess-abc123... \
     --shard 0        # 1 on the next machine, 2 on the one after
   ```

4. Read the result and replay it in the Grid:

   ```bash
   curl -s http://<master-host>:9000/training/session/trainsess-abc123.../result
   # open http://<master-host>:9000/grid/ and scrub the event log
   ```

## LoRA adapters over the same wire

```bash
python -m examples.distributed_training.run_local --lora --shards 2 --rounds 3
```

Needs `torch transformers peft`. Now the round's POST body is a LoRA
**adapter state dict** (~400KB for Pythia-70M) instead of a numpy vector,
and the ~282MB base model never crosses the socket — each shard process
loads its own frozen copy from the HF cache. This is the "low-rank delta,
not a full checkpoint, is the unit of communication" thesis running over
a real network. `tests/test_lora_over_wire.py` pins the merged adapter
tensor-for-tensor against the single-process `lora_demo.run_local_sgd_lora`.

## What "bit-for-bit" costs, and what it doesn't claim

The bit-for-bit / tensor-for-tensor guarantee is about the *transport not
corrupting the computation*. It is not a claim about scale: the numpy run
is a few-hundred-parameter MLP on synthetic data; the LoRA run is
Pythia-70M with a rank-8 adapter on ~1MB of text for a few dozen steps.
The point is that the mechanism is correct and the transport is faithful.
