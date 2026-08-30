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

## What "bit-for-bit" costs, and what it doesn't claim

The bit-for-bit guarantee is about the *transport not corrupting the
computation*. It is not a claim about scale: this is still the
few-hundred-parameter numpy MLP on synthetic non-IID data. The LoRA /
Pythia-70M path (`tron/training/lora_demo.py`) is the "real model" story;
wiring *that* through this same transport is tracked in ROADMAP.md.
