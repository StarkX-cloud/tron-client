"""Phase 3: the distributed training demo.

This is deliberately small-scale (a hand-written numpy MLP, synthetic
data, seconds of CPU time) — the goal is to demonstrate the training
*algorithms* honestly, not to demonstrate a framework integration or a
large model. See benchmark.py for the runnable comparison and
ARCHITECTURE.md / ROADMAP.md for what scaling this up would need.

- model.py: a small MLP with manual forward/backward (numerically
  verified — see tests/test_training_model.py), plus flat-parameter
  get/set so weight-space operations are one-liners.
- data.py: synthetic classification data, non-IID sharded across
  simulated nodes (the honest hard case for distributed training).
- local_sgd.py: naive sync-every-step baseline vs. DiLoCo-style local SGD
  (many local steps between infrequent outer syncs), instrumented for
  communication bytes and wall-clock.
- merge.py: weight-space merging (task arithmetic, TIES-lite) as the
  zero-communication alternative — train fully independently, merge after.
"""
