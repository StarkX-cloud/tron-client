"""Extract TRON's distributed-training work into a minimal, standalone
repository — the "publish it as its own repo" half of the writeup task.

    python scripts/extract_training_repo.py [OUT_DIR]

Default OUT_DIR is ./dist/tron-distributed-training. The script only
copies files and writes a README + requirements; it does not run git and
does not push anywhere. After it runs:

    cd dist/tron-distributed-training
    git init && git add -A && git commit -m "Initial extraction"
    python -m pytest -q

Nothing is deleted from this repo.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (source path relative to REPO) -> included as-is
INCLUDE_DIRS = [
    "tron/spine",
    "tron/training",
    "tron_runtime",
]
INCLUDE_FILES = [
    "queue_server.py",   # the master the shard workers POST to (parity test drives it)
    "writeup/distributed-training.md",
    "examples/distributed_training/run_local.py",
    "examples/distributed_training/README.md",
    "tests/test_training_model.py",
    "tests/test_training_data.py",
    "tests/test_local_sgd.py",
    "tests/test_training_merge.py",
    "tests/test_spine.py",
    "tests/test_spine_integration.py",
    "tests/test_distributed_training.py",
    "pytest.ini",
    # The LoRA tests (test_lora_demo.py / test_lora_spine.py) need torch +
    # transformers + peft, which requirements.txt leaves optional — the
    # lora_demo.py / lora_spine.py modules themselves are still extracted.
]

REQUIREMENTS = """\
numpy>=1.24
scipy>=1.11
fastapi>=0.115
uvicorn>=0.30
pydantic>=2.0
httpx>=0.27
# numpy: the training algorithms. scipy: the spine matcher. fastapi/uvicorn/
# httpx: queue_server.py (the master) and the over-the-wire parity test.

# optional — the LoRA / Pythia-70M scale-up path (tron/training/lora_demo.py):
#   pip install torch transformers peft accelerate safetensors
"""

# A minimal package root — the full repo's tron/__init__.py re-exports a
# whole SDK surface (tron.remote, tron.client, ...) that this extraction
# doesn't carry. tron.spine / tron.training / tron.training.distributed
# are all imported as submodules and don't need anything from here.
MINIMAL_TRON_INIT = '"""TRON - extracted distributed-training subset. See writeup/."""\n'

README = """\
# TRON — low-communication distributed training

Extracted from the TRON project. This standalone tree contains the
distributed-training work and the execution "spine" it records into.

Start with **[writeup/distributed-training.md](writeup/distributed-training.md)** —
it explains the idea, the numbers, and exactly what is and isn't claimed.

## Layout

- `tron/spine/` — content-addressed artifacts + an append-only, replayable
  event log. Training runs record into this.
- `tron/training/` — the numpy MLP, non-IID sharding, local SGD, weight
  merging, LoRA-on-Pythia-70M, and `distributed/` (shards as separate
  processes exchanging parameter vectors over HTTP).
- `examples/distributed_training/run_local.py` — spin up a master + N
  shard subprocesses locally.
- `tests/` — every quantitative claim in the writeup is pinned here,
  including a bit-for-bit parity test between the in-process and
  over-the-wire runs.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q
python -m tron.training.benchmark
python -m examples.distributed_training.run_local --shards 3 --rounds 4
```

The LoRA scale-up (`python -m tron.training.benchmark_lora`) additionally
needs `torch transformers peft` and takes ~20 minutes CPU-only.
"""


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    out = Path(argv[0]) if argv else REPO / "dist" / "tron-distributed-training"
    if out.exists():
        print(f"refusing to overwrite existing {out} — remove it first")
        return 1
    out.mkdir(parents=True)

    copied = 0
    for d in INCLUDE_DIRS:
        src = REPO / d
        dst = out / d
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        copied += sum(1 for _ in dst.rglob("*") if _.is_file())
    for f in INCLUDE_FILES:
        src = REPO / f
        if not src.exists():
            print(f"  skip (missing): {f}")
            continue
        dst = out / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    (out / "tron" / "__init__.py").write_text(MINIMAL_TRON_INIT, encoding="utf-8")
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    (out / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n.tron_spine*/\n*.db\n.pytest_cache/\n", encoding="utf-8"
    )

    print(f"extracted {copied} files into {out}")
    print("next:")
    print(f"  cd {out}")
    print("  git init && git add -A && git commit -m 'Initial extraction'")
    print("  python -m pytest -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
