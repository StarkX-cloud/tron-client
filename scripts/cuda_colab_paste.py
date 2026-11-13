# Paste this entire cell into a NEW Google Colab notebook.
# Runtime → Change runtime type → GPU (T4) → Save → Run.
#
# Produces: /content/economic_autopilot_cuda_benchmark.json
# Download that file (or copy the printed JSON) and drop it into the PR.

import os, json, pathlib, subprocess, sys

assert __import__("torch").cuda.is_available(), "Enable GPU runtime first (Runtime → Change runtime type → GPU)"
print("GPU:", __import__("torch").cuda.get_device_name(0))

REPO = "/content/tron-client"
BRANCH = "cursor/economic-autopilot-38d1"
if not pathlib.Path(REPO, ".git").exists():
    subprocess.check_call(
        ["git", "clone", "--branch", BRANCH, "https://github.com/StarkX-cloud/tron-client.git", REPO]
    )
os.chdir(REPO)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt",
     "-r", "requirements-dev.txt", "-r", "requirements-training.txt"]
)
out = "/content/economic_autopilot_cuda_benchmark.json"
subprocess.check_call(
    [
        sys.executable, "-m", "benchmarks.economic_autopilot",
        "--real-model", "--trials", "1",
        "--model", "EleutherAI/pythia-410m",
        "--max-new-tokens", "16",
        "--hourly-rate", "0.50",
        "--out", out,
    ]
)
subprocess.call(["nvidia-smi"])
text = pathlib.Path(out).read_text()
print(text[:4000])
try:
    from google.colab import files  # type: ignore
    files.download(out)
except Exception:
    print("Download manually:", out)
