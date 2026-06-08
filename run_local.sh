#!/bin/bash
# Run the TRON backend locally without Docker
# Usage: ./run_local.sh
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

source .venv/bin/activate

pip install -r requirements.txt

echo "Starting TRON queue server..."
python queue_server.py
