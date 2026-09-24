"""Minimal .env loader (no third-party dependency).

Loads KEY=VALUE lines from a .env file into ``os.environ`` so carrier
credentials can live in a file instead of the shell. Existing environment
variables win by default (``override=False``) so real deployment secrets are
never clobbered by a checked-in file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def load_env(path: Optional[str] = None, *, override: bool = False) -> Dict[str, str]:
    """Load a .env file into os.environ. Returns the keys it set.

    Silently does nothing if the file is absent. Supports ``KEY=VALUE``,
    ``export KEY=VALUE``, ``#`` comments, blank lines, and single/double
    quoted values. Not a full dotenv parser — no variable interpolation.
    """
    env_path = Path(path) if path else Path(".env")
    if not env_path.is_file():
        return {}

    applied: Dict[str, str] = {}
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
