"""Append-only JSONL store for CallRecords.

Local file, no network egress by default — this traces a caller's own LLM
usage, which may include sensitive prompts/business data, so sending it
anywhere without being asked is out of scope for this module entirely.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tron_costtrace.schema import CallRecord


class JsonlStore:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def append(self, record: CallRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def read_all(self) -> list[CallRecord]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(CallRecord.from_dict(json.loads(line)))
        return records
