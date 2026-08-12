"""JSONL experiment logger (W&B optional, not required)."""
from __future__ import annotations

import json
import time
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **kwargs):
        row = {"ts": time.time(), **kwargs}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
