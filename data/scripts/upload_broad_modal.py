"""Upload broad 2k to Modal structsvg-data volume (Windows-safe)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    src = ROOT / "data" / "processed" / "svg_diagrams"
    if not (src / "train_manifest.jsonl").exists():
        raise SystemExit(f"missing {src / 'train_manifest.jsonl'}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = ["modal", "volume", "put", "structsvg-data", str(src), "data/processed/svg_diagrams"]
    print(" ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
