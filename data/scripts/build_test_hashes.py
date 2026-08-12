"""Build hash list from SVG-Diagrams test split for train dedup."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from structsvg_lib.svg_ops import extract_svg_blob, validate_svg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "broad" / "test_hashes.jsonl")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    ds = load_dataset("starvector/svg-diagrams", split="test")
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for row in ds:
            raw = row.get("Svg") or row.get("svg") or ""
            svg = extract_svg_blob(raw) or raw
            val = validate_svg(svg, try_render=False)
            if not val.sha256:
                continue
            f.write(json.dumps({"id": row.get("Filename"), "sha256": val.sha256}) + "\n")
            n += 1
    print(f"wrote {n} test hashes → {args.out}")


if __name__ == "__main__":
    main()
