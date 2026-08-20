"""Build hash list from SVG-Diagrams test split for train dedup."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.scripts.broad_io import ErrorLogger, ProgressTracker, print_summary  # noqa: E402
from structsvg_lib.svg_ops import extract_svg_blob, validate_svg

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def build_test_hashes(out_path: Path) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    ds = load_dataset("starvector/svg-diagrams", split="test")
    errors = ErrorLogger(out_path.parent / "errors.jsonl")
    n = 0
    skipped = 0
    total = len(ds)
    progress = ProgressTracker(total=total, desc="test hashes", unit="row", max_updates=20)

    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            row_id = row.get("Filename") or f"test_{n}"
            try:
                raw = row.get("Svg") or row.get("svg") or ""
                svg = extract_svg_blob(raw) or raw
                val = validate_svg(svg, try_render=False)
                if not val.sha256:
                    skipped += 1
                else:
                    f.write(json.dumps({"id": row_id, "sha256": val.sha256}) + "\n")
                    n += 1
            except Exception as e:  # noqa: BLE001
                errors.log("test_hashes", str(row_id), None, type(e).__name__, str(e))
                skipped += 1
            progress.tick(kept=n, rejected=skipped)
    progress.close()

    print_summary("test_hashes", kept=n, rejected=skipped, errors_logged=errors.count)
    print(f"wrote {n} test hashes → {out_path}")
    return {"kept": n, "rejected": skipped, "errors_logged": errors.count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "broad" / "test_hashes.jsonl")
    args = ap.parse_args()
    build_test_hashes(args.out)


if __name__ == "__main__":
    main()
