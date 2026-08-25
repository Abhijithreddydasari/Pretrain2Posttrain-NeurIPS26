"""Write pred SVG attempts from a generations JSONL for quick visual inspection."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _extract_svg(text: str) -> str | None:
    m = re.search(r"(<svg[\s\S]*?</svg>)", text, flags=re.IGNORECASE)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.preds.parent / f"{args.preds.stem}_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    with args.preds.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rid = row["id"].replace("/", "_").replace("\\", "_")
            raw = out_dir / f"{rid}.txt"
            raw.write_text(row.get("pred_text", ""), encoding="utf-8")
            svg = _extract_svg(row.get("pred_text", ""))
            if svg:
                (out_dir / f"{rid}.svg").write_text(svg, encoding="utf-8")
                n += 1
    print(f"wrote previews to {out_dir} ({n} parseable SVG blocks)")


if __name__ == "__main__":
    main()
