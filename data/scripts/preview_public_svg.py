"""Pull a small public-diagram sample and write a browser gallery.

This is an optional visual audit for the public SVG-Diagrams source pool.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from structsvg_lib.broad_features import feature_bucket
from structsvg_lib.svg_ops import extract_svg_blob, validate_svg


HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Public diagram preview</title>
<style>
  body { font-family: Segoe UI, sans-serif; margin: 24px; background: #111; color: #eee; }
  h1 { font-size: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
  .card { background: #1c1c1c; border: 1px solid #333; border-radius: 8px; padding: 8px; }
  .meta { font-size: 12px; color: #aaa; margin-bottom: 8px; word-break: break-all; }
  .stage { background: white; height: 280px; overflow: auto; border-radius: 4px; }
  .stage svg { max-width: 100%; height: auto; display: block; }
  .bucket { color: #8ecae6; }
</style>
</head>
<body>
"""


def collect_svg_diagrams(split: str, n: int, scan_cap: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("starvector/svg-diagrams", split=split, streaming=True)
    out: list[dict] = []
    seen = 0
    for row in ds:
        seen += 1
        if seen > scan_cap or len(out) >= n:
            break
        raw = row.get("Svg") or row.get("svg") or ""
        svg = extract_svg_blob(raw) or raw
        val = validate_svg(svg, try_render=False)
        if not val.parse_ok or not val.normalized:
            continue
        if val.n_drawable < 2:
            continue
        if len(val.normalized) > 40000:
            continue
        b = feature_bucket(val.normalized)
        if b == "path_soup":
            continue
        out.append(
            {
                "id": row.get("Filename") or f"{split}_{seen}",
                "split": split,
                "bucket": b,
                "n_drawable": val.n_drawable,
                "n_elements": val.n_elements,
                "svg": val.normalized,
                "source": "starvector/svg-diagrams",
            }
        )
    return out


def write_gallery(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        HTML_HEAD,
        f"<h1>Public SVG preview ({len(rows)} samples)</h1>",
        "<p>starvector/svg-diagrams. Open this file in a browser. Click around; ignore our data/fixtures stubs.</p>",
        '<div class="grid">',
    ]
    for r in rows:
        svg = r["svg"]
        # keep inline SVG; strip XML declaration if present
        if svg.startswith("<?xml"):
            svg = svg.split("?>", 1)[-1]
        parts.append('<div class="card">')
        parts.append(
            f'<div class="meta"><span class="bucket">{html.escape(r["bucket"])}</span> · '
            f'{html.escape(str(r["split"]))} · {r["n_drawable"]} drawable · '
            f'{html.escape(str(r["id"])[:48])}</div>'
        )
        parts.append(f'<div class="stage">{svg}</div></div>')
    parts.append("</div></body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=40, help="from official-ish test split (~474)")
    ap.add_argument("--n-train", type=int, default=40, help="from the large HF train pool")
    ap.add_argument("--scan-cap", type=int, default=800)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "public_preview")
    args = ap.parse_args()

    rows: list[dict] = []
    print("loading svg-diagrams test...")
    try:
        test_rows = collect_svg_diagrams("test", args.n_test, args.scan_cap)
        print(f"  test kept {len(test_rows)}")
        rows.extend(test_rows)
    except Exception as e:  # noqa: BLE001
        print(f"  test failed: {e}")

    print("loading svg-diagrams train (stream)...")
    try:
        train_rows = collect_svg_diagrams("train", args.n_train, args.scan_cap)
        print(f"  train kept {len(train_rows)}")
        rows.extend(train_rows)
    except Exception as e:  # noqa: BLE001
        print(f"  train failed: {e}")

    if not rows:
        raise SystemExit("no rows collected — check HF access / datasets install")

    man = args.out / "preview.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)
    with man.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "svg"}) + "\n")
    html_path = args.out / "gallery.html"
    write_gallery(rows, html_path)
    print(f"open in browser:\n  {html_path}")


if __name__ == "__main__":
    main()
