"""Build SVG-Diagrams test eval manifest from starvector/svg-diagrams test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from structsvg_lib.svg_ops import TRAIN_RENDER_LONG_EDGE, extract_svg_blob, normalize_svg, parse_svg, render_pil, validate_svg


def _repo_rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_test_manifest(out_root: Path) -> dict:
    from datasets import load_dataset

    out_root = Path(out_root)
    png_dir = out_root / "pngs"
    svg_dir = out_root / "svgs"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("starvector/svg-diagrams", split="test")
    rows = []
    for i, row in enumerate(ds):
        eid = row.get("Filename") or row.get("filename") or f"svgdiag_test_{i:04d}"
        raw = row.get("Svg") or row.get("svg") or ""
        svg = extract_svg_blob(raw) or raw
        root_el, err = parse_svg(svg)
        if root_el is not None:
            svg = normalize_svg(root_el)
        val = validate_svg(svg, try_render=False)
        svg_path = svg_dir / f"{eid}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        try:
            img = render_pil(svg, size=TRAIN_RENDER_LONG_EDGE)
            png_path = png_dir / f"{eid}.png"
            img.save(png_path)
            image_path = str(png_path.as_posix())
        except Exception:
            image_path = None
        rows.append(
            {
                "id": eid,
                "split": "test",
                "bench": "svg_diagrams",
                "svg_path": _repo_rel(svg_path),
                "image_path": _repo_rel(png_path) if image_path else None,
                "svg_sha256": val.sha256,
            }
        )

    man_path = out_root / "test_manifest.jsonl"
    with man_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    stats = {"n": len(rows), "manifest": str(man_path)}
    print(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "svg_diagrams_test")
    args = ap.parse_args()
    build_test_manifest(args.out)


if __name__ == "__main__":
    main()
