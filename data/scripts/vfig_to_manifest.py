"""Build eval manifests from QijiaHe/VFIG-Bench (400 ID + 198 OOD)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from structsvg_lib.svg_ops import TRAIN_RENDER_LONG_EDGE, normalize_svg, parse_svg, render_pil, validate_svg


def _repo_rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _write_row(row: dict, out_dir: Path, *, split: str, render: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    eid = row["id"]
    rec = {"id": eid, "split": split, "bench": "vfig"}
    if row.get("svg"):
        svg = row["svg"]
        root, err = parse_svg(svg)
        if root is not None:
            svg = normalize_svg(root)
        svg_path = out_dir / f"{eid}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        rec["svg_path"] = _repo_rel(svg_path)
        rec["svg"] = svg
        val = validate_svg(svg, try_render=False)
        rec["svg_sha256"] = val.sha256
    if row.get("image") is not None:
        img = row["image"].convert("RGB")
        if render:
            img = render_pil(row.get("svg") or "", size=TRAIN_RENDER_LONG_EDGE) if row.get("svg") else img
        png_path = out_dir / f"{eid}.png"
        img.save(png_path)
        rec["image_path"] = _repo_rel(png_path)
    return rec


def build_vfig_manifests(out_root: Path) -> dict:
    from datasets import load_dataset

    out_root = Path(out_root)
    stats = {}

    # ID: 400 with gold SVG
    ds_id = load_dataset("QijiaHe/VFIG-Bench", "VFIG-Bench", split="test")
    id_rows = []
    id_dir = out_root / "id"
    for i, row in enumerate(ds_id):
        fname = row.get("filename") or f"vfig_id_{i:04d}"
        eid = Path(fname).stem if fname else f"vfig_id_{i:04d}"
        svg = row.get("svg") or ""
        # render gold SVG for consistent 960px inputs
        try:
            img = render_pil(svg, size=TRAIN_RENDER_LONG_EDGE)
        except Exception:
            img = None
        rec = _write_row({"id": eid, "svg": svg, "image": img}, id_dir, split="id", render=False)
        id_rows.append(rec)

    id_man = out_root / "id_manifest.jsonl"
    with id_man.open("w", encoding="utf-8") as f:
        for r in id_rows:
            slim = {k: v for k, v in r.items() if k != "svg"}
            f.write(json.dumps(slim) + "\n")
    stats["id_n"] = len(id_rows)

    # OOD: 198 image only
    ds_ood = load_dataset("QijiaHe/VFIG-Bench", "VFIG-Bench-OOD", split="test")
    ood_rows = []
    ood_dir = out_root / "ood"
    for i, row in enumerate(ds_ood):
        fname = row.get("filename") or f"vfig_ood_{i:04d}"
        eid = Path(fname).stem if fname else f"vfig_ood_{i:04d}"
        img = row["image"].convert("RGB")
        png_path = ood_dir / f"{eid}.png"
        ood_dir.mkdir(parents=True, exist_ok=True)
        img.save(png_path)
        ood_rows.append(
            {
                "id": eid,
                "split": "ood",
                "bench": "vfig",
                "image_path": _repo_rel(png_path),
            }
        )

    ood_man = out_root / "ood_manifest.jsonl"
    with ood_man.open("w", encoding="utf-8") as f:
        for r in ood_rows:
            f.write(json.dumps(r) + "\n")
    stats["ood_n"] = len(ood_rows)
    stats["id_manifest"] = str(id_man)
    stats["ood_manifest"] = str(ood_man)
    print(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "vfig_bench")
    args = ap.parse_args()
    build_vfig_manifests(args.out)


if __name__ == "__main__":
    main()
