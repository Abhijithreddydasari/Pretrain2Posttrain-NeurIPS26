"""Post-stage validation gates for the broad coreset pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.scripts.broad_io import ROOT, resolve_asset_path
from structsvg_lib.svg_ops import TRAIN_RENDER_LONG_EDGE

DEFAULT_OUT = Path("data/processed/broad")


def _fail(msg: str) -> dict:
    return {"ok": False, "error": msg}


def check_test_hashes(out_dir: Path) -> dict:
    path = out_dir / "test_hashes.jsonl"
    if not path.exists():
        return _fail(f"missing {path}")
    n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if n < 400:
        return _fail(f"test_hashes too small: {n} (expected ~443)")
    return {"ok": True, "test_hashes": n}


def check_scan(out_dir: Path, *, pilot: bool = False) -> dict:
    pool = out_dir / "pool_index.parquet"
    stats_path = out_dir / "scan_stats.json"
    if not pool.exists():
        return _fail(f"missing {pool}")
    if not stats_path.exists():
        return _fail(f"missing {stats_path}")

    import pandas as pd

    df = pd.read_parquet(pool)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    kept = int(stats.get("kept", len(df)))
    min_kept = 500 if pilot else 10_000
    if len(df) < min_kept:
        return _fail(f"pool too small: {len(df)} (min {min_kept})")
    if kept != len(df):
        return _fail(f"scan_stats.kept={kept} != parquet rows={len(df)}")

    png_dir = out_dir / "pool_pngs"
    sample = df.head(5)
    for _, row in sample.iterrows():
        p = resolve_asset_path(row["png_path"])
        if not p.exists():
            return _fail(f"missing pool png {p}")
        from PIL import Image

        img = Image.open(p)
        if img.size != (img.size[0], img.size[0]):
            return _fail(f"pool png not square letterbox: {p} {img.size}")

    return {
        "ok": True,
        "pool_rows": len(df),
        "scanned": stats.get("scanned"),
        "rejected": stats.get("rejected"),
    }


def _resolve_fig_dir(out_dir: Path, fig_dir: Path | None) -> Path:
    if fig_dir is not None:
        return fig_dir
    local = out_dir / "figures"
    paper = ROOT / "paper" / "figures"
    if local.exists() and any(local.glob("broad_*.png")):
        return local
    if paper.exists() and any(paper.glob("broad_*.png")):
        return paper
    return local


def check_embed(out_dir: Path) -> dict:
    pool = out_dir / "pool_index.parquet"
    visual = out_dir / "embeddings" / "visual_fp16.npy"
    if not visual.exists():
        return _fail(f"missing {visual}")

    import pandas as pd

    df = pd.read_parquet(pool)
    arr = np.load(visual)
    if len(df) != len(arr):
        return _fail(f"pool rows {len(df)} != embeddings {len(arr)}")
    if arr.ndim != 2 or arr.shape[1] < 256:
        return _fail(f"unexpected embedding shape {arr.shape}")

    meta_path = out_dir / "embed_meta.json"
    errors_logged = 0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        errors_logged = int(meta.get("errors_logged", 0))
    if errors_logged > 0:
        return _fail(f"embed logged {errors_logged} errors; fix before select")

    norms = np.linalg.norm(arr.astype(np.float32), axis=1)
    n_zero = int(np.sum(norms < 1e-6))
    if n_zero:
        return _fail(f"{n_zero} zero-norm embedding rows (failed image loads)")

    return {"ok": True, "embed_rows": len(arr), "embed_dim": int(arr.shape[1]), "errors_logged": errors_logged}


def check_select(out_dir: Path, *, pilot: bool = False, target_n: int = 2000) -> dict:
    target = min(40, target_n) if pilot else target_n
    man_name = "candidates_pilot.jsonl" if pilot else "train_manifest.jsonl"
    man = out_dir / man_name
    if not man.exists():
        return _fail(f"missing {man}")

    rows = [json.loads(line) for line in man.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < target:
        return _fail(f"manifest has {len(rows)} rows (expected >={target})")

    from PIL import Image

    bad_size = 0
    missing = 0
    for row in rows:
        p = resolve_asset_path(row["image_path"])
        if not p.exists():
            missing += 1
            continue
        img = Image.open(p)
        if img.size != (TRAIN_RENDER_LONG_EDGE, TRAIN_RENDER_LONG_EDGE):
            bad_size += 1

    if missing:
        return _fail(f"{missing} manifest images missing (checked all {len(rows)} rows)")
    if bad_size:
        return _fail(f"{bad_size} images not {TRAIN_RENDER_LONG_EDGE}px (re-run select with clean assets)")

    stats_path = out_dir / "selection_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    return {"ok": True, "selected": len(rows), "target_n": target, "stats": stats.get("bucket_histogram_selected")}


def check_visualize(out_dir: Path, *, fig_dir: Path | None = None) -> dict:
    fig_dir = _resolve_fig_dir(out_dir, fig_dir)
    required = [
        "broad_coreset_coverage.png",
        "broad_bucket_distribution.png",
        "broad_difficulty_hist.png",
        "broad_thumbnail_grid.png",
    ]
    missing = [f for f in required if not (fig_dir / f).exists()]
    if missing:
        return _fail(f"missing figures: {missing}")
    return {"ok": True, "figures": required, "fig_dir": str(fig_dir)}


def run_checks(out_dir: Path, *, stage: str, pilot: bool = False, target_n: int = 2000, fig_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir)
    checks = {
        "test_hashes": lambda: check_test_hashes(out_dir),
        "scan": lambda: check_scan(out_dir, pilot=pilot),
        "embed": lambda: check_embed(out_dir),
        "select": lambda: check_select(out_dir, pilot=pilot, target_n=target_n),
        "visualize": lambda: check_visualize(out_dir, fig_dir=fig_dir),
    }
    order = ["test_hashes", "scan", "embed", "select", "visualize"]
    if stage == "all":
        results = {name: checks[name]() for name in order}
        ok = all(r.get("ok") for r in results.values())
        return {"ok": ok, "results": results}
    if stage not in checks:
        return _fail(f"unknown stage {stage}")
    return checks[stage]()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Validate broad pipeline stage outputs")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stage", choices=["test_hashes", "scan", "embed", "select", "visualize", "all"], required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--target-n", type=int, default=2000)
    ap.add_argument("--fig-dir", type=Path, default=None)
    args = ap.parse_args()

    result = run_checks(args.out, stage=args.stage, pilot=args.pilot, target_n=args.target_n, fig_dir=args.fig_dir)
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
