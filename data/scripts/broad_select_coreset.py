"""Stage 3: cluster + coreset select 2k broad training manifest."""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.scripts.broad_io import ErrorLogger, print_summary, progress_bar, write_json  # noqa: E402
from data.scripts.broad_scan_pool import load_pool_index, resolve_pool_image, structural_matrix  # noqa: E402
from structsvg_lib.broad_features import dedup_by_phash  # noqa: E402
from structsvg_lib.svg_ops import validate_svg

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

N_CLUSTERS = 4000
ALLOC_MEDOID = 1400
ALLOC_COMPLEX = 300
ALLOC_RARE = 200
ALLOC_RESERVE = 100
PHASH_NEAR_DUP = 3
PNG_SIZE = 448


def combine_features(visual: np.ndarray, structural: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    vis = visual.astype(np.float32)
    vis_norm = vis / (np.linalg.norm(vis, axis=1, keepdims=True) + 1e-8)
    struct_scaled = StandardScaler().fit_transform(structural.astype(np.float32))
    return np.concatenate([vis_norm, alpha * struct_scaled], axis=1)


def select_coreset(
    out_dir: Path,
    *,
    target_n: int = 2000,
    pilot: bool = False,
    alpha: float = 2.0,
    seed: int = 42,
) -> dict:
    pool_path = out_dir / "pool_index.parquet"
    visual_path = out_dir / "embeddings" / "visual_fp16.npy"
    if not pool_path.exists():
        raise FileNotFoundError(f"missing {pool_path}")
    if not visual_path.exists():
        raise FileNotFoundError(f"missing {visual_path}; run broad_embed first")

    df = load_pool_index(pool_path)
    visual = np.load(visual_path)
    if len(df) != len(visual):
        raise RuntimeError(f"pool rows {len(df)} != embeddings {len(visual)}")

    if pilot:
        target_n = min(40, len(df))
        n_clusters = min(80, len(df) // 2, target_n)
        alloc = _pilot_alloc(target_n)
    else:
        n_clusters = min(N_CLUSTERS, len(df))
        alloc = {
            "medoid": ALLOC_MEDOID,
            "complex": ALLOC_COMPLEX,
            "rare": ALLOC_RARE,
            "reserve": ALLOC_RESERVE,
        }

    rng = random.Random(seed)
    structural = structural_matrix(df)
    x = combine_features(visual, structural, alpha=alpha)

    if len(df) < n_clusters:
        raise RuntimeError(f"pool too small ({len(df)}) for clustering; re-run scan")

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(x)
    centers = kmeans.cluster_centers_

    selected: list[int] = []
    selection_reason: dict[int, str] = {}

    # Coverage medoids with sqrt(cluster_size) weighting
    cluster_ids = list(range(n_clusters))
    sizes = np.bincount(labels, minlength=n_clusters)
    weights = {c: max(1.0, float(sizes[c]) ** 0.5) for c in cluster_ids}
    ordered_clusters = sorted(cluster_ids, key=lambda c: weights[c], reverse=True)

    bar = progress_bar(total=len(ordered_clusters), desc="selecting medoids", unit="cluster")
    for c in ordered_clusters:
        if len(selected) >= alloc["medoid"]:
            break
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            bar.update(1)
            continue
        dists = np.linalg.norm(x[idx] - centers[c], axis=1)
        pick = int(idx[int(np.argmin(dists))])
        if pick not in selected:
            selected.append(pick)
            selection_reason[pick] = "medoid"
        bar.update(1)
    bar.close()

    # High complexity
    difficulties = df["difficulty"].to_numpy()
    complex_order = np.argsort(-difficulties)
    for idx in complex_order:
        if len([s for s in selected if selection_reason.get(s) == "complex"]) >= alloc["complex"]:
            break
        ii = int(idx)
        if ii not in selected:
            selected.append(ii)
            selection_reason[ii] = "complex"

    # Rare: smallest clusters
    rare_clusters = sorted(cluster_ids, key=lambda c: sizes[c])
    for c in rare_clusters:
        if len([s for s in selected if selection_reason.get(s) == "rare"]) >= alloc["rare"]:
            break
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        pick = int(idx[0])
        if pick not in selected:
            selected.append(pick)
            selection_reason[pick] = "rare"

    # Reserve random fill
    remaining = [i for i in range(len(df)) if i not in selected]
    rng.shuffle(remaining)
    for idx in remaining:
        if len([s for s in selected if selection_reason.get(s) == "reserve"]) >= alloc["reserve"]:
            break
        selected.append(idx)
        selection_reason[idx] = "reserve"

    # Fill to target_n
    for idx in remaining:
        if len(selected) >= target_n:
            break
        if idx not in selected:
            selected.append(idx)
            selection_reason[idx] = "fill"

    selected = selected[:target_n]

    # Post-selection phash dedup with backfill
    selected = _dedup_phash(selected, df, remaining, selection_reason, target_n, rng)

    manifest_rows = _materialize_assets(df, selected, selection_reason, out_dir, pilot=pilot, target_n=target_n, rng=rng)

    if len(manifest_rows) < target_n and not pilot:
        raise RuntimeError(f"only materialized {len(manifest_rows)}/{target_n} assets; check errors.jsonl")

    man_path = out_dir / ("candidates_pilot.jsonl" if pilot else "train_manifest.jsonl")
    with man_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")

    bucket_hist = {}
    for r in manifest_rows:
        bucket_hist[r["bucket"]] = bucket_hist.get(r["bucket"], 0) + 1

    stats = {
        "target_n": target_n,
        "selected": len(manifest_rows),
        "n_clusters": n_clusters,
        "alpha": alpha,
        "seed": seed,
        "allocation": alloc,
        "selection_reason_counts": _count_reasons(selection_reason, selected),
        "bucket_histogram_selected": bucket_hist,
        "difficulty_quantiles_selected": _quantiles([r["difficulty"] for r in manifest_rows]),
        "pilot": pilot,
    }
    write_json(out_dir / "selection_stats.json", stats)
    write_json(out_dir / "selection_ids.json", {"ids": [r["id"] for r in manifest_rows]})
    return stats


def _pilot_alloc(target_n: int) -> dict:
    return {
        "medoid": int(target_n * 0.7),
        "complex": int(target_n * 0.15),
        "rare": int(target_n * 0.1),
        "reserve": target_n - int(target_n * 0.7) - int(target_n * 0.15) - int(target_n * 0.1),
    }


def _count_reasons(reasons: dict[int, str], selected: list[int]) -> dict[str, int]:
    c: dict[str, int] = {}
    for i in selected:
        r = reasons.get(i, "unknown")
        c[r] = c.get(r, 0) + 1
    return c


def _quantiles(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {}
    a = np.array(vals)
    return {
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
    }


def _dedup_phash(
    selected: list[int],
    df,
    remaining: list[int],
    reasons: dict[int, str],
    target_n: int,
    rng: random.Random,
) -> list[int]:
    """Dedup selected indices using bucketed phash on dataframe rows."""
    pool_extra = [i for i in remaining if i not in selected]
    rng.shuffle(pool_extra)
    ordered = selected + pool_extra

    row_dicts = []
    for idx in ordered:
        r = df.iloc[idx].to_dict()
        r["_pool_idx"] = idx
        row_dicts.append(r)

    kept_rows, _ = dedup_by_phash(row_dicts, max_hamming=PHASH_NEAR_DUP)
    kept = [int(r["_pool_idx"]) for r in kept_rows[:target_n]]
    for idx in kept:
        if idx not in reasons:
            reasons[idx] = "backfill"
    return kept


def _materialize_assets(
    df,
    selected: list[int],
    reasons: dict[int, str],
    out_dir: Path,
    *,
    pilot: bool,
    target_n: int,
    rng: random.Random,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = ROOT.resolve()
    svg_out = out_dir / "svgs"
    png_out = out_dir / "pngs"
    svg_out.mkdir(parents=True, exist_ok=True)
    png_out.mkdir(parents=True, exist_ok=True)
    errors = ErrorLogger(out_dir / "errors.jsonl")
    rows: list[dict] = []
    tried: set[int] = set()

    candidates = list(selected)
    extra = [i for i in range(len(df)) if i not in selected]
    rng.shuffle(extra)
    candidates.extend(extra)

    bar = progress_bar(total=target_n, desc="writing assets", unit="file")
    for pool_idx in candidates:
        if len(rows) >= target_n:
            break
        if pool_idx in tried:
            continue
        tried.add(pool_idx)
        r = df.iloc[pool_idx]
        row_id = f"broad_{r['id']}"[:64]
        try:
            src = ROOT / r["svg_path"]
            if not src.exists():
                raise FileNotFoundError(f"missing {src}")
            svg_text = src.read_text(encoding="utf-8")
            val = validate_svg(svg_text, try_render=False)
            if not val.ok:
                raise ValueError(f"validate failed: {val.errors}")

            dst_svg = svg_out / f"{row_id}.svg"
            shutil.copy2(src, dst_svg)
            dst_png = png_out / f"{row_id}.png"
            pool_png = ROOT / r["png_path"] if r.get("png_path") else None
            if pool_png and pool_png.exists():
                from PIL import Image

                img = Image.open(pool_png).convert("RGB")
                if img.size != (PNG_SIZE, PNG_SIZE):
                    img = img.resize((PNG_SIZE, PNG_SIZE))
                img.save(dst_png)
            else:
                img = resolve_pool_image(r.to_dict(), render_size=PNG_SIZE)
                img.save(dst_png)

            row = {
                "id": row_id,
                "sha256": r["sha256"],
                "bucket": r["bucket"],
                "n_drawable": int(r["n_drawable"]),
                "n_elements": int(r["n_elements"]),
                "difficulty": float(r["difficulty"]),
                "selection_reason": reasons.get(pool_idx, "unknown"),
                "cluster_pool_idx": int(pool_idx),
                "svg_path": str(dst_svg.resolve().relative_to(root).as_posix()),
                "image_path": str(dst_png.resolve().relative_to(root).as_posix()),
                "source": r["source"],
            }
            if pilot:
                row["svg"] = svg_text
            rows.append(row)
            bar.update(1)
        except Exception as e:  # noqa: BLE001
            errors.log("select", row_id, r.get("sha256"), type(e).__name__, str(e))
    bar.close()

    if len(rows) < len(selected):
        logging.warning("asset materialization: %d/%d succeeded", len(rows), len(selected))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Select broad 2k coreset")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--target-n", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "broad")
    args = ap.parse_args()

    stats = select_coreset(
        args.out,
        target_n=args.target_n,
        pilot=args.pilot,
        alpha=args.alpha,
        seed=args.seed,
    )
    print_summary("select", kept=stats["selected"], target=stats["target_n"])
    out_name = "candidates_pilot.jsonl" if args.pilot else "train_manifest.jsonl"
    print(f"wrote {args.out / out_name}")


if __name__ == "__main__":
    main()
