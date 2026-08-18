"""Stage 4: paper figures for broad coreset selection."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.scripts.broad_io import progress_bar, write_json  # noqa: E402
from data.scripts.broad_scan_pool import load_pool_index, structural_matrix  # noqa: E402
from data.scripts.broad_select_coreset import combine_features  # noqa: E402
from structsvg_lib.svg_ops import render_pil

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

BUCKETS = ["workflow_like", "geometry_like", "labeled", "other"]
SUBSAMPLE_POOL = 10_000


def _load_selected_indices(out_dir: Path) -> set[int]:
    stats_path = out_dir / "selection_stats.json"
    man_path = out_dir / "train_manifest.jsonl"
    pilot_path = out_dir / "candidates_pilot.jsonl"
    manifest = pilot_path if pilot_path.exists() and not man_path.exists() else man_path
    if not manifest.exists():
        manifest = pilot_path if pilot_path.exists() else man_path
    indices: set[int] = set()
    if not manifest.exists():
        return indices
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "cluster_pool_idx" in row:
            indices.add(int(row["cluster_pool_idx"]))
    return indices


def visualize(out_dir: Path, *, pilot: bool = False, alpha: float = 2.0, seed: int = 42) -> dict:
    pool_path = out_dir / "pool_index.parquet"
    visual_path = out_dir / "embeddings" / "visual_fp16.npy"
    if not pool_path.exists() or not visual_path.exists():
        raise FileNotFoundError("run scan + embed before visualize")

    fig_dir = ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_pool_index(pool_path)
    visual = np.load(visual_path)
    structural = structural_matrix(df)
    x = combine_features(visual, structural, alpha=alpha)

    selected_idx = _load_selected_indices(out_dir)
    if not selected_idx:
        # fallback: match by sha256 from manifest
        man = out_dir / ("candidates_pilot.jsonl" if pilot else "train_manifest.jsonl")
        if man.exists():
            sel_sha = set()
            for line in man.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    sel_sha.add(json.loads(line)["sha256"])
            selected_idx = {i for i, sha in enumerate(df["sha256"]) if sha in sel_sha}

    rng = np.random.default_rng(seed)
    pool_plot_idx = np.arange(len(df))
    if len(pool_plot_idx) > SUBSAMPLE_POOL:
        pool_plot_idx = rng.choice(pool_plot_idx, size=SUBSAMPLE_POOL, replace=False)

    plot_idx = np.unique(np.concatenate([pool_plot_idx, list(selected_idx)]))
    x_plot = x[plot_idx].astype(np.float32)

    reducer = PCA(n_components=2, random_state=seed)
    z = reducer.fit_transform(x_plot)

    pool_mask = np.isin(plot_idx, list(pool_plot_idx))
    sel_mask = np.isin(plot_idx, list(selected_idx))

    _plot_coverage(z, pool_mask, sel_mask, fig_dir / "broad_coreset_coverage")

    _plot_bucket_distribution(df, selected_idx, fig_dir / "broad_bucket_distribution")

    _plot_difficulty_hist(df, selected_idx, fig_dir / "broad_difficulty_hist")

    thumb_stats = _plot_thumbnail_grid(df, selected_idx, fig_dir / "broad_thumbnail_grid", pilot=pilot)

    stats = {
        "figures": [
            "broad_coreset_coverage.pdf",
            "broad_bucket_distribution.pdf",
            "broad_difficulty_hist.pdf",
            "broad_thumbnail_grid.pdf",
        ],
        "pool_subsample": int(len(pool_plot_idx)),
        "selected": len(selected_idx),
        "thumbnails_rendered": thumb_stats.get("rendered", 0),
    }
    write_json(out_dir / "viz_stats.json", stats)
    return stats


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_coverage(z: np.ndarray, pool_mask: np.ndarray, sel_mask: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(z[pool_mask & ~sel_mask, 0], z[pool_mask & ~sel_mask, 1], s=4, c="#888888", alpha=0.35, label="pool")
    if sel_mask.any():
        plt.scatter(z[sel_mask, 0], z[sel_mask, 1], s=12, c="#e76f51", alpha=0.85, label="selected")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Broad coreset coverage (PCA)")
    plt.legend(markerscale=2)
    _save_fig(path)


def _plot_bucket_distribution(df, selected_idx: set[int], path: Path) -> None:
    pool_counts = {b: 0 for b in BUCKETS}
    sel_counts = {b: 0 for b in BUCKETS}
    for i, row in df.iterrows():
        b = row["bucket"] if row["bucket"] in pool_counts else "other"
        pool_counts[b] += 1
        if i in selected_idx:
            sel_counts[b] += 1

    x = np.arange(len(BUCKETS))
    w = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - w / 2, [pool_counts[b] for b in BUCKETS], width=w, label="pool", color="#8ecae6")
    plt.bar(x + w / 2, [sel_counts[b] for b in BUCKETS], width=w, label="selected", color="#e76f51")
    plt.xticks(x, BUCKETS, rotation=15)
    plt.ylabel("count")
    plt.title("Bucket distribution: pool vs selected")
    plt.legend()
    _save_fig(path)


def _plot_difficulty_hist(df, selected_idx: set[int], path: Path) -> None:
    pool_d = df["difficulty"].to_numpy()
    sel_d = df.iloc[list(selected_idx)]["difficulty"].to_numpy() if selected_idx else np.array([])

    plt.figure(figsize=(7, 4))
    plt.hist(pool_d, bins=40, alpha=0.55, label="pool", color="#8ecae6", density=True)
    if len(sel_d):
        plt.hist(sel_d, bins=40, alpha=0.65, label="selected", color="#e76f51", density=True)
    plt.xlabel("difficulty score")
    plt.ylabel("density")
    plt.title("Difficulty distribution")
    plt.legend()
    _save_fig(path)


def _plot_thumbnail_grid(df, selected_idx: set[int], path: Path, *, pilot: bool) -> dict:
    if not selected_idx:
        return {"rendered": 0}

    # one per bucket + fill to 12
    picks: list[int] = []
    for b in BUCKETS:
        for i in selected_idx:
            if df.iloc[i]["bucket"] == b:
                picks.append(i)
                break
    rest = [i for i in selected_idx if i not in picks]
    picks.extend(rest[: max(0, 12 - len(picks))])
    picks = picks[:12]

    cols, rows_n = 4, 3
    fig, axes = plt.subplots(rows_n, cols, figsize=(10, 7.5))
    axes_flat = axes.flatten()

    rendered = 0
    bar = progress_bar(total=len(picks), desc="thumbnails", unit="img")
    for ax, idx in zip(axes_flat, picks):
        ax.axis("off")
        try:
            svg_path = ROOT / df.iloc[idx]["svg_path"]
            img = render_pil(svg_path.read_text(encoding="utf-8"), size=512)
            ax.imshow(img)
            ax.set_title(str(df.iloc[idx]["bucket"])[:12], fontsize=8)
            rendered += 1
        except Exception:  # noqa: BLE001
            ax.set_title("err", fontsize=8)
        bar.update(1)
    bar.close()

    for ax in axes_flat[len(picks) :]:
        ax.axis("off")

    plt.suptitle("Selected broad diagram samples")
    plt.tight_layout()
    _save_fig(path)
    return {"rendered": rendered}


def main():
    ap = argparse.ArgumentParser(description="Visualize broad coreset for paper")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "broad")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stats = visualize(args.out, pilot=args.pilot, alpha=args.alpha, seed=args.seed)
    print(f"wrote figures to {ROOT / 'paper' / 'figures'}: {stats['figures']}")


if __name__ == "__main__":
    main()
