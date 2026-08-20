"""Post-hoc analysis plots for broad / svg_diagrams pipeline stats.

Reads scan_stats.json + selection_stats.json (and optional train_manifest.jsonl)
and writes paper-ready funnel / rejection / bucket / selection figures.

Usage:
  python -m data.scripts.broad_analyze --out data/processed/svg_diagrams
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "data" / "processed" / "svg_diagrams"

FUNNEL_STAGES = [
    ("HF train stream", "scanned"),
    ("Pass A (validate + VFIG)", "staging_after_pass_a"),
    ("Pass B (render)", "after_render_pass_b"),
    ("Pass C (pool)", "kept"),
]

REJECT_ORDER = [
    "vfig_low_clean",
    "validate",
    "phash_near_dup",
    "vfig_no_geometry",
    "too_long",
    "exact_dup",
    "vfig_too_many_complex",
    "test_leak",
    "render_fail",
    "path_soup",
]

BUCKETS = ["workflow_like", "geometry_like", "labeled", "other"]
SEL_REASONS = ["medoid", "complex", "rare", "reserve", "fill", "backfill"]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_funnel(scan: dict, select: dict, path: Path) -> None:
    stages = list(FUNNEL_STAGES) + [("Coreset (train)", "selected")]
    values = [int(scan.get(k, 0)) for _, k in FUNNEL_STAGES]
    values.append(int(select.get("selected", 0)))

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(stages))
    bars = ax.barh(y, values, color="#4c78a8", height=0.55)
    ax.set_yticks(y, stages)
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.set_title("Broad pipeline funnel (starvector/svg-diagrams train)")

    for bar, val, (_, key) in zip(bars, values, stages):
        pct = ""
        if key == "scanned" and val:
            pct = " (100%)"
        elif scan.get("scanned"):
            if key == "selected":
                pct = f" ({100 * val / scan['scanned']:.2f}%)"
            else:
                ref = scan.get(key, val)
                pct = f" ({100 * ref / scan['scanned']:.1f}% of scanned)"
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}{pct}", va="center", fontsize=9)

    _save(fig, path)


def plot_rejections(scan: dict, path: Path) -> None:
    reasons = scan.get("rejected_by_reason", {})
    labels, vals = [], []
    for r in REJECT_ORDER:
        if r in reasons and reasons[r]:
            labels.append(r.replace("_", "\n"))
            vals.append(int(reasons[r]))
    for r, v in sorted(reasons.items(), key=lambda x: -x[1]):
        if r not in REJECT_ORDER and v:
            labels.append(r.replace("_", "\n"))
            vals.append(int(v))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(labels))
    ax.barh(y, vals, color="#e76f51", height=0.6)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("rejected rows")
    ax.set_title("Scan rejections by reason")
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:,}", va="center", fontsize=8)
    _save(fig, path)


def plot_bucket_proportions(scan: dict, select: dict, path: Path) -> None:
    pool_hist = scan.get("bucket_histogram", {})
    sel_hist = select.get("bucket_histogram_selected", {})

    def _props(hist: dict) -> list[float]:
        total = sum(hist.get(b, 0) for b in BUCKETS) or 1
        return [100 * hist.get(b, 0) / total for b in BUCKETS]

    pool_p = _props(pool_hist)
    sel_p = _props(sel_hist)
    x = np.arange(len(BUCKETS))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, pool_p, width=w, label=f"pool (n={sum(pool_hist.values()):,})", color="#8ecae6")
    ax.bar(x + w / 2, sel_p, width=w, label=f"selected (n={select.get('selected', 0):,})", color="#e76f51")
    ax.set_xticks(x, BUCKETS, rotation=15)
    ax.set_ylabel("share (%)")
    ax.set_title("Structural bucket mix: pool vs coreset")
    ax.legend()
    _save(fig, path)


def plot_selection_reasons(select: dict, path: Path) -> None:
    counts = select.get("selection_reason_counts", {})
    labels = [r for r in SEL_REASONS if counts.get(r)]
    vals = [int(counts[r]) for r in labels]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#9d755d", "#bab0ac"]
    ax.pie(vals, labels=[f"{l}\n({v})" for l, v in zip(labels, vals)],
           autopct="%1.1f%%", colors=colors[: len(vals)], startangle=90)
    ax.set_title("Coreset selection quotas")
    _save(fig, path)


def write_summary_md(scan: dict, select: dict, embed: dict | None, path: Path) -> None:
    pool_hist = scan.get("bucket_histogram", {})
    sel_hist = select.get("bucket_histogram_selected", {})
    rej = scan.get("rejected_by_reason", {})
    lines = [
        "# Broad coreset run summary (auto-generated)",
        "",
        f"- **Source:** `{scan.get('hf_id', '?')}` @ `{scan.get('revision', '?')[:12]}…`",
        f"- **Scanned:** {scan.get('scanned', 0):,} → **pool:** {scan.get('kept', 0):,} → **train:** {select.get('selected', 0):,}",
        f"- **Yield:** {100 * scan.get('kept', 0) / max(scan.get('scanned', 1), 1):.2f}% pool, "
        f"{100 * select.get('selected', 0) / max(scan.get('scanned', 1), 1):.3f}% final coreset",
        "",
        "## Rejections (top)",
        "",
    ]
    for r in REJECT_ORDER:
        if rej.get(r):
            lines.append(f"- `{r}`: {rej[r]:,}")
    lines.extend(["", "## Pool buckets", ""])
    for b in BUCKETS:
        if pool_hist.get(b):
            lines.append(f"- `{b}`: {pool_hist[b]:,}")
    lines.extend(["", "## Selected buckets", ""])
    for b in BUCKETS:
        if sel_hist.get(b):
            lines.append(f"- `{b}`: {sel_hist[b]:,}")
    q = select.get("difficulty_quantiles_selected", {})
    if q:
        lines.extend([
            "",
            "## Difficulty quantiles (selected)",
            "",
            f"- p50: {q.get('p50', 0):.1f}, p90: {q.get('p90', 0):.1f}",
        ])
    if embed:
        lines.extend([
            "",
            "## Embed",
            "",
            f"- SigLIP rows: {embed.get('rows', '?')}, batch: {embed.get('batch_size', '?')}, "
            f"wall ~{embed.get('timing_s', {}).get('total', '?')}s",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(out_dir: Path, *, fig_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir)
    fig_dir = fig_dir or (out_dir / "figures" / "analysis")
    fig_dir.mkdir(parents=True, exist_ok=True)

    scan = _load_json(out_dir / "scan_stats.json")
    select = _load_json(out_dir / "selection_stats.json")
    embed_path = out_dir / "embed_meta.json"
    embed = _load_json(embed_path) if embed_path.is_file() else None

    plot_funnel(scan, select, fig_dir / "broad_funnel")
    plot_rejections(scan, fig_dir / "broad_rejections")
    plot_bucket_proportions(scan, select, fig_dir / "broad_bucket_proportions_pct")
    plot_selection_reasons(select, fig_dir / "broad_selection_reasons")
    write_summary_md(scan, select, embed, out_dir / "RUN_SUMMARY.md")

    return {
        "fig_dir": str(fig_dir),
        "figures": [
            "broad_funnel.png",
            "broad_rejections.png",
            "broad_bucket_proportions_pct.png",
            "broad_selection_reasons.png",
        ],
        "summary": str(out_dir / "RUN_SUMMARY.md"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analysis plots for broad/svg_diagrams stats")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fig-dir", type=Path, default=None)
    args = ap.parse_args()
    stats = analyze(args.out, fig_dir=args.fig_dir)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
