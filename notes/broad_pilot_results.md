# Broad pipeline — Modal pilot results

**Run:** Modal `structsvg-broad`, `--stage all --pilot`  
**Date:** 2026-08-20  
**Volume:** `structsvg-data` → `/root/data/processed/broad/`  
**HF revision:** `aacd39c8a8c82b2e5a0f81c10c4cbdc346ff7f0f` (`starvector/svg-diagrams`)

All gates: **PASS**

---

## Scan (5k pilot rows)

| Metric | Value |
|--------|------:|
| Scanned | 5,000 |
| Pass A survivors | 4,247 |
| After render (pass B) | 4,246 |
| After phash dedup (pool) | **1,773** |
| Rejected total | 3,227 |
| Render failures | 1 |
| Workers | 8 |

**Rejection breakdown:** validate 746 · too_long 7 · render_fail 1 · phash_near_dup 2,473

**Pool bucket histogram:** labeled 1,497 · workflow_like 266 · geometry_like 10

---

## Embed

| Metric | Value |
|--------|------:|
| Pool rows embedded | 1,773 |
| Embedding dim | 768 (SigLIP base) |
| Device | CUDA (L4) |
| Batch size | 64 |
| Errors | 0 |
| Wall time (reported) | ~16 min 12 s |

Note: 16 min is dominated by **per-batch PNG reads from Modal volume**, not GPU inference. Fixed in code via image preload + batch_size 128 (see `broad_embed.py`).

---

## Select (pilot target_n=40)

| Metric | Value |
|--------|------:|
| Selected | 40 |
| Clusters | 40 |
| Alpha (structural weight) | 2.0 |

**Allocation:** medoid 28 · complex 6 · rare 4 · reserve 2

**Selected buckets:** labeled 33 · workflow_like 7

**Difficulty quantiles (selected):** p25 18.6 · p50 34.6 · p75 38.4 · p90 87.8

---

## Visualize

Figures written to `/root/data/processed/broad/figures/`:

- `broad_coreset_coverage.png`
- `broad_bucket_distribution.png`
- `broad_difficulty_hist.png`
- `broad_thumbnail_grid.png`

---

## Known issues found post-pilot

1. **Missing diagram text in PNGs** — Graphviz SVGs use transparent backgrounds; PIL `RGBA→RGB` without compositing turned transparency black, hiding black labels. Fixed in `structsvg_lib/svg_ops.py` (white background + alpha composite + font mapping).

2. **Re-run required** — After render fix, re-run **scan pass B onward** locally or on Modal (`--fresh-embed` for embed) so pool PNGs and train PNGs include text.

3. **Nested `figures/figures/`** — If downloading from volume, use path `broad/figures/` not `broad/figures/figures/`.

---

## Commands to reproduce

```bash
modal run data/scripts/modal_broad_app.py --stage all --pilot
modal run data/scripts/modal_broad_app.py --stage check --pilot
```

After render fix (local smoke):

```bash
python -m data.scripts.broad_select_coreset --pilot --fresh  # if added
python -m data.scripts.broad_embed --pilot --fresh
```

Or full Modal re-run with `--fresh-embed`.
