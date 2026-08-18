# Data

## Layout

```text
data/
  schema/           scene-graph + SVG contracts
  scripts/          download, filter, StructSVG generators
  raw/              downloaded public data (gitignored)
  processed/        manifests + rendered pairs (gitignored)
  splits/           locked split IDs (small JSON committed when ready)
  fixtures/         tiny checked-in examples for tests
```

## Public preview (optional)

```bash
python -m data.scripts.preview_public_svg
```

Then open `data/processed/public_preview/gallery.html` in a browser.

## Broad 2k coreset pipeline (Phase A)

Four stages under `data/processed/broad/`:

```bash
# 0) Test-split dedup hashes
python -m data.scripts.build_test_hashes

# 1) Pilot (~5k scan → ~3–4k pool; embed all pool rows; select 40)
python -m data.scripts.broad_scan_pool --pilot
python -m data.scripts.broad_embed --pilot
python -m data.scripts.broad_select_coreset --pilot
python -m data.scripts.broad_visualize --pilot

# 2) Full run (~2–4 hrs; GPU recommended for embed)
python -m data.scripts.broad_scan_pool
python -m data.scripts.broad_embed
python -m data.scripts.broad_select_coreset
python -m data.scripts.broad_visualize
```

**Outputs:** `train_manifest.jsonl` (2k), `scan_stats.json`, `selection_stats.json`, `errors.jsonl`, figures in `paper/figures/`.

Set `BROAD_TQDM=0` to disable progress bars (e.g. in CI).

## Conditions

1. **Broad 2k** — coreset from `starvector/svg-diagrams` train pool  
2. **StructSVG 2k** — workflows + geometry with scene-graph sidecars  

External eval only: FlowGen, SVG-Diagrams test (~474).

Do not commit raw HF dumps, large PNG grids, or model outputs.
