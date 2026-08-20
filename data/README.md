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

# 3) Modal full run (recommended for ~182k scan)
modal run data/scripts/modal_broad_app.py --stage all
# pilot on Modal:
modal run data/scripts/modal_broad_app.py --stage all --pilot

# Per-stage gates (local or after Modal download):
python -m data.scripts.broad_checks --stage scan
python -m data.scripts.broad_checks --stage embed
python -m data.scripts.broad_checks --stage select
python -m data.scripts.broad_checks --stage all --pilot
```

**Outputs:** `train_manifest.jsonl` (2k), `scan_stats.json`, `selection_stats.json`, `errors.jsonl`, figures in `data/processed/broad/figures/` (or pass `--fig-dir paper/figures`).

Set `BROAD_TQDM=0` to disable progress bars (e.g. in CI).

## Conditions

| Condition | Train source | Role |
|-----------|--------------|------|
| **Broad 2k** | Coreset from `starvector/svg-diagrams` train pool | Heterogeneous real diagram SVG syntax |
| **StructSVG 2k** | `generate_structsvg.py` (workflows + geometry) | Structure-designed data with gold scene graphs |

**External eval only (never train):** FlowGen, SVG-Diagrams test (~474), **VFIG-Bench** (see below).

Do not commit raw HF dumps, large PNG grids, or model outputs.

## VFIG data — where it fits (and where it does not)

[VFIG](https://arxiv.org/abs/2603.24575) released **VFIG-Data** (~66k image–SVG pairs on HF `QijiaHe/VFIG-Data`) and **VFIG-Bench** (held-out scientific figures). We do **not** fold VFIG into the locked Broad 2k condition for v0 — that would change provenance and break the matched Broad vs StructSVG comparison mid-pipeline.

| Use | v0 (Aug 29) | Later |
|-----|-------------|-------|
| **VFIG-Bench eval** | Secondary external bench: score base + SFT checkpoints (validity, component metrics, optional VLM-judge) | Primary comparability to VFIG paper |
| **VFIG-Data as train** | No — keep Broad = StarVector pool | Optional 2k coreset ablation (“VFIG-shaped broad”) or curriculum stage |
| **VFIG-Data-Shapes-and-Arrows** (~6.5k) | No | Primitive-heavy third condition (closest to VFIG stage-1 curriculum) |
| **Cross-dedup** | If sampling VFIG later, dedup vs Broad + SVG-Diagrams test hashes | Same `build_test_hashes` + phash pipeline |

**Recommended v0 path:** finish StarVector broad coreset → train Gemma → eval on StructSVG (primary) + VFIG-Bench subset (secondary, no training on VFIG-Data).

Adapter script (planned): `data/scripts/vfig_to_manifest.py` — render figures, normalize to `notes/canonical_svg.md`, emit eval manifest only.
