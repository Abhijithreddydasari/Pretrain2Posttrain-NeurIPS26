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

## Public preview (do this first)

The `data/fixtures/` SVGs are **gold-recovery stubs**, not the visual target. Do not scale StructSVG from them.

```bash
python -m data.scripts.preview_public_svg
```

Then open `data/processed/public_preview/gallery.html` in a browser.

## Conditions

1. **Broad 2k** — filtered from `starvector/svg-diagrams` train pool  
2. **StructSVG 2k** — workflows + geometry with scene-graph sidecars  

External eval only: FlowGen, SVG-Diagrams test (~474).

Do not commit raw HF dumps, large PNG grids, or model outputs.
