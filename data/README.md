# Data

The submitted study trains on a curated **SVG-Diagrams coreset** and evaluates on held-out SVG-Diagrams and two VFIG subsets. **2,000 pairs were selected; 1,995 complete sequences were used for SFT.** VFIG is evaluation-only in this study.

Processed images, SVGs, and manifests are gitignored. This directory contains the scripts for preparing them, not a redistributable data release.

## Layout

```text
data/
  scripts/                    source adapters, curation, checks, visualization
  raw/                        downloaded source data (gitignored)
  processed/                  generated artifacts (gitignored)
    svg_diagrams/             broad train_manifest.jsonl, pngs/, svgs/
    svg_diagrams_test/        held-out test_manifest.jsonl and assets
    vfig_bench/               id_manifest.jsonl, ood_manifest.jsonl and assets
```

## Broad coreset provenance

Source: [starvector/svg-diagrams](https://huggingface.co/datasets/starvector/svg-diagrams), training split. The completed curation run records revision **`aacd39c8a8c82b2e5a0f81c10c4cbdc346ff7f0f`** and seed **42**.

The funnel is **182,144 scanned → 44,807 pass validation/filtering → 30,011 candidates after rendering and perceptual deduplication → 2,000 selected**. The processor-based 8,192-token sequence gate then removes five examples, giving **1,995 training pairs**.

The geometry-cleanliness filter, informed by VFIG, requires `(B + K) / N ≥ 0.40` and `C ≤ 50`: B counts rectangles/circles/ellipses, K lines/polylines, and C paths/polygons. Normalized SVG hashes exclude overlap with the held-out SVG-Diagrams test set; the manuscript reports no overlap against 442 valid test hashes. This is a source-test overlap audit, not a claim of complete decontamination against every external dataset.

SigLIP image embeddings and structural SVG features support clustering and diversity selection: **1,400 medoids, 300 high-complexity cases, 200 rare-cluster cases, and 100 reserve selections**. The selected 2,000 contain **1,296 labeled, 687 workflow-like, and 17 geometry-like** diagrams. These bucket counts precede the five training-time length exclusions.

![Curation funnel from 182,144 source examples to 2,000 selected pairs](../assets/broad_funnel.png)

![Examples from the broad coreset](../assets/broad_thumbnail_grid.png)

*Training rasters are letterboxed to 960 × 960. The funnel and gallery describe the selected coreset, before sequence-length filtering.*

Additional committed plots: [rejections](../assets/broad_rejections.png), [bucket proportions](../assets/broad_bucket_proportions.png), [embedding coverage](../assets/broad_coreset_coverage.png), and [difficulty](../assets/broad_difficulty_hist.png).

## Recreate the broad pipeline

Install the [root requirements](../requirements.txt). Large-scale curation uses model/data downloads; Modal additionally requires account setup. Run from the repository root.

```bash
# Held-out test hashes
python -m data.scripts.build_test_hashes

# Local pilot
python -m data.scripts.broad_scan_pool --pilot
python -m data.scripts.broad_embed --pilot --fresh
python -m data.scripts.broad_select_coreset --pilot
python -m data.scripts.broad_visualize --pilot

# Local full curation
python -m data.scripts.broad_scan_pool
python -m data.scripts.broad_embed --fresh
python -m data.scripts.broad_select_coreset
python -m data.scripts.broad_visualize
```

Alternatively, use the separate Modal data app:

```bash
modal run data/scripts/modal_broad_app.py --stage all --pilot
modal run data/scripts/modal_broad_app.py --stage all --fresh-embed
```

These are curation jobs, not SFT jobs. Preserve the recorded source revision, seed, filter settings, and output hashes when attempting an exact reconstruction. Inspect the scripts' current defaults before rerunning: [datasets.yaml](../configs/datasets.yaml) still contains unfilled revision fields, so a new run is not automatically an immutable replay.

For an existing completed run in your `structsvg-data` volume:

```bash
modal volume get structsvg-data broad/train_manifest.jsonl data/processed/svg_diagrams/
modal volume get structsvg-data broad/pngs data/processed/svg_diagrams/
modal volume get structsvg-data broad/svgs data/processed/svg_diagrams/
modal volume get structsvg-data broad/scan_stats.json data/processed/svg_diagrams/
modal volume get structsvg-data broad/selection_stats.json data/processed/svg_diagrams/
python -m data.scripts.broad_checks --stage select --out data/processed/svg_diagrams
python -m data.scripts.broad_analyze --out data/processed/svg_diagrams
```

The private volume is not a public download endpoint. New users must run curation or obtain a released artifact bundle. `broad_analyze` generates plots and a run summary in the processed output directory. See [training](../train/README.md) for uploading prepared pairs to the training volume and using the v2 config.

## Evaluation data

The benchmark adapters are implemented:

```bash
python -m data.scripts.vfig_to_manifest
python -m data.scripts.svg_diagrams_test_manifest
```

- **VFIG-Bench:** 400 source benchmark examples with reference SVG; the study scores a fixed subset of **128**.
- **VFIG-Bench-OOD:** 198 image-only examples; the study scores a fixed subset of **128**. Image similarity is possible without reference SVG, but gold-code comparison is not.
- **SVG-Diagrams test:** held-out examples from the training source; the study scores a fixed subset of **128**.

Retain the sweep's frozen subset manifests instead of independently resampling for each checkpoint. Both VFIG splits are external to the broad training source. See [evaluation](../eval/README.md) for sampling, coverage, metrics, and the 65-pair candidate diagnostic.

## Scope and reuse

VFIG-Data training, synthetic StructSVG, and curriculum experiments are not part of the submitted results. The `structsvg_lib` name survives only as shared infrastructure.

Source datasets retain their original licenses. No processed-data download or repository-level license is included yet. Do not infer redistribution rights from the availability of curation scripts.
