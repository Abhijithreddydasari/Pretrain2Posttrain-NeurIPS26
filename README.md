# Pretrain2Posttrain Workshop, NeurIPS 2026

Does SFT on a multimodal base VLM teach **valid SVG first**, and **diagram structure** (entities, connections, layout) only later?

This repo is the code and data for a short paper aimed at the NeurIPS 2026 workshop [*Transitioning from Pre-Training to Post-Training*](https://pretrain2posttrain.github.io/call.html).

The setup is deliberately small and matched. **Gemma 4 E4B base**, LoRA SFT only, two 2k training conditions with similar token budgets:

1. **Broad** - heterogeneous public diagrams from [starvector/svg-diagrams](https://huggingface.co/datasets/starvector/svg-diagrams), filtered and coreset-selected.
2. **StructSVG** - controlled workflows + geometry with gold scene graphs and a compositional OOD split.

Primary eval is on StructSVG (validity, typed entity/relation F1, spatial aggregates). We save dense checkpoints (0, 5, 10, 20, 40, 60, 80, 100%) to see *when* syntax and structure move. VFIG-Bench is secondary eval only; we borrowed their code filter for broad curation but do not train on VFIG-Data in v0.

**Working claim:** post-training mostly buys SVG syntax/format; compositional structure lags unless the data and eval are built for it.

---

## What is done

| Piece | Status |
|-------|--------|
| Broad 2k data pipeline | Done (182k scanned → 2k train) |
| Eval harness + gold recovery | Done |
| StructSVG generator | Pilot manifests; full 2k next |
| Gemma E4B SFT + checkpoint curves | In progress |

---

## Pipeline

End-to-end flow for the workshop study: two matched 2k SFT conditions, dense checkpoints, structure-first eval on StructSVG.

```mermaid
flowchart TB
  subgraph Data
    HF["HF stream<br/>starvector/svg-diagrams<br/>182k rows"]
    VFIG["Validate + VFIG filter<br/>Clean ≥ 0.40, C ≤ 50"]
    POOL["Pool ~30k<br/>SigLIP + k-means"]
    BROAD["Broad train 2k"]
    GEN["StructSVG generator"]
    STRUCT["StructSVG train 2k<br/>workflows + geometry"]
  end

  BASE["Gemma 4 E4B base<br/>checkpoint 0%"]

  subgraph Train["LoRA SFT"]
    CKPT["Checkpoints<br/>0, 5, 10, 20, 40, 60, 80, 100%"]
  end

  subgraph Eval["Primary eval (StructSVG)"]
    VAL["Validity"]
    F1["Entity / relation F1"]
    SPA["Spatial aggregate"]
    OOD["ID vs compositional OOD"]
  end

  HF --> VFIG --> POOL --> BROAD
  GEN --> STRUCT
  BASE --> CKPT
  BROAD --> CKPT
  STRUCT --> CKPT
  CKPT --> VAL
  CKPT --> F1
  CKPT --> SPA
  CKPT --> OOD
```

Broad curation detail (Modal stages, dedup, quotas): [data/README.md](data/README.md).

---

## Broad 2k coreset

Training pairs for the heterogeneous condition. Built from the [starvector/svg-diagrams](https://huggingface.co/datasets/starvector/svg-diagrams) train split at revision `aacd39c8…` (seed 42). Scripts: [data/scripts/](data/scripts/).

| Stage | Count |
|-------|------:|
| HF train rows scanned | 182,144 |
| Pass A (validate + VFIG) | 44,807 |
| Pool after phash dedup | 30,011 |
| **Train coreset** | **2,000** |

About 1.1% of scanned rows survive to the final set. Most drops happen at the VFIG cleanliness filter (path-heavy SVGs) and canonical validation; two rows were removed for overlap with the held-out test hashes.

The coreset has **687** workflow_like, **1,296** labeled, and **17** geometry_like examples. Selection upsampled workflow_like diagrams slightly relative to the pool while keeping cluster coverage via SigLIP + structural-feature medoids.

<table border="0" cellspacing="20" cellpadding="0" align="center">
  <tr>
    <td align="center" valign="top" width="50%">
      <img src="assets/broad_funnel.png" alt="182k to 2k funnel" width="440"/>
      <br/>
      <em>Funnel: HF train stream → filtered pool → 2k coreset.</em>
    </td>
    <td align="center" valign="top" width="50%">
      <img src="assets/broad_rejections.png" alt="Rejection breakdown" width="440"/>
      <br/>
      <em>Where rows drop out before selection.</em>
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="50%">
      <img src="assets/broad_bucket_proportions.png" alt="Bucket proportions in pool vs coreset" width="440"/>
      <br/>
      <em>Pool vs selected bucket mix.</em>
    </td>
    <td align="center" valign="top" width="50%">
      <img src="assets/broad_coreset_coverage.png" alt="PCA coverage of pool vs coreset" width="440"/>
      <br/>
      <em>SigLIP + structural features: pool (gray) vs selected (color).</em>
    </td>
  </tr>
</table>

<p align="center">
  <img src="assets/broad_thumbnail_grid.png" alt="Sample thumbnails from coreset" width="640"/>
  <br/>
  <em>Random samples from the 2k train set (960×960 letterboxed PNGs).</em>
</p>

Processed outputs are written to `data/processed/svg_diagrams/` (gitignored). See [data/README.md](data/README.md) for Modal commands and how to regenerate these plots.

---

## Evaluation

Code: [structsvg_lib/](structsvg_lib/) (parse, render, scene-graph extract, metrics) and [eval/](eval/).

| Metric | Role |
|--------|------|
| Validity | XML parse + render + canonical profile (syntax) |
| Entity / relation F1 | Typed scene-graph match |
| Spatial aggregate | Workflow reachability; geometry relation accuracy |
| ID vs compositional OOD | Structure generalization |
| DINO | Secondary perceptual |
| Controls | Correct / shuffled / blank image |

Generations are cached once; metrics can be rescored without re-running the model.

```bash
python -m eval.gold_recovery
```

---

## Training (next)

Local smoke: E2B QLoRA on 8GB ([configs/train_e2b_qlora_smoke.yaml](configs/train_e2b_qlora_smoke.yaml)). Full runs: E4B LoRA on Modal ([train/modal_app.py](train/modal_app.py), [configs/train_e4b_broad.yaml](configs/train_e4b_broad.yaml)).

Loss is CE on **target SVG tokens only** (image + prompt masked). Checkpoints are saved at fixed fractions of the matched 2k schedule.

---

## Repo layout

```text
notes/          research statement, experiment card, canonical SVG spec
configs/        train / eval YAML
data/           schemas, broad + StructSVG scripts
train/          LoRA SFT + Modal entrypoints
eval/           runners, rubrics, fixtures
structsvg_lib/  shared SVG + graph + metrics
paper/          draft
assets/         README figures (from broad analysis)
```

More detail: [notes/research_statement.md](notes/research_statement.md), [notes/experiment_card.md](notes/experiment_card.md), [notes/canonical_svg.md](notes/canonical_svg.md).

---

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You need Gemma 4 **base** accepted on Hugging Face, `HF_TOKEN` set, and Modal secret `huggingface-secret` for cloud runs.

### Quick commands

```bash
# StructSVG pilot
python -m data.scripts.generate_structsvg --pilot

# Broad analysis plots (if processed data is local)
python -m data.scripts.broad_analyze --out data/processed/svg_diagrams

# Local train smoke
python -m train.lora_sft --config configs/train_e2b_qlora_smoke.yaml --dry-run

# Modal broad pipeline (full scan)
modal run data/scripts/modal_broad_app.py --stage all
```

---

## Related work

- **StarVector / SVG-Diagrams** - broad data source; we study post-training timing, not leaderboard scores.
- **VFIG** (He et al., 2026) - code filter for broad pool; structure-aware eval inspiration.
- **FlowGen** - external topology benchmark (planned).
- **LIMA / Chu** - claim style and ID/OOD framing.

---

## Citation

Paper draft: [paper/draft.md](paper/draft.md). Citation block will go here after submission.

If you use the broad curation scripts or StructSVG generator, please cite this repo once the workshop paper is public.
