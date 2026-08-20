# Train

LoRA/QLoRA SFT on a frozen VLM **base** checkpoint. Loss: autoregressive CE on **target SVG tokens only** (image + prompt masked).

## Pipeline overview

```text
Phase A — data (see data/README.md)
  broad 2k manifest  +  structsvg 2k manifest  (+ token-budget match)

Phase B — Gemma E4B (locked v0)
  base infer @ 0%  →  SFT broad 2k  →  SFT structsvg 2k
  save adapters @ 0,5,10,20,40,60,80,100%

Phase C — eval (see eval/README.md)
  StructSVG ID/OOD (primary)
  FlowGen, SVG-Diagrams test, VFIG-Bench (secondary)
  controls: correct / shuffled / blank image

Phase D — optional replication (post-v0 or if budget allows)
  second open **base** VLM, same protocol, sparse checkpoints
```

## Commands

```bash
# Data preview / dry-run (no GPU)
python -m train.lora_sft --config configs/train_e2b_qlora_smoke.yaml --dry-run

# Local 8GB smoke (E2B QLoRA) — requires HF access + CUDA
python -m train.lora_sft --config configs/train_e2b_qlora_smoke.yaml

# Base inference dry-run
python -m train.base_infer --manifest data/fixtures/smoke_manifest.jsonl --dry-run

# E4B conditions (Modal recommended)
modal run train/modal_app.py --task smoke
modal run train/modal_app.py --task train --config configs/train_e4b_broad.yaml
modal run train/modal_app.py --task train --config configs/train_e4b_structsvg.yaml

# Modal broad data pipeline (separate app)
modal run data/scripts/modal_broad_app.py --stage all --pilot
```

Configs: `configs/train_e2b_qlora_smoke.yaml`, `train_e4b_broad.yaml`, `train_e4b_structsvg.yaml`.

Modal secret: `huggingface-secret` with `HF_TOKEN`. Volumes: HF cache, `structsvg-data`, outputs.

## Training conditions (v0 locked)

| Run | Config | Manifest | N |
|-----|--------|----------|---|
| Base (0%) | — | — | no training; eval only |
| Broad SFT | `train_e4b_broad.yaml` | `data/processed/broad/train_manifest.jsonl` | 2k |
| StructSVG SFT | `train_e4b_structsvg.yaml` | `data/processed/structsvg/train_manifest.jsonl` | 2k |

Both SFT runs use **matched supervised token budget** (`match_token_budget: true`). Same prompt, canonical SVG profile, 960px letterboxed renders (`TRAIN_RENDER_LONG_EDGE`).

**Not in v0:** RL/GRPO, mixed SFT, training on VFIG-Data, full SVG-Stack.

## Pre→Post: one model vs several

The workshop asks what **post-training** changes relative to **pretraining**. That is answered primarily **within one architecture**:

- **Checkpoint 0%** = pretrained base on the same task + metrics (with and without SVG-prefix scaffold).
- **Checkpoints 5–100%** = what one SFT stage adds, and **when** (validity vs entity/relation F1 vs spatial).

A second (or third) model family is **not required** for a valid submission if the Gemma curves clearly separate syntax from structure and include base-vs-SFT tables.

| Tier | Models | Purpose |
|------|--------|---------|
| **v0 must** | `google/gemma-4-E4B` **base** | Locked claim; dense checkpoint curves; broad vs StructSVG |
| **v0 nice** | — | VFIG-Bench **eval only** on Gemma checkpoints (no VFIG train) |
| **v1 replication** | One open **base** VLM (e.g. `Qwen3-VL-4B` base if HF exposes it) | “Does syntax lead structure?” replicate on a second pretrain recipe |
| **Out of scope** | VFIG’s instruct checkpoints (`Qwen3-VL-4B-Instruct`, etc.) | Already post-aligned; confounds pre→post attribution |

Use **base** checkpoints only for cross-model comparison. VFIG trained on instruct models — cite them as related systems work, not as pretraining baselines.

If adding a second model: same 2k manifests, same checkpoint **percentages** (can subsample to 0/20/60/100% to save compute), same StructSVG eval. Frame as **replication**, not a leaderboard bakeoff.

## Relation to VFIG

- **VFIG** optimizes end-state figure→SVG quality (66k data, curriculum, SFT+RL).
- **We** measure emergence during **plain SFT from base** on matched 2k data.

Do not train on VFIG-Data for v0. After Gemma runs, optionally score checkpoints on VFIG-Bench to situate results next to their paper without competing on their training recipe.

## Gates (stop if)

- Gold SVGs fail `python -m eval.gold_recovery`
- Tiny batch cannot overfit on smoke config
- Shuffled-image scores ≈ correct-image (model ignoring vision)
- Assistant-only loss mask wrong on first real forward pass
