# Train

LoRA/QLoRA SFT on a frozen VLM **base** checkpoint. Loss: autoregressive CE on **target SVG tokens only** (image + prompt masked).

## Pipeline overview (v0)

```text
Phase A — data (done)
  broad 2k manifest  →  data/processed/svg_diagrams/

Phase B — Gemma E4B (locked v0)
  base infer @ 0%  →  SFT broad 2k
  save adapters @ 0,5,10,20,40,60,80,100%

Phase C — eval (see eval/README.md)
  VFIG-Bench 400 (primary, gold SVG)
  VFIG-Bench-OOD 198 (image-only generalization)
  SVG-Diagrams test, controls (secondary)

Phase D — optional follow-up (post broad curves)
  (a) sequential SFT: continue adapter on VFIG-Data 2k coreset, or
  (b) fresh base + SFT on VFIG-Data 2k only
  Re-eval on VFIG-Bench (must exclude train IDs from the 400 test set)
```

## Commands

```bash
# Data preview / dry-run (no GPU)
python -m train.lora_sft --config configs/train_e4b_broad.yaml --dry-run

# Local 8GB smoke (E2B QLoRA) — if processor loads on your transformers build
python -m train.lora_sft --config configs/train_e2b_qlora_smoke.yaml

# Base inference dry-run
python -m train.base_infer --manifest data/processed/svg_diagrams/train_manifest.jsonl --dry-run

# E4B broad SFT (Modal — L4 24GB)
modal run train/modal_app.py --task smoke      # load-only VRAM check
modal run train/modal_app.py --task train_dry  # 2000 rows, no GPU train
modal run train/modal_app.py --task verify_mask
modal run train/modal_app.py --task probe      # 2 train steps, peak VRAM
modal run train/modal_app.py --task train      # full SFT → /vol/out/e4b_broad/

# After train — eval on Modal (sequential, not parallel with train)
modal run train/modal_app.py --task infer --manifest data/processed/vfig_bench/id_manifest.jsonl --out /vol/out/generations/base_0pct_vfig_prompt.jsonl --protocol prompt
modal run train/modal_app.py --task sweep --protocol prompt   # all checkpoints × 3 benches

# Modal broad data pipeline (separate app; already done)
modal run data/scripts/modal_broad_app.py --stage all --pilot
```

Configs: `configs/train_e2b_qlora_smoke.yaml`, `configs/train_e4b_broad.yaml`.

Modal secret: `huggingface-secret` with `HF_TOKEN`. Volumes: HF cache, `structsvg-outputs`.  
Training entrypoint: **`train/modal_app.py`** (not `data/scripts/modal_broad_app.py`, which is the data pipeline).

**GPU:** **L4** only (24GB). Probe at ~21 s/step; full run ~750 steps ≈ 4–5 h. Adapters saved to volume `structsvg-outputs` under `e4b_broad/checkpoint_pct_XXX/`.

**Eval timing:** Training and eval are **separate**. `--task train` only writes LoRA adapters. Run `--task infer` per checkpoint/bench, or `--task sweep` after train to generate + score all checkpoints on VFIG ID/OOD + SVG-Diagrams test sequentially.

## Training conditions (v0)

| Run | Config | Manifest | N |
|-----|--------|----------|---|
| Base (0%) | — | — | no training; eval only |
| Broad SFT | `train_e4b_broad.yaml` | `data/processed/svg_diagrams/train_manifest.jsonl` | 2k |

Model: **`google/gemma-4-E4B`** — the **base** (pretrained) checkpoint, **not** `google/gemma-4-E4B-it`.

**Not in v0:** RL/GRPO, training on VFIG-Data before broad curves exist, full SVG-Stack.

## Pre→Post: one model vs several

The workshop asks what **post-training** changes relative to **pretraining**. That is answered primarily **within one architecture**:

- **Checkpoint 0%** = pretrained base on the same task + metrics (with and without SVG-prefix scaffold).
- **Checkpoints 5–100%** = what one SFT stage adds, and **when** (validity vs structure scores on VFIG-Bench).

| Tier | Models | Purpose |
|------|--------|---------|
| **v0 must** | `google/gemma-4-E4B` **base** | Dense checkpoint curves; broad SFT; VFIG-Bench eval |
| **v1 ablation** | Same base, optional 2nd SFT on VFIG-Data 2k | Does structure-designed data move structure metrics? |
| **v1 replication** | One other open **base** VLM | Replication of syntax-before-structure timing |
| **Out of scope** | VFIG instruct checkpoints | Already post-aligned |

## Relation to VFIG

- **VFIG** optimizes end-state figure→SVG quality (66k data, curriculum SFT+RL).
- **We** measure emergence during **plain SFT from base** on broad 2k, then score on **held-out VFIG-Bench**.

Do not train on VFIG-Data until broad checkpoint curves are saved and evaluated.

## Gates (stop if)

- `python -m eval.gold_recovery` fails on fixtures
- Broad dry-run cannot load 2000 rows
- Shuffled-image scores ≈ correct-image (model ignoring vision)
- Assistant-only loss mask wrong on first E4B forward pass (Modal `--verify-loss-mask`)
