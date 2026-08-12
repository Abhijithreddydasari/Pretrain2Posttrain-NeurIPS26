# Pretrain2Posttrain — Structured SVG under SFT

Standalone NeurIPS 2026 workshop research package.

**Claim:** Post-training mainly teaches diagram **SVG syntax / format**; **compositional structure** is the bottleneck unless data+eval are designed for it.

**Task:** diagram image → **canonical native SVG** reconstruction (not a whiteboard product).

## Venues

1. **Primary:** [Pre-Training to Post-Training](https://pretrain2posttrain.github.io/call.html) — short paper ≈4–5 pages; **Aug 29, 2026 AoE**
2. **Optional:** [New In ML @ NeurIPS 2026](https://newinml.github.io/NewInML2026NeurIPS/)

## Locked experiment

| Piece | Choice |
|-------|--------|
| Model | `google/gemma-4-E4B` base (local smoke: E2B QLoRA @ 8GB) |
| Conditions | Base · 2k broad diagrams · 2k StructSVG (workflows + geometry) |
| Eval | Validity, entity/relation F1, spatial aggregate, ID/OOD, dense checkpoints |
| External | FlowGen topology; SVG-Diagrams DINO (secondary) |

## Non-goals (v0)

Whiteboard UI, RL, full SVG-Stack training, multi-model bakeoffs, sequential tutor actions.

## Layout

```text
notes/     statement, experiment card, canonical SVG, primer, research note
configs/   model / train / eval YAML
data/      schemas, scripts (broad filter + StructSVG generators)
train/     LoRA SFT + Modal entrypoints
eval/      validity, structure, metrics, runners
paper/     draft + figures
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**You must:** accept Gemma 4 E4B **base** license on Hugging Face; set `HF_TOKEN`; create Modal secret `huggingface-secret`.

## How to run

```bash
# 1) Gold-recovery / unit checks
python -m eval.gold_recovery

# 2) Generate StructSVG pilot
python -m data.scripts.generate_structsvg --pilot

# 3) Filter broad candidates (streams HF; no full manual browse)
python -m data.scripts.filter_broad_svg --pilot

# 4) Local overfit smoke (8GB → E2B QLoRA)
python -m train.lora_sft --config configs/train_e2b_qlora_smoke.yaml

# 5) Modal E4B (after smoke OK)
modal run train/modal_app.py::smoke_e4b
```

See `notes/research_statement.md`, `notes/experiment_card.md`, `notes/primer.md`.

## Status

Scientific contract **locked**. Eval harness + StructSVG generators + train/Modal stubs are in-repo.

**Gates passed locally:** gold recovery, unit tests, train dry-run, StructSVG pilot manifests.

**You still need:** HF Gemma 4 base access, Modal secret, then overfit → matched E4B runs (see `notes/setup_checklist.md`).
