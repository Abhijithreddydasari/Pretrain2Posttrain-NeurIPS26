# Pretrain2Posttrain — Structured Educational Drawing under SFT

Standalone NeurIPS 2026 workshop research package.

**Claim:** Post-training mainly teaches educational drawing **syntax / format**; **compositional spatial structure** is the bottleneck unless data+eval are designed for it.

This is **not** a whiteboard product repo. Product Y (screenshot → multimodal model → drawing primitives) inspired the task; this repo must stand alone.

## Venues

1. **Primary:** [Transitioning from Pre-Training to Post-Training](https://pretrain2posttrain.github.io/call.html) (NeurIPS 2026) — short paper ≈4–5 pages; deadline **Aug 29, 2026 AoE**
2. **Optional:** [New In ML @ NeurIPS 2026](https://newinml.github.io/NewInML2026NeurIPS/) — 2–8 pages; non-archival; concurrent OK

## Goals

- Empirical study: base VLM → LoRA SFT → checkpoint eval
- Separate **validity/format** from **structural/compositional** competence
- In-domain vs OOD composition holdout
- Open, reproducible small study (not SOTA chasing)

## Non-goals (v0)

- Whiteboard / tutor UI
- RL / DPO / GRPO
- Huge web SVG scrapes
- Multi-model bakeoffs
- Animation or handwriting style transfer
- Depending on product Y runtime

## Repo layout

```text
notes/          research statement, experiment card, decisions log
data/           raw / processed / splits (empty until schema locked)
train/          LoRA SFT scripts (later)
eval/           metrics, rubrics, checkpoint curve tooling (later)
paper/          draft + figures
```

Start with:

- [`notes/research_statement.md`](notes/research_statement.md)
- [`notes/experiment_card.md`](notes/experiment_card.md)
- [`notes/decisions.md`](notes/decisions.md)

## How to run (later)

Training and eval commands will land here after output schema + eval harness are locked. Expected shape:

```text
# 1) build/filter dataset
# 2) zero-shot baseline on base VLM
# 3) LoRA SFT with checkpoint dumps
# 4) eval validity + structure on ID and OOD
```

Until then: **do not** treat missing `train/` scripts as incomplete science — the mailable artifact is the locked claim + protocol.

## Status

Scaffold + research note ready for faculty email. Full training deferred pending schema/taxonomy/HF/GPU confirmation.
