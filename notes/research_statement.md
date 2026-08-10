# Research Statement

**Working title:** Can Post-Training Induce Structured Educational Drawing? Syntax vs Compositional Spatial Structure under SFT

**Author status:** Solo-author OK. Package doubles as a cold-email research note to a faculty mentor.

**Primary venue:** NeurIPS 2026 Workshop — *Transitioning from Pre-Training to Post-Training*  
https://pretrain2posttrain.github.io/call.html  
(Short paper 4–5 pages; non-archival; deadline **Aug 29, 2026 AoE**; reciprocal reviewer required.)

**Optional venue:** New In ML @ NeurIPS 2026  
https://newinml.github.io/NewInML2026NeurIPS/  
(2–8 pages; non-archival; concurrent submission explicitly allowed; eligibility = no prior top-venue paper.)

---

## One-sentence claim (LIMA-style)

> Post-training mainly teaches educational drawing **syntax / output format**; **compositional spatial / hierarchical structure** is the bottleneck and remains weak unless we explicitly design data and evaluation for it.

---

## Research question

Does LoRA SFT on a multimodal base VLM teach (a) markup/protocol validity and (b) compositional spatial structure for educational diagrams — and **when** does each emerge across training checkpoints?

This is a mechanics / behavior-across-training study, not a SOTA drawing demo.

---

## Hypotheses

| ID | Hypothesis |
|----|------------|
| **H1** | Markup / SVG (or primitive-protocol) **validity rises early** in SFT. |
| **H2** | **Compositional / spatial structure lags** and plateaus under plain SFT. |
| **H3** | A multimodal **base** model already has **partial** diagram capability from pretraining; SFT mostly **sharpens format** more than structure. |

**Falsification:** If structure metrics rise as fast as validity under plain SFT (in-domain and on the OOD compositional holdout), H1/H2 are wrong. If the base zero-shot is near floor on both axes and SFT must invent rather than sharpen, H3 is weakened.

---

## Why this fits Pre→Post

CFP axes we hit:

- **Mechanics of post-training** — what does one SFT stage buy for structured generation?
- **Behavior across training** — checkpoint curves for validity vs structure
- **Failure modes / limits** — format competence without compositional competence
- **Evaluation & open science** — protocol, splits, checkpoints, reproducible small study

We are **not** pitching a whiteboard product or a leaderboard win.

---

## Task (standalone; Y is inspiration only)

**Input:** image of an educational diagram / board state (screenshot or rendered diagram).  
**Output:** structured drawing in **one locked format** (prefer native SVG; compact primitive protocol acceptable if SVG proves too noisy).  

Training examples **must include the image**. Text-only SVG SFT is out of scope for the multimodal claim.

Domain: educational diagrams only (trees, linked lists, free-body diagrams, simple graphs, etc.) — not logos, art, or handwriting style transfer.

---

## Scientific style we emulate

- **LIMA** — bold falsifiable claim + minimal intervention
- **Chu et al. (SFT Memorizes, RL Generalizes)** — in-distribution vs OOD composition holdout; memorization vs generalization (**no RL in v0**)
- **Tülu** — stages have jobs; attribute gains to checkpoints (v0 = one stage: LoRA SFT)

---

## Intervention (v0)

```text
base VLM  →  LoRA SFT on educational image→drawing pairs  →  checkpoint eval
```

- Prefer true **base / pretrained** checkpoint (not instruct)
- Primary model: `google/gemma-4-E2B` or `google/gemma-4-E4B` (HF, not Ollama IT tags)
- Fallback: `OpenGVLab/InternVL3-2B-Pretrained` (or 1B)
- Last resort: instruct VLM, framed as **continued post-training**, not induction from pretraining
- **No** full-weight unfreeze for first run
- **No** DPO / GRPO / RL until metrics exist

---

## What success looks like

Scientific observations such as:

> “Validity jumps by mid-SFT; topology/composition metrics stay flat on OOD.”

Not: “We achieve 82% validity.”

---

## Non-goals (v0)

- Whiteboard product UI
- Preference / RL rabbit hole
- 100k internet SVG scrape
- Multi-model bakeoffs / giant models
- Animation, handwriting style transfer, full multimodal agent stack
- Dependence on product Y’s runtime

---

## Timeline (to Aug 29, 2026 AoE)

| Window | Deliverable |
|--------|-------------|
| **Now (mailable)** | This statement + experiment card + repo scaffold |
| **Week of lock** | Output schema + diagram taxonomy + eval harness stubs; confirm HF access + GPU |
| **+1 week** | ~500–1000 filtered image→drawing pairs; ID / OOD splits; base zero-shot baseline |
| **+2 weeks** | One LoRA SFT run; multi-checkpoint eval curves; qualitative failure gallery |
| **Final week** | Short paper draft (4–5 pp Pre→Post); optional New In ML retune; OpenReview submit |

Stretch only if core curves exist: tiny ablation (e.g. format-only vs structure-aware targets) or a second small model family.

---

## Conflict note (chat export vs this lock)

Earlier chat export briefly listed multimodality as out of scope and floated text→SVG. **This prompt wins:** the study is multimodal image→structured educational drawing. See `notes/decisions.md`.
