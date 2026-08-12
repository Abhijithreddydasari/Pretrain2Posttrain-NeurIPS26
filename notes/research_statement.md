# Research Statement

**Working title:** When Does Post-Training Buy SVG Syntax vs Diagram Structure?

**Author status:** Solo-author OK. Package doubles as a cold-email research note to a faculty mentor.

**Primary venue:** NeurIPS 2026 Workshop — *Transitioning from Pre-Training to Post-Training*  
https://pretrain2posttrain.github.io/call.html  
(Short paper 4–5 pages; non-archival; deadline **Aug 29, 2026 AoE**; reciprocal reviewer required.)

**Optional venue:** New In ML @ NeurIPS 2026  
https://newinml.github.io/NewInML2026NeurIPS/  
(2–8 pages; non-archival; concurrent submission explicitly allowed; eligibility = no prior top-venue paper.)

---

## One-sentence claim (LIMA-style)

> Post-training mainly teaches diagram **SVG syntax / output format**; **compositional spatial / topological structure** remains the bottleneck unless data and evaluation are designed for it.

---

## Research question

Does LoRA SFT on a multimodal **base** VLM teach (a) markup validity and (b) recoverable diagram structure for image→native-SVG reconstruction — and **when** does each emerge across training checkpoints?

This is a mechanics / behavior-across-training study, not an SVG SOTA demo.

---

## Hypotheses

| ID | Hypothesis |
|----|------------|
| **H1** | SVG **parse/render validity rises early** in SFT. |
| **H2** | **Typed structure** (entities/relations) **lags** and plateaus under plain / broad SFT, especially on compositional OOD. |
| **H3** | A multimodal **base** model already has **partial** diagram capability from pretraining; SFT mostly **sharpens format** more than structure. |

**Falsification:** If structure metrics rise at the same normalized rate as validity on ID **and** OOD under plain SFT, H1/H2 are wrong. If base is near floor on both axes and SFT invents rather than sharpens, H3 is weakened.

---

## Locked task

```text
diagram image  →  canonical native SVG reconstruction
```

- Not tutor-like “partial board + instruction → edit” (deferred).
- Training examples **must include the image**.
- Output: **canonical native SVG** only (see `notes/canonical_svg.md`).

---

## Locked experiment

| Axis | Choice |
|------|--------|
| Model | `google/gemma-4-E4B` **base** (HF); local smoke: E2B + QLoRA on 8GB |
| Train | LoRA / QLoRA SFT only; no full unfreeze; no RL in v0 |
| Conditions | (1) base zero-shot (2) **2k broad-diagram** SFT (3) **2k StructSVG** SFT — matched token budget |
| Primary bench | **StructSVG** — workflows + geometry; ID + compositional OOD |
| External topology | **FlowGen** (node–edge triplets / Strict F1 style) |
| Secondary | **SVG-Diagrams** perceptual/comparability (DINO); not the main claim |
| Checkpoints | 0, 5, 10, 20, 40, 60, 80, 100% |

---

## Why this fits Pre→Post

- **Mechanics of post-training** — what one SFT stage buys for structured generation
- **Behavior across training** — dense early checkpoint curves
- **Failure modes** — valid SVG with wrong topology
- **Evaluation & open science** — protocol, splits, reproducible small study

We are **not** pitching a whiteboard product or beating StarVector on SVG-Bench.

---

## Core metrics (pre-registered)

1. Parse / render validity  
2. Typed entity (node) F1  
3. Typed relation (edge) F1  
4. One spatial aggregate (workflow: reachability; geometry: relation accuracy)  
5. DINO similarity (secondary)  
6. ID–OOD gap  
7. Emergence times `t50` / `t90` of base→final gain; area between syntax and structure curves  

Bootstrap over examples for CIs. Exact string match is **not** a primary metric.

---

## Non-goals (v0)

- Whiteboard product UI / sequential draw trajectories  
- DPO / GRPO / RL (until SFT curves exist)  
- Training on full SVG-Stack / 100k+ scrapes  
- Multi-model bakeoffs  
- Mixed SFT, VFig/SVGenius, real-iPad train (follow-ups)

---

## Timeline (to Aug 29, 2026 AoE)

| Window | Deliverable |
|--------|-------------|
| Aug 12–13 | Locked contract, accounts, mailable note, repo setup |
| Aug 14–16 | Eval harness + gold recovery; public filter; StructSVG pilot |
| Aug 17–18 | Final splits; base baselines; overfit gates |
| Aug 19–22 | Matched E4B SFT runs + checkpoint generation |
| Aug 23–25 | Frozen eval, human audit, figures |
| Aug 25–27 | Short paper draft |
| Aug 28–29 | Repro dry-run + OpenReview submit |

---

## Success looks like

> “Validity jumps by mid-SFT; topology stays flat on compositional OOD under broad data; structure-designed data narrows the gap.”

Not: “We achieve 82% validity.”
