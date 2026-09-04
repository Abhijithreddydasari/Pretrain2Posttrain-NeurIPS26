# Research Statement

**Working title:** What Does Post-Training Teach a Vision Model About Long SVG Generation?

**Author status:** Solo-author OK. Package doubles as a cold-email research note to a faculty mentor.

**Primary venue:** NeurIPS 2026 Workshop — *Transitioning from Pre-Training to Post-Training*  
https://pretrain2posttrain.github.io/call.html  
(Short paper 4–5 pages; non-archival; deadline **Aug 29, 2026 AoE**; reciprocal reviewer required.)

**Optional venue:** New In ML @ NeurIPS 2026  
https://newinml.github.io/NewInML2026NeurIPS/  
(2–8 pages; non-archival; concurrent submission explicitly allowed; eligibility = no prior top-venue paper.)

---

## One-sentence claim (LIMA-style)

> Across a dense SFT checkpoint trajectory, we separate learning to emit and terminate SVG from learning to reconstruct the input diagram faithfully.

---

## Research question

Does LoRA SFT on a multimodal **base** VLM teach (a) SVG opening, closure, and renderability and (b) faithful image reconstruction—and when does each change across training checkpoints?

This is a mechanics / behavior-across-training study, not an SVG SOTA demo.

---

## Hypotheses

| ID | Hypothesis |
|----|------------|
| **H1** | SVG opening/closure and parse/render validity improve earlier than visual fidelity. |
| **H2** | Long-output failure—especially repetition and length-limit termination—remains a bottleneck even after two-epoch SFT. |
| **H3** | The base-to-SFT trajectory distinguishes pre-trained visual knowledge from post-trained output-format compliance. |

**Falsification:** H1 is unsupported if all-example fidelity rises at the same normalized rate as validity. H2 is unsupported if length-limit and non-closure rates vanish. H3 is descriptive rather than causal because the pretraining mixture is unknown.

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
| Train | Completed two-epoch, 8192-token bf16 LoRA SFT on the Broad 2k coreset; no RL |
| Conditions | Base/0% and Broad-SFT checkpoints at 5, 10, 20, 40, 60, 80, 100% |
| Primary bench | VFIG-Bench ID (gold SVG) |
| Generalization | VFIG-Bench OOD syntax/termination probe (image-only) |
| Secondary | SVG-Diagrams test with gold SVG |
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

1. SVG opening rate, closing rate, and length-limit termination rate
2. Raw parse/render validity
3. SSIM and DINO over **all** gold examples, with invalid generations scored as zero
4. SSIM and DINO conditional on raw validity (diagnostic only)
5. Conservative recovered-prefix validity/fidelity (separately labelled secondary analysis)
6. Emergence times `t50` / `t90` of base→final gain

Bootstrap over examples for CIs. Exact string match is **not** a primary metric.

---

## Non-goals (v0)

- Whiteboard product UI / sequential draw trajectories  
- DPO / GRPO / RL (until SFT curves exist)  
- Training on full SVG-Stack / 100k+ scrapes  
- Multi-model bakeoffs  
- Synthetic structure data, mixed SFT, VFIG training, real-iPad train (follow-ups)

---

## Deadline execution

| Window | Deliverable |
|--------|-------------|
| Overnight Sep 4 | vLLM smoke, then 128-example × 3-bench × 8-checkpoint generation |
| Sep 4 | Score cached outputs, bootstrap CIs, inspect failure gallery |
| Sep 4–5 | Write compact checkpoint-trajectory paper and submit |

---

## Success looks like

> A complete, falsifiable checkpoint curve showing whether broad SFT improves SVG syntax, termination, and image fidelity together or at different rates—including a negative result if validity remains low.

Not: “We achieve 82% validity.”
