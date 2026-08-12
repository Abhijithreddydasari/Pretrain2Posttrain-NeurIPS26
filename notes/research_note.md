# Research Note — Structured SVG under Post-Training

**For:** faculty feedback / research internship inquiry  
**Venue target:** NeurIPS 2026 Workshop *Transitioning from Pre-Training to Post-Training* (deadline Aug 29, 2026 AoE)  
**Optional:** New In ML @ NeurIPS 2026 (non-archival; concurrent OK)

---

## Claim

Post-training mainly teaches diagram **SVG syntax / format**; **compositional topology and spatial structure** remain weak unless data and evaluation are designed for them.

## Question

For multimodal **base** VLMs, when does LoRA SFT induce (a) valid native SVG and (b) recoverable diagram structure on image→SVG reconstruction?

## Design (minimal)

| Piece | Choice |
|-------|--------|
| Model | Gemma 4 E4B **base** → LoRA/QLoRA SFT |
| Conditions | Base · 2k broad public diagrams · 2k structure-designed (**StructSVG**) |
| Domains | Workflows + geometry constructions |
| Eval | Validity; entity/relation F1; spatial aggregate; ID vs compositional OOD; dense checkpoints (0–100%, early-heavy) |
| External | FlowGen topology; SVG-Diagrams perceptual (secondary) |

**Falsify:** structure rises as fast as validity on ID and OOD under plain/broad SFT.

## Why Pre→Post

Checkpoint attribution, failure modes (valid but wrong topology), and open eval — not a drawing demo or tutor product.

## Ask

Blunt feedback on whether the question and matched-data contrast are scoped correctly. If the direction resonates, interest in a short research internship / remote collaboration before the workshop deadline.

---

## Email draft (customize)

```text
Subject: Research note — post-training for diagram SVG structure (NeurIPS'26 Pre→Post)

Hi Prof. [Name] —

I'm [Name], [one-line affiliation]. I'm preparing a short empirical paper for the
NeurIPS 2026 "Pre-Training to Post-Training" workshop on whether LoRA SFT induces
SVG syntax vs compositional diagram structure (checkpoint + OOD analysis; base VLM).

Attached is a 2-page note with hypotheses, eval design, and a 3-week plan.
I'd value blunt feedback on whether the question is interesting / scoped right.
If the direction resonates, I'm also looking for a short research internship
or remote collaboration (post-training / VLM structured generation / CV).

Happy to jump on a 15-min call.
[GitHub / PDF link]
```
