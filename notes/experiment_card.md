# Experiment Card (v0 — locked)

Status: **task, metrics, conditions, and grammars locked**. Implement eval before full training.

---

## 1. Claim under test

LoRA SFT mainly buys **SVG syntax/format**; **compositional structure** lags unless data+eval pressure it. Structure-designed data should change checkpoint timing and OOD topology relative to matched broad-diagram SFT.

---

## 2. Protocol

| Axis | Choice |
|------|--------|
| Task | Diagram **image → canonical native SVG** |
| Model | `google/gemma-4-E4B` base; smoke: `E2B` QLoRA @ 8GB local |
| Train | LoRA / QLoRA SFT; assistant/target-SVG tokens only in loss |
| Stages | Single stage SFT (no DPO/GRPO/RL) |
| Conditions | Base · Broad 2k · StructSVG 2k (matched supervised token budget) |
| Output | Canonical SVG (`notes/canonical_svg.md`) |
| Holdout | StructSVG compositional OOD |

### Inference protocols (document both)

1. **Fixed reconstruction prompt** — image + short instruction to emit SVG  
2. **SVG-prefix scaffold** — same prompt, generation starts with `<svg` (helps raw base models)

### Checkpoint schedule

```text
0%, 5%, 10%, 20%, 40%, 60%, 80%, 100%
```

---

## 3. Data

### Broad-SVG (2k train)

- Source pool: HF `starvector/svg-diagrams` train (~182k; pin revision; audit vs paper’s ~472 test definition)
- Filter: parse/render, safety profile, length/element bounds
- Dedup against SVG-Diagrams external test (~474) via normalized SVG hash + perceptual hash
- Stratified sample to 2k under token budget

### StructSVG (2k train / 250 ID / 250 OOD)

| Grammar | Content |
|---------|---------|
| Workflows | Boxes, decisions, directed edges; Graphviz→SVG; scene-graph JSON |
| Geometry | Points, segments, triangles/circles, incidence/containment/relative position |

**Train/ID:** single branch or merge; ≤6 workflow nodes; unnested; single geometry relations.  
**OOD:** branch+merge; nested groups; longer paths; unseen arrangements of seen primitives; (later) clean→noisy render.

### External eval (not primary train)

- **FlowGen** — topology / Strict F1 style after deterministic SVG conversion + triplet extraction  
- **SVG-Diagrams test** — secondary DINO / comparability  

Clean rasters first. Noisy whiteboard + iPad = test-only follow-up.

---

## 4. Metrics (pre-registered)

| # | Metric | Role |
|---|--------|------|
| 1 | Parse / render validity | Syntax / H1 |
| 2 | Typed entity F1 | Structure |
| 3 | Typed relation F1 | Structure |
| 4 | Spatial aggregate (reachability or geometry-relation acc.) | Structure |
| 5 | DINO similarity | Secondary perceptual |
| 6 | ID–OOD gap | Composition / H2 |
| 7 | `t50` / `t90` emergence; area(syntax−structure) | Timing |

Controls: correct image · shuffled image · blank image.

Exact string match: **not** primary.

---

## 5. Falsification map

| Observation | Interpretation |
|-------------|----------------|
| Validity ↑ early; structure flat (esp. OOD) | Supports H1/H2 |
| Structure ↑ with validity on ID **and** OOD | Falsifies H1/H2 |
| Broad ≈ structured on topology | Data-design claim weakened; still report |
| Base already strong on structure | Supports H3; sharpening story |

Do **not** change metrics post-hoc to rescue the hypothesis. Do **not** add RL to rescue v0.

---

## 6. Compute

| Resource | Use |
|----------|-----|
| Local RTX 5060 8GB | E2B QLoRA overfit / smoke |
| Modal (~$350) | E4B main runs; HF cache + adapter volumes |
| Hard ceiling | User-set before full training |

---

## 7. Deliverables

- Checkpoint curves (validity, structure, ID/OOD)  
- Broad vs StructSVG comparison  
- Qualitative failure gallery  
- 4–5 page Pre→Post draft + repro package  

## Deferred

Mixed SFT, RL, VFig/SVGenius **training**, sequential tutor actions, iPad train, multi-seed if budget-bound.

### VFIG (eval first, train later)

- **v0:** VFIG-Bench as secondary external eval on Gemma base + SFT checkpoints (no training on `QijiaHe/VFIG-Data`).
- **v1+:** Optional 2k coreset from VFIG-Data-Complex-Diagrams as a third broad-style condition; Shapes-and-Arrows subset for curriculum-like ablation. Dedup vs Broad + SVG-Diagrams test before any train use.

### Multi-model replication (optional)

Workshop fit does **not** require multiple families in v0. The primary pre→post bridge is **Gemma 4 E4B base @ 0%** vs SFT checkpoints on the **same** weights.

If budget allows after Gemma: add **one** open **base** VLM (not VFIG’s instruct backbones), same manifests and StructSVG eval, sparse checkpoints (e.g. 0/20/60/100%) — tests whether syntax-before-structure timing replicates across pretraining recipes.
