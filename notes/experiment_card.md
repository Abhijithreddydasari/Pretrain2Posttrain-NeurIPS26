# Experiment Card (deadline v0 — locked 2026-09-04)

Status: Broad v2 SFT is complete; generation/evaluation is the critical path.

---

## 1. Claim under test

Measure whether broad LoRA SFT changes SVG syntax/termination before it changes faithful image reconstruction, without assuming that either improvement must occur.

---

## 2. Protocol

| Axis | Choice |
|------|--------|
| Task | Diagram **image → canonical native SVG** |
| Model | `google/gemma-4-E4B` base; smoke: `E2B` QLoRA @ 8GB local |
| Train | Two-epoch bf16 LoRA SFT; assistant/target-SVG tokens only in loss; 8192-token sequences |
| Stages | Single stage SFT (no DPO/GRPO/RL) |
| Conditions | Base/0% · Broad 2k checkpoints at 5/10/20/40/60/80/100% |
| Output | Canonical SVG (`notes/canonical_svg.md`) |
| Eval | VFIG ID/OOD and SVG-Diagrams fixed seeded subsets |

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

### External eval (not primary train)

- **FlowGen** — topology / Strict F1 style after deterministic SVG conversion + triplet extraction  
- **SVG-Diagrams test** — secondary DINO / comparability  

Clean rasters first. Noisy whiteboard + iPad = test-only follow-up.

---

## 4. Metrics (pre-registered)

| # | Metric | Role |
|---|--------|------|
| 1 | Parse / render validity | Syntax / H1 |
| 2 | SVG opening, closure, length-limit rates | Termination |
| 3 | SSIM across all gold examples; invalid = 0 | Primary fidelity |
| 4 | DINO across all gold examples; invalid = 0 | Primary perceptual fidelity |
| 5 | Valid-only SSIM/DINO | Diagnostic |
| 6 | Conservative prefix-recovery validity/fidelity | Secondary salvage analysis |
| 7 | `t50` / `t90` emergence | Timing |

Controls: correct image · shuffled image · blank image.

Exact string match: **not** primary.

---

## 5. Falsification map

| Observation | Interpretation |
|-------------|----------------|
| Validity ↑ before all-example fidelity | Supports syntax-before-reconstruction claim |
| Fidelity ↑ with validity | Falsifies timing separation; report |
| Non-closure/repetition persists | Supports long-generation bottleneck |
| All metrics stay near floor | Negative result: broad SFT did not teach reliable SVG reconstruction |

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
- Base-to-SFT checkpoint comparison
- Qualitative failure gallery  
- 4–5 page Pre→Post draft + repro package  

## Deferred

Curriculum SFT, RL ordering, multi-model replication, VFIG **training**, sequential tutor actions, iPad train, and multi-seed runs.

### VFIG (eval first, train later)

- **v0:** VFIG-Bench as secondary external eval on Gemma base + SFT checkpoints (no training on `QijiaHe/VFIG-Data`).
- **v1+:** Optional 2k coreset from VFIG-Data-Complex-Diagrams as a third broad-style condition; Shapes-and-Arrows subset for curriculum-like ablation. Dedup vs Broad + SVG-Diagrams test before any train use.

### Multi-model replication (optional)

Workshop fit does **not** require multiple families in v0. The primary pre→post bridge is **Gemma 4 E4B base @ 0%** vs SFT checkpoints on the **same** weights.

If budget allows after the checkpoint curves, add one open base VLM on the same fixed evaluation manifests.
