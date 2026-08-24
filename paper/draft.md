# Paper draft — Pre→Post short paper skeleton

**Title:** When Does SFT Buy SVG Syntax vs Diagram Structure?

**Status:** skeleton for NeurIPS 2026 Workshop *Transitioning from Pre-Training to Post-Training* (4–5 pages). Fill results after checkpoint runs.

---

## Abstract (draft)

We study whether LoRA supervised fine-tuning on a multimodal **base** VLM primarily teaches **SVG syntax** or also induces **compositional diagram structure** for image-to-native-SVG reconstruction. We SFT on a 2k broad-diagram coreset from StarVector, save dense early checkpoints, and evaluate on **VFIG-Bench** (400 gold-SVG ID + 198 image-only OOD) plus secondary SVG-Diagrams perceptual scores. We hypothesize that validity rises early while structure-oriented scores lag—especially on OOD—reflecting what plain broad SFT buys relative to the pretrained base.

## 1. Introduction

Post-training is often credited with “teaching the model to draw structured diagrams.” We pressure-test a LIMA-style claim: SFT mainly sharpens **output format**; **topology and spatial composition** remain bottlenecks. Unlike StarVector-style leaderboard vectorization, our contribution is **when** syntax and structure emerge and **how** data design changes that timing.

## 2. Related work

Brief: LIMA; Chu et al. (SFT vs generalization); Tülu stage attribution; StarVector / SVG-Bench; FlowGen flowchart parsing. Position: mechanics of post-training for structured generation, not SVG SOTA.

## 3. Task and canonical SVG

Image → canonical native SVG (`notes/canonical_svg.md`). Forbidden scripts/rasters; fixed viewBox; bounded size.

## 4. Data

**Broad 2k (SFT A).** From the HF `starvector/svg-diagrams` train pool (~182k), we retain canonical-parseable SVGs, remove exact and perceptual near-duplicates, and exclude examples matching the held-out SVG-Diagrams test split (normalized SHA256). Each diagram is embedded with SigLIP on a rendered raster and concatenated with structural feature vectors (tag counts, tree depth, path complexity, text density). We select 2,000 training examples via mini-batch k-means medoid sampling over ~4k clusters, with additional slots for high-complexity and rare-cluster coverage (Fig. `broad_coreset_coverage`, `broad_bucket_distribution`, `broad_difficulty_hist`).

- **Broad 2k (SFT).** From the HF `starvector/svg-diagrams` train pool (~182k), filtered and coreset-selected (see README).
- **Eval:** VFIG-Bench 400 (primary); VFIG-Bench-OOD 198; SVG-Diagrams test (secondary).
- **Optional follow-up:** 2nd-stage SFT on VFIG-Data 2k coreset (exclude bench IDs).

## 5. Method

Gemma 4 E4B **base** + LoRA/QLoRA. Loss = CE on SVG tokens only. Checkpoints at 0–100% early-heavy. Protocols: fixed prompt and SVG-prefix scaffold.

## 6. Metrics

Validity; entity F1; relation F1; spatial aggregate; DINO (secondary); ID–OOD gap; \(t_{50}/t_{90}\); area(syntax−structure). Controls: correct / shuffled / blank images.

## 7. Results

*[Tables/figures after runs]*

- Checkpoint curves on VFIG-Bench (Fig. 1)
- Base @ 0% vs SFT (Table 1)
- ID vs OOD gaps (Table 2)
- Qualitative failures: valid SVG, wrong topology (Fig. 2)

## 8. Discussion

Interpret emergence lag; limitations of SVG→triplet extraction; base vs IT framing; no RL in this study.

## 9. Conclusion

*[One paragraph]*

## Follow-ups

Mixed SFT; noisy/iPad test; topology-aware RL; VFig/SVGenius.

---

### Figure placeholders

- `paper/figures/checkpoint_curves.png`
- `paper/figures/failure_grid.png`
