# Experiment Card (v0)

Status: **design locked in principle; schema/taxonomy/compute still open** (see questions at end of session notes).  
Do **not** start full training or giant dataset generation until this card + eval schema are confirmed.

---

## 1. Claim under test

Post-training (LoRA SFT) mainly buys **drawing syntax / format**; **compositional spatial structure** lags unless data+eval explicitly pressure it.

---

## 2. Protocol (minimal intervention)

| Axis | Choice |
|------|--------|
| Task | Educational diagram **image → structured drawing** |
| Model | Prefer `google/gemma-4-E2B` or `E4B` **base**; fallback InternVL3-*-Pretrained; last resort instruct (honest framing) |
| Train | **LoRA / adapter SFT only** (no full unfreeze in v0) |
| Stages | Single stage: SFT. No DPO/GRPO/RL in v0 |
| Data size | N ≈ **500–2000** quality pairs (filter: must parse + render) |
| Output | **ONE** format locked before train: native SVG **or** compact primitive protocol (prefer SVG if feasible) |
| Holdout | **OOD compositional** split: unseen combinations of seen primitives/structures |

### Baselines / conditions

1. Base zero-shot (and optional few-shot) on same prompts  
2. Mid-SFT checkpoints (e.g. 25% / 50% / 75% / 100% of steps)  
3. Final LoRA SFT  
4. In-domain vs OOD composition (same metrics both)

---

## 3. Data plan (v0)

**Domain only:** educational diagrams — trees, linked lists, free-body / force diagrams, simple graphs / DAGs, (optional) bar/flow sketches for teaching. **Not** logos, art, icons, handwriting.

**Pair construction (conceptual):**

1. Generate or author a **structured ground-truth** drawing (SVG or primitives) with known topology  
2. Render to PNG (and/or lightly corrupt / screenshot-like) as the **input image**  
3. Train: `(image, optional short text instruction) → target markup/protocol`  
4. Filter: parse fail / render fail → drop  

**Splits:**

| Split | Purpose |
|-------|---------|
| Train | Seen diagram families + seen primitive combos |
| ID val/test | Same families, held-out instances |
| OOD composition | Unseen **combinations** of seen parts (Chu-style memorization probe) |

Exact category list and OOD construction rules: **still open** (block building until locked).

---

## 4. Metrics (define before serious training)

| # | Metric | What it tests |
|---|--------|----------------|
| 1 | **Parse / render validity** | Well-formed markup; renders without crash |
| 2 | **Structural / topology match** | Nodes, edges, hierarchy / connectivity — **not** CLIP alone |
| 3 | **Checkpoint curves** | Validity vs structure vs steps (workshop-critical) |
| 4 | **ID vs OOD composition** | Format learning vs compositional generalization |
| 5 | **Human rubric (~50)** | Teaching usefulness / spatial correctness on a small set |

Optional secondary: CLIP / perceptual similarity — **supporting only**, never the primary structure claim.

**Success narrative:** curves and gap stories (“validity early, structure lags on OOD”), not a single headline %.

---

## 5. Falsification map

| Observation | Interpretation |
|-------------|----------------|
| Validity ↑ early; structure flat (esp. OOD) | Supports H1/H2 |
| Structure ↑ in lockstep with validity on ID **and** OOD | Falsifies H1/H2 as stated |
| Base already strong on structure; SFT only tidies tags | Supports H3 strongly; paper becomes “sharpening” story |
| Base near floor on both; SFT invents format *and* structure | Weakens H3; still useful if checkpoint timing differs |

---

## 6. Compute assumptions (TBD — ask before train)

Assumed for planning (not locked): single GPU or Colab-class box capable of **2B–4B VLM LoRA**. Exact VRAM / local vs Colab: **open**.

---

## 7. Deliverables for Aug 29

- Checkpoint curves (validity, structure, ID/OOD)  
- Qualitative failure gallery (valid but wrong topology; collapsed hierarchy; template regurgitation)  
- Short paper 4–5 pages (Pre→Post) + optional New In ML packaging  
- Repro: data recipe, train config, eval scripts (this repo)

---

## 8. Explicitly deferred

RL/verifiable rewards, multi-model bakeoff, full FT ablation, product Y integration, large web scrapes.
