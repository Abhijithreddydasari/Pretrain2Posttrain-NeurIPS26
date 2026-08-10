# Decisions log (living)

Append-only. Newest first. If chat export conflicts with the user lock prompt, **the lock prompt wins**.

---

## 2026-08-09 — Initial locks from user brief + chat export

### Locked

| Decision | Choice |
|----------|--------|
| Claim | Post-training mainly teaches educational drawing **syntax/format**; compositional spatial/hierarchical structure is the bottleneck |
| H1 | Validity rises early in SFT |
| H2 | Structure lags / plateaus |
| H3 | Multimodal base has partial capability; SFT sharpens format > structure |
| Falsify | Structure rises as fast as validity under plain SFT → H1/H2 wrong |
| Model primary | `google/gemma-4-E2B` or `google/gemma-4-E4B` **BASE** (HF), not Ollama IT |
| Model fallback | `OpenGVLab/InternVL3-2B-Pretrained` (or 1B) |
| Train v0 | LoRA SFT only; no full unfreeze; no DPO/GRPO/RL |
| Task modality | **Image in → structured drawing out** (must train with images) |
| Domain | Educational diagrams only |
| Data scale | N ≈ 500–2000, quality + render/parse filter |
| Eval | Validity, topology/structure, checkpoint curves, ID vs OOD composition, ~50 human rubrics |
| Venues | Primary Pre→Post; optional New In ML; both non-archival; dual-submit likely OK (verify at submit) |
| Product Y | Inspiration only; no runtime dependency |
| v0 non-goals | Whiteboard UI, RL, 100k scrape, bakeoffs, animation/style transfer |

### Conflict resolved

- Chat export early lock file said multimodality **out of scope** and suggested text→SVG.  
- Later chat + current user prompt require multimodal because the motivating loop is screenshot → model → drawing.  
- **Resolution:** multimodal image→drawing is locked; text-only SVG SFT is not the paper.

### Still open (block building)

1. Exact output schema: native SVG vs compact primitive protocol — **leaning native SVG** (user 2026-08-09)  
2. Exact diagram category taxonomy + OOD composition rule — **leaning single domain first**  
3. HF gated access to Gemma 4 **base** (E4B preferred; Ollama IT ≠ train start)  
4. Compute: Modal ~$350 credits / Kaggle ~$100 / local RTX 5060 — **prefer Modal or local for LoRA; confirm 5060 VRAM**

### Emerging preferences (2026-08-09 evening)

- Task shape: whiteboard-like **image in → SVG out**; target SVG available for supervised loss  
- Model: Gemma 4 **E4B base** via HF weights (not Ollama for training)  
- RL: deferred until SFT checkpoint results exist  
- Data: start **one educational domain**; prefer synthetic controllable GT over iPad-only corpus  
- Near-term non-science goal: cold-email US/EU faculty for workshop feedback and/or research internship (CV / post-training)

### Process

- No full training or giant dataset gen until statement + eval schema confirmed by user.  
- Near-term artifact: professor-mailable note (statement + experiment card), not a trained checkpoint.
