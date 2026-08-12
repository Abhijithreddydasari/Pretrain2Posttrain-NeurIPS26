# Decisions log (living)

Append-only. Newest first. If chat export conflicts with the user lock prompt, **the lock prompt wins**.

---

## 2026-08-12 — Structured SVG SFT lock (plan accepted)

### Locked

| Decision | Choice |
|----------|--------|
| Task | Diagram **image → canonical native SVG reconstruction** |
| Output | Native SVG only (`notes/canonical_svg.md`); no primitive protocol |
| Claim | SFT mainly buys SVG **syntax/format**; **structure** lags unless data+eval designed for it |
| H1–H3 | Validity early; structure lags; base has partial skill |
| Model | `google/gemma-4-E4B` **base**; local smoke `E2B` QLoRA @ 8GB |
| Train | LoRA/QLoRA SFT; no full FT; no RL in v0 |
| Conditions | Base · **2k broad** · **2k StructSVG** (matched token budget) |
| StructSVG grammars | **Workflows** + **geometry constructions** |
| Primary bench | StructSVG ID + compositional OOD |
| External topology | FlowGen |
| Secondary | SVG-Diagrams perceptual (DINO) |
| Checkpoints | 0, 5, 10, 20, 40, 60, 80, 100% |
| Deferred | Mixed SFT, RL, VFig/SVGenius, sequential tutor, iPad train |

### Process

- Eval harness and gold-recovery before full data gen / training spend.
- Append to this log only when the user explicitly locks a choice.

---

## 2026-08-09 — Initial locks from user brief + chat export

### Locked (historical)

| Decision | Choice |
|----------|--------|
| Claim | Post-training mainly teaches educational drawing **syntax/format**; compositional spatial/hierarchical structure is the bottleneck |
| Model primary | Gemma 4 **BASE** (HF), not Ollama IT |
| Train v0 | LoRA SFT only; no RL |
| Task modality | **Image in → structured drawing out** |
| Venues | Pre→Post primary; New In ML optional |

### Conflict resolved

- Early chat listed multimodality out of scope / text→SVG.  
- **Resolution:** multimodal image→SVG is locked.
