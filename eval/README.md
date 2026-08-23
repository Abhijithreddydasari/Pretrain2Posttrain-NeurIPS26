# Eval

Primary entrypoints:

```bash
# Sanity: gold SVG parse/render/extract on fixtures
python -m eval.gold_recovery

# Score cached generations vs a gold manifest (when scene graphs exist)
python -m eval.run_eval --manifest data/processed/vfig_bench/id_manifest.jsonl --preds outputs/generations/base_id.jsonl

# Plot checkpoint curves after per-checkpoint eval JSON exists
python -m eval.checkpoint_curves --curves outputs/metrics/curves_example.json
```

## v0 eval stack (Aug 2026)

| Bench | Split | Gold SVG? | What it measures | Role in paper |
|-------|-------|-----------|------------------|---------------|
| **VFIG-Bench** | 400 ID test | Yes | Validity, render success, pixel sim (SSIM/LPIPS/VisualSim), VFIG component scores (shapes/arrows where applicable), optional VLM-judge (presence/layout/connectivity/details) | **Primary** — hard real scientific figures |
| **VFIG-Bench-OOD** | 198 OOD | No (image only) | Validity, render success, perceptual sim, VLM-judge / qualitative structure | **OOD generalization probe** — no typed F1 |
| **SVG-Diagrams test** | ~474 | Yes | DINO / perceptual comparability to StarVector line | Secondary |
| **Controls** | any | — | Correct vs shuffled vs blank image on same prompts | Vision sanity (H3) |
| **FlowGen** | external | partial | Topology / Strict F1 after triplet extraction | Optional external topology |

**Pre→Post hook:** run every bench at **base @ 0%** and at SFT checkpoints **0, 5, 10, 20, 40, 60, 80, 100%**. Plot *when* validity vs structure metrics move.

Generations are cached once (`outputs/generations/`); metrics can be rescored without re-running the model.

Shared code: `structsvg_lib/` (parse, render, metrics) + `eval/` runners.

## Planned (not wired yet)

- `data/scripts/vfig_to_manifest.py` — download VFIG-Bench, render PNGs, emit eval manifests
- VFIG component + VLM-judge scorers aligned with He et al. 2026
