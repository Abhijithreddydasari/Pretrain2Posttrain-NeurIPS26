# Eval

Primary entrypoints:

```bash
# Score cached generations vs a gold manifest
python -m eval.run_bench_eval --manifest data/processed/vfig_bench/id_manifest.jsonl --preds outputs/generations/base_id.jsonl

# Plot checkpoint curves after per-checkpoint eval JSON exists
python -m eval.checkpoint_curves --curves outputs/metrics/curves_example.json
```

## v0 eval stack (Aug 2026)

| Bench | Split | Gold SVG? | What it measures | Role in paper |
|-------|-------|-----------|------------------|---------------|
| **VFIG-Bench** | 400 ID test | Yes | Raw/recovered validity, closure and length-limit rates, SSIM and DINO | **Primary** — hard real scientific figures |
| **VFIG-Bench-OOD** | 198 OOD | No (image only) | Raw/recovered validity, closure and length-limit rates | **OOD syntax probe** |
| **SVG-Diagrams test** | ~474 | Yes | DINO / perceptual comparability to StarVector line | Secondary |
| **Controls** | any | — | Correct vs shuffled vs blank image on same prompts | Vision sanity (H3) |
| **FlowGen** | external | partial | Topology / Strict F1 after triplet extraction | Optional external topology |

**Pre→Post hook:** run every bench at **base @ 0%** and at SFT checkpoints **0, 5, 10, 20, 40, 60, 80, 100%**. Plot *when* validity vs structure metrics move.

Generations are cached once (`outputs/generations/`); metrics can be rescored without re-running the model.

Use the same fixed seeded subset at every checkpoint. The deadline run uses 128 examples from each bench (3,072 generations across eight distinct checkpoints). Report bootstrap CIs. Similarity over all gold examples, with invalid generations scored as zero, is primary; similarity conditional on validity is diagnostic.

Recovery closes only a deterministic well-formed SVG prefix and discards an incomplete tail. Raw and recovered metrics remain separate.

Shared code: `structsvg_lib/` (legacy name; generic parse/render/metrics only) + `eval/` runners.

## vLLM sweep

```bash
modal run train/modal_app.py --task sweep --backend vllm --gen-only --max-samples 128 \
  --benches vfig_id,vfig_ood,svg_diagrams --run-name eval128_ctx8192
```

The sweep uses an 8192-token model context and batch 64 on A100-80GB, writes a coverage manifest, resumes exact missing IDs, and refuses to score incomplete files. Run the three benches as separate detached Modal jobs when wall time matters; their generation paths must remain distinct.
