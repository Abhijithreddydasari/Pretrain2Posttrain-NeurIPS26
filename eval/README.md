# Eval

Primary entrypoints:

```bash
python -m eval.gold_recovery
python -m eval.run_eval --manifest data/processed/structsvg/id_manifest.jsonl --preds outputs/generations/base_id.jsonl
python -m eval.checkpoint_curves --curves outputs/metrics/curves_example.json
```

Modules live partly in `structsvg_lib/` (svg_ops, scene_graph, extract, metrics).
