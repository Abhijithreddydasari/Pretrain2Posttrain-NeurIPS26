# Evaluation

The submitted study combines **free SVG generation** across eight evaluation points with a **65-pair counterfactual discrimination diagnostic** at base, 20%, and final. Results and interpretation are summarized in the [root README](../README.md#results).

## Free-generation protocol

- Fixed seed-42 subsets of **128 examples each** from held-out SVG-Diagrams, VFIG-Bench, and VFIG-Bench-OOD.
- Base/0%, 5%, 10%, 20%, 40%, 60%, 80%, 100%: **3,072 generations**. `final` duplicates 100%.
- Same reconstruction prompt and image for every checkpoint; greedy decoding, 8,192-token context, 4,096-token output budget.
- **Executable validity:** XML parse, static native-SVG profile, drawable content, and successful rendering.
- **End-to-end fidelity:** SSIM and DINOv2 image similarity, with invalid generations scored as zero. Valid-only scores are diagnostic because their example subset changes across checkpoints.
- **Failure diagnostics:** SVG closure, output-limit rate, repetition, and parse failures.
- **Uncertainty:** paired example bootstrap with 4,000 resamples and 95% intervals; exact McNemar tests for base-to-final validity changes.

SVG-Diagrams is held out from the training source. Both VFIG splits are external scientific figures; “VFIG-ID” does not mean in-distribution for this SFT run. VFIG-OOD has reference images but no gold SVG: image similarity remains possible, while gold-SVG length or structural comparisons do not.

Raw and conservatively recovered outputs are scored separately. Recovery closes a well-formed SVG prefix and discards an incomplete tail; recovered scores never replace raw results. Similarity is not a direct topology score. The submitted evaluation does not include VLM-judge scores, entity/relation F1, or FlowGen results.

## Generate and score

First prepare the [benchmark manifests](../data/README.md#evaluation-data) and [training artifacts](../train/README.md). Cloud generation:

```bash
modal run train/modal_app.py --task sweep --backend vllm --gen-only --adapter-root /vol/out/e4b_broad_v2 --max-samples 128 --sample-seed 42 --max-new-tokens 4096 --benches vfig_id,vfig_ood,svg_diagrams --run-name reproduction_eval128
```

The sweep writes cached predictions and frozen subset/coverage manifests and can resume missing IDs. Keep the same subsets across checkpoints. Score a cached file against its matching subset manifest, using your actual downloaded paths:

```bash
python -m eval.run_bench_eval --manifest PATH_TO_SUBSET.jsonl --preds PATH_TO_PREDICTIONS.jsonl --out outputs/metrics/reproduction.json
```

Scoring checks coverage by default; do not use `--allow-missing` for reported comparisons. `--skip-dino` provides syntax and SSIM checks without loading DINOv2. General per-file scoring is separate from the paired statistical analysis used in the manuscript.

## Counterfactual diagnostic

Each source supplies an original SVG, an edited SVG, and their rendered images. Both candidates have the same token count. Length-normalized, teacher-forced SVG likelihood is measured under both images.

**Strict matching** requires both images to prefer their corresponding candidate, with each margin greater than `1e-6`. The secondary interaction **G** adds the two matching-minus-mismatching margins and cancels a fixed candidate-only preference. A positive G does not require both rankings to be correct.

The frozen probe contains **24 text, 12 box-position, 13 connector-endpoint, and 16 combined edits**. White images measure candidate priors; unrelated images test whether successful ranking changes depend on the corresponding image. All pairs were visually inspected before the main scores. Two candidates fail the stricter free-generation profile; the analysis retains them in the frozen probe and reports a profile-eligible sensitivity analysis.

- [build_multiedit_probe.py](build_multiedit_probe.py) — construction of the submitted v3 probe; depends on earlier QA artifacts and a saved tokenizer.
- [score_svg_probe.py](score_svg_probe.py) and [modal_svg_probe.py](../train/modal_svg_probe.py) — likelihood scoring at base, 20%, final.
- [analyze_svg_probe.py](analyze_svg_probe.py) — strict matching, paired changes, Wilson rate intervals, controls, and generation/discrimination joins.
- [build_svg_probe.py](build_svg_probe.py), [finalize_svg_probe.py](finalize_svg_probe.py), and [svg_probe_figures.py](svg_probe_figures.py) — earlier text-only probe stages and shared utilities; not substitutes for the frozen v3 set.

With the frozen v3 candidate directory restored locally and adapters on Modal:

```bash
modal run train/modal_svg_probe.py --pcts 0,20,100 --run-name reproduction_svg_probe_v3
```

The entrypoint mounts `outputs/svg_probe_v3/`. Download the resulting score files before running the analysis below. A newly generated and inspected candidate set constitutes a new diagnostic unless it matches the frozen manifest.

## Reproducing the submitted analysis

The following artifacts exist in the research workspace but are **gitignored and not distributed in this clone**:

- `outputs/analysis/paper_v1/sweep_analysis.json` and `per_example.jsonl` — aggregate and example-level free-generation evidence.
- Frozen subset manifests and cached predictions at the paths in `SWEEPS` inside [analyze_cached_sweep.py](analyze_cached_sweep.py).
- `outputs/svg_probe_v3/` — candidate SVGs/images, `pairs.jsonl`, protocol, and QA records.
- `outputs/svg_probe_v3_results/scores_000.jsonl`, `scores_020.jsonl`, `scores_100.jsonl`, and `run.json` — raw probe scores and run metadata.

**Current release gap:** `analyze_cached_sweep.py` imports `ink_f1` from `outputs.analysis.ink_sanity`, an untracked helper. It also uses workspace-specific input paths. Its full analysis cannot run from a clean clone until the helper, evidence, and matching path layout are supplied. The commands below require that evidence bundle.

Once those dependencies are restored:

```bash
python -m eval.analyze_cached_sweep
python -m eval.analyze_svg_probe --data outputs/svg_probe_v3 --scores outputs/svg_probe_v3_results --out outputs/analysis/svg_probe_v3
```

The first writes the free-generation summary and per-example records; the second joins probe scores against those records and writes probe statistics and figures. [checkpoint_curves.py](checkpoint_curves.py) is a general emergence-statistics utility; `curves_example.json` is illustrative data, not the submitted result.

The committed [paper figures](../paper/neurips2026/figures/) can be inspected without model or data downloads. See [paper/README.md](../paper/README.md) for figure provenance and build notes.
