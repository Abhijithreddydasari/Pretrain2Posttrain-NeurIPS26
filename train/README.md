# Training

The submitted experiment is the completed **Broad v2** run: pretrained `google/gemma-4-E4B` with bf16 LoRA on 1,995 image–SVG pairs. Use [train_e4b_broad_v2.yaml](../configs/train_e4b_broad_v2.yaml); the CLI still defaults to the earlier broad config, so pass v2 explicitly.

## Reported configuration

- Frozen bf16 base and bf16 adapters; no 4-bit quantization in the main run.
- LoRA rank 16, alpha 32, dropout 0.05, all linear modules.
- Cross-entropy on target SVG tokens only; image and prompt tokens are masked.
- Two epochs, 500 optimizer steps, learning rate `1e-4`, `adamw_torch`.
- Per-device batch 2 × gradient accumulation 4 = effective batch 8; gradient checkpointing enabled.
- 960 × 960 letterboxed images and an 8,192-token complete-sequence limit.
- 2,000 selected pairs → five overlength examples discarded → **1,995 retained**. Complete targets are filtered rather than truncated.
- Base/0% plus adapters at steps **25, 50, 100, 200, 300, 400, 500**, corresponding to **5, 10, 20, 40, 60, 80, 100%**. `final` duplicates 100%.

The submitted run uses one broad SFT condition. Synthetic StructSVG, VFIG training, curriculum, RL, and additional model families are outside its results. E2B QLoRA is a separate local smoke configuration.

## Setup

Install the [root requirements](../requirements.txt), configure Modal authentication, and accept the Gemma model's Hugging Face access terms. Set `HF_TOKEN` locally when loading the model and provide it through the Modal secret **`huggingface-secret`** for remote jobs.

Modal uses these volumes:

- `structsvg-hf-cache` at `/vol/hf` — model cache.
- `structsvg-data` at `/vol/data` — prepared training pairs.
- `structsvg-outputs` at `/vol/out` — adapters, logs, generations, and metrics.

Train/probe jobs use **A100-80GB**; the bf16 load-only smoke uses **L4**. See [modal_app.py](modal_app.py) for the image definitions. Dependencies are version ranges, so rebuilding the image is not an exact recreation of the original environment.

## Prepare and validate

First follow the [data instructions](../data/README.md) to populate `data/processed/svg_diagrams/` with the manifest, PNGs, and SVGs.

```bash
python -m pytest -q
python -m train.lora_sft --config configs/train_e4b_broad_v2.yaml --dry-run
python -m data.scripts.upload_broad_modal
modal run train/modal_app.py --task upload_status
modal run train/modal_app.py --task smoke
modal run train/modal_app.py --task train_dry --config train_e4b_broad_v2.yaml
modal run train/modal_app.py --task verify_mask --config train_e4b_broad_v2.yaml
```

The upload helper writes the local broad directory to the data volume. The remote loader resolves supported volume layouts. Check upload status before training.

The dry-run loads the manifest and previews only the first eight pairs; it does not run the full processor/token gate. The loss-mask check loads the model and verifies supervision on a batch. Neither replaces inspection of the completed run's token-budget report.

## Run SFT

These commands launch cloud GPU work. Choose a new `train.output_dir` in a copied config when running a new experiment, so the completed run remains identifiable.

```bash
modal run train/modal_app.py --task probe --config train_e4b_broad_v2.yaml
modal run train/modal_app.py --task train --config train_e4b_broad_v2.yaml
```

With the checked-in v2 configuration, adapters and metadata are written under `/vol/out/e4b_broad_v2/`. Training and checkpoint evaluation are separate entrypoints; launching training does not produce the paper's evaluation curves.

Retain `checkpoint_manifest.json`, `token_budget_report.json`, `train_log.jsonl`, `train_log.json`, `trainer_state.json`, the saved tokenizer/processor, and every scheduled adapter. The checkpoint manifest maps training percentages to actual optimizer steps and adapter paths.

## Evaluate the trajectory

Prepare the benchmark manifests before launching inference; see [evaluation](../eval/README.md). The submitted protocol uses base plus seven checkpoints, seed 42, 128 examples per benchmark, an 8,192-token context, and a 4,096-token output budget.

```bash
modal run train/modal_app.py --task vllm_smoke --adapter-root /vol/out/e4b_broad_v2 --run-name reproduction_smoke
modal run train/modal_app.py --task sweep --backend vllm --gen-only --adapter-root /vol/out/e4b_broad_v2 --max-samples 128 --sample-seed 42 --max-new-tokens 4096 --benches vfig_id,vfig_ood,svg_diagrams --run-name reproduction_eval128
```

The smoke uses a short output cap to validate loading and generation; its predictions are not paper results. Retain the sweep's subset and coverage manifests. The 0% evaluation uses the unchanged base; `checkpoint_pct_000` is separately saved with initialized LoRA.

**Known protocol limitation:** training used the chat-template turn terminator, while generation used the tokenizer's declared end-of-sequence token. Preserve that distinction when reproducing the submitted results; a stopping-token change is a new ablation. [eos_protocol_audit.py](../eval/eos_protocol_audit.py) contains the audit.
