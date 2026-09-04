# Setup checklist (you)

Do these before spending Modal credits on full E4B runs.

## Hugging Face

1. Create/login at https://huggingface.co
2. Accept license for **`google/gemma-4-E4B`** (base, **not** `-it`) and confirm the model page is accessible
3. Optionally also accept `google/gemma-4-E2B` for local 8GB smoke
4. Create a fine-grained token with read access to gated repos
5. Copy `.env.example` → `.env` and set `HF_TOKEN=...` (never commit)

```bash
huggingface-cli login
# or: $env:HF_TOKEN="hf_..."
```

## Modal (broad data pipeline)

1. Same HF secret as train: **`huggingface-secret`** with `HF_TOKEN`
2. Volume **`structsvg-data`** stores pipeline outputs under `broad/` on the volume; download full run to **`data/processed/svg_diagrams/`** locally
3. Pilot on Modal:

```bash
modal run data/scripts/modal_broad_app.py --stage all --pilot
```

4. Full 182k run:

```bash
modal run data/scripts/modal_broad_app.py --stage all
```

5. Re-embed from scratch if pool changed:

```bash
modal run data/scripts/modal_broad_app.py --stage embed --fresh-embed
```

6. Validate outputs (runs on volume):

```bash
modal run data/scripts/modal_broad_app.py --stage check
```

7. Download volume artifacts locally (example):

```bash
modal volume get structsvg-data broad/train_manifest.jsonl data/processed/svg_diagrams/
modal volume get structsvg-data broad/pngs data/processed/svg_diagrams/
modal volume get structsvg-data broad/scan_stats.json data/processed/svg_diagrams/
python -m data.scripts.broad_analyze --out data/processed/svg_diagrams
```

Deps for the Modal image are pinned in `requirements-broad-modal.txt`.

## Modal (E4B train smoke)

1. `pip install modal` then `modal setup`
2. Set a spend ceiling (~$350 credits); profile L4/A10G before A100
3. Smoke:

```bash
modal run train/modal_app.py --task smoke
```

## Local (RTX 5060 8GB)

- Use **E2B + QLoRA** only for overfit/smoke
- Main curves: Modal E4B

## Datasets

```bash
# Broad coreset (Phase A) — see data/README.md
python -m data.scripts.build_test_hashes
python -m data.scripts.broad_scan_pool --pilot
python -m data.scripts.broad_embed --pilot
python -m data.scripts.broad_select_coreset --pilot
python -m data.scripts.broad_visualize --pilot

```

## Gates before full train

1. `python -m pytest -q` → PASS
2. `python -m train.lora_sft --config configs/train_e2b_qlora_smoke.yaml --dry-run`  
3. Local or Modal overfit 16–32 examples  
4. Shuffled/blank image control on a handful of preds  
5. `python -m data.scripts.broad_checks --stage all` (or `--pilot`) → all gates PASS  
6. Spot-check 20–30 PNGs in `data/processed/svg_diagrams/pngs/` are **960×960** and readable  
7. Evaluate all saved checkpoints on a fixed seeded subset.

## Outreach

- Customize `notes/research_note.md` email with your bio + 3–5 faculty  
- Send after claim/metrics approval — do not wait for final curves
