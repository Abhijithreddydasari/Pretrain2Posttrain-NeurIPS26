# Train

```bash
# Data preview / dry-run (no GPU)
python -m train.lora_sft --config configs/train_e2b_qlora_smoke.yaml --dry-run

# Local 8GB smoke (E2B QLoRA) — requires HF access + CUDA
python -m train.lora_sft --config configs/train_e2b_qlora_smoke.yaml

# Base inference dry-run
python -m train.base_infer --manifest data/fixtures/smoke_manifest.jsonl --dry-run

# Modal
modal setup
# create secret huggingface-secret with HF_TOKEN
modal run train/modal_app.py --task smoke
```

Loss: autoregressive CE on target SVG tokens only (verify masking on first smoke).
