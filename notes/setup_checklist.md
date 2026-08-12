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

## Modal

1. `pip install modal` then `modal setup`
2. Create secret **`huggingface-secret`** with key `HF_TOKEN` (Modal dashboard → Secrets)
3. Set a hard spend ceiling mentally (~$350 credits); prefer L4/A10G profiling before A100
4. Smoke:

```bash
modal run train/modal_app.py --task smoke
```

## Local (RTX 5060 8GB)

- Use **E2B + QLoRA** only for overfit/smoke
- Main curves: Modal E4B

## Datasets

```bash
# StructSVG pilot (no HF needed)
python -m data.scripts.generate_structsvg --pilot

# Broad filter pilot (needs `datasets` + network)
python -m data.scripts.filter_broad_svg --pilot

# Optional: test-hash dedup file
python -m data.scripts.build_test_hashes
```

## Gates before full train

1. `python -m eval.gold_recovery` → PASS  
2. `python -m train.lora_sft --config configs/train_e2b_qlora_smoke.yaml --dry-run`  
3. Local or Modal overfit 16–32 examples  
4. Shuffled/blank image control on a handful of preds  
5. Then launch matched broad vs StructSVG E4B runs  

## Outreach

- Customize `notes/research_note.md` email with your bio + 3–5 faculty  
- Send after claim/metrics approval — do not wait for final curves
