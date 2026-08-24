# SFT agent handoff prompt

Copy everything below the line into a **new Cursor agent chat**. This agent owns **training only** through broad + StructSVG SFT and checkpoint saves. Eval/VFIG/paper come after.

---

You are the **SFT training agent** for the NeurIPS 2026 workshop repo **Pretrain2Posttrain** (`Pretrain2Posttrain-NeurIPS26` on GitHub).

**Mission:** Get **Gemma 4 E4B base** LoRA SFT working on Modal with **bf16** (not 4-bit), save checkpoints at **0, 5, 10, 20, 40, 60, 80, 100%** of training, run **broad 2k** SFT, then hand off for **VFIG-Bench eval**. Optional **2nd-stage VFIG-Data SFT** is post broad curves — not v0 blocker. Deadline **29 Aug 2026 AoE**.

Read first (in order):
1. `AGENTS.md` (local agent memory — richest operational context)
2. `notes/research_statement.md` + `notes/experiment_card.md`
3. `notes/canonical_svg.md`
4. `train/lora_sft.py`, `train/model_load.py`, `train/data_utils.py`, `train/modal_app.py`
5. `configs/train_e4b_broad.yaml`

Do **not** build a whiteboard app. Do **not** add RL/DPO. Do **not** change eval metrics post-hoc. Do **not** train on VFIG-Data.

---

## Scientific setup (locked)

**Task:** diagram image → canonical native SVG (native SVG only, fixed `viewBox="0 0 512 512"`).

**Claim:** SFT mainly buys SVG **syntax**; **structure** (entity/relation topology) lags unless data+eval are designed for it.

**Model:** `google/gemma-4-E4B` **base** (NOT `-it`). Hugging Face gated — needs `HF_TOKEN`.

**Training method:** **bf16 LoRA SFT** — frozen bf16 base weights + trainable LoRA adapters. **Do NOT use 4-bit QLoRA for the main E4B study** (user locked this 2026-08-22; VFIG appendix uses bf16 for SFT). Local 8GB RTX 5060 may use E2B + QLoRA for smoke/overfit only.

**Loss:** Cross-entropy on **assistant SVG tokens only**. Image + user prompt tokens must be masked (`labels = -100`). If prompt tokens get loss, the experiment is wrong.

**LoRA config (locked):** r=16, alpha=32, dropout=0.05, target_modules=all-linear, task_type=CAUSAL_LM.

**Hyperparams (configs):** lr=1e-4, epochs=3, per_device_batch=1, grad_accum=8, max_seq_length=4096, bf16=true, seed=42.

**Prompt (fixed):**
```
Reconstruct the diagram as a single canonical SVG.
Use viewBox="0 0 512 512". Output only SVG markup.
```

**Checkpoints to save:** 0%, 5%, 10%, 20%, 40%, 60%, 80%, 100% of total training steps (early-heavy for hypothesis H1).

**Two train stages (v0 + optional follow-up):**
1. **Broad 2k** — heterogeneous diagrams (READY) → **v0 must complete**
2. **VFIG-Data 2k** (optional) — structure-designed scientific figures; **only after** broad checkpoint eval on VFIG-Bench; exclude the 400 bench IDs from train

**Base @ 0%:** Run inference on base model **before** SFT for workshop "pretrain vs post-train" table. Two protocols: plain `prompt` and `svg_prefix` scaffold (`base_infer.py`).

---

## Data ready now (broad 2k)

Local path (user's machine):
- Manifest: `data/processed/svg_diagrams/train_manifest.jsonl` (2000 rows)
- Images: `data/processed/svg_diagrams/pngs/` (960×960 letterboxed)
- SVGs: `data/processed/svg_diagrams/svgs/`
- Config: `configs/train_e4b_broad.yaml`

Broad pipeline stats (completed Aug 2026 on Modal):
- 182,144 HF rows scanned → 30,011 pool → **2,000** train
- VFIG filter: Clean=(B+K)/N ≥ 0.40, C ≤ 50
- HF revision: `aacd39c8a8c82b2e5a0f81c10c4cbdc346ff7f0f`

Verify manifest `image_path` / `svg_path` resolve on the machine running train. Run dry-run first.

---

## Eval (after train — not this agent's blocker)

Primary: **VFIG-Bench 400** (gold SVG) + **VFIG-Bench-OOD 198** (image only). Secondary: SVG-Diagrams test, controls. See `eval/README.md`.

---

## Known broken / incomplete train code (you must fix)

The repo has **stubs**, not a working E4B VLM SFT pipeline. Expect to implement:

### 1. Config updates for bf16
In `configs/train_e4b_broad.yaml`:
- Set `load_in_4bit: false`
- Change `optim: adamw_torch` (not `paged_adamw_8bit`)
- Keep `torch_dtype: bfloat16`, `bf16: true`

### 2. Multimodal SFTTrainer (TRL + Transformers 5)
Current `lora_sft.py` passes `processing_class=processor.tokenizer` — likely wrong for VLM.

Fix pattern (see TRL docs / HF cookbook):
- Pass **`processing_class=processor`** (full processor)
- Dataset: conversational format with `image` column OR `messages` with image content
- TRL auto-selects `DataCollatorForVisionLanguageModeling` when processor is passed
- Set `packing=False` for VLMs
- Consider `max_length=None` or high limit to avoid truncating image tokens

Refs:
- https://huggingface.co/docs/trl/main/en/training_vlm_sft
- https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl

### 3. Assistant-only loss mask
Verify on first real batch:
- Print decoded input vs labels
- Prompt/image positions should have label `-100`
- Only SVG assistant tokens contribute to loss

If TRL's `assistant_only_loss` or response template is available for this processor version, use it. Otherwise custom collator.

### 4. Checkpoint scheduler
`checkpoints_pct: [0, 5, 10, 20, 40, 60, 80, 100]` in yaml is not wired.

Implement:
- Compute total steps = (len(dataset) / effective_batch) × epochs
- Map each pct → step index
- Save adapter + training state at those steps
- Save **0%** = base model before train (or base infer output separately)
- Write `checkpoint_manifest.json` with step, pct, path

### 5. Modal production train (`train/modal_app.py`)
Current state:
- `smoke_e4b` loads model (still 4-bit in code — **update to bf16** for consistency)
- `train_remote` is a **placeholder stub**

You must:
- Mount/copy repo into Modal image (pattern from `data/scripts/modal_broad_app.py` — `.env()` before `add_local_dir`, `modal.is_local()` for paths)
- Mount volumes: `/vol/hf`, `/vol/data`, `/vol/out`
- Upload or sync broad data to `/vol/data/...` OR bake paths relative to mounted processed dir
- GPU: **L4** first for bf16 LoRA E4B; if OOM → **A10G** (24GB); A100 last resort
- Secret: **`huggingface-secret`** with `HF_TOKEN`
- Run: `python -m train.lora_sft --config configs/train_e4b_broad.yaml`
- Commit outputs to `structsvg-outputs` volume
- Timeout: 6h+ per run; broad + struct = two jobs

### 6. Transformers 5 Gemma 4 loading (`train/model_load.py`)
- Never pass `load_in_4bit=` to `from_pretrained`
- For bf16: `dtype=torch.bfloat16`, no `quantization_config`
- Skip `prepare_model_for_kbit_training` when not 4-bit
- Tries: `AutoModelForMultimodalLM`, `AutoModelForImageTextToText`, `AutoModelForCausalLM`

---

## Gates — do NOT start long Modal runs until ALL pass

```bash
python -m eval.gold_recovery                                    # must PASS
python -m train.lora_sft --config configs/train_e4b_broad.yaml --dry-run   # 2000 rows, images load
python -m train.lora_sft --config configs/train_e2b_qlora_smoke.yaml       # local overfit 32 ex (optional)
```

Manual checks:
- [ ] One training step runs without OOM on target GPU
- [ ] Loss decreases on 16–32 example overfit
- [ ] Loss mask: assistant tokens only (print labels)
- [ ] One generated SVG after short train is parseable XML (rough validity)
- [ ] Modal smoke loads E4B in **bf16** on L4
- [ ] Shuffled-image base pred ≠ correct-image pred (sanity)

---

## Execution order (deadline-driven)

### Phase A — Fix + smoke (days 1–2)
1. Update configs to bf16 LoRA
2. Fix `lora_sft.py` collator + loss mask
3. Local dry-run + E2B overfit smoke
4. Wire Modal train entrypoint; bf16 smoke on L4
5. Run **base @ 0%** inference on VFIG-Bench subset (`base_infer.py`) — both protocols

### Phase B — Broad SFT (day 3)
1. Sync broad 2k to Modal volume if needed
2. Full broad SFT with checkpoint saves
3. Download adapters to `outputs/e4b_broad/` locally
4. Quick sanity: generate 5 SVGs from 20% and 100% checkpoints

### Phase C — Optional VFIG 2nd-stage SFT (after broad eval)
1. Build 2k VFIG-Data coreset (exclude VFIG-Bench 400 IDs)
2. Sequential SFT from broad 100% adapter **or** fresh base + VFIG-only SFT
3. Re-eval on VFIG-Bench

### Handoff to eval agent
Deliver:
- `checkpoint_manifest.json` per condition
- LoRA adapter dirs at each pct
- `outputs/generations/base_0pct_{prompt,svg_prefix}.jsonl`
- Training logs (loss curve CSV or W&B if enabled)
- Note any OOM/config changes in a short `outputs/TRAIN_LOG.md`

---

## What NOT to do

- Do not use 4-bit for main E4B runs
- Do not switch to `-it` instruct checkpoint
- Do not full fine-tune all weights
- Do not add RL/DPO/GRPO
- Do not train on VFIG-Data
- Do not change checkpoint list or metrics to make results look better
- Do not block on FlowGen or VFIG-Bench — eval agent handles those
- Do not rewrite `notes/decisions.md` without user lock

---

## File reference

| File | Purpose |
|------|---------|
| `train/lora_sft.py` | Main SFT entrypoint — needs VLM fixes |
| `train/model_load.py` | Gemma 4 loader (Transformers 5) |
| `train/data_utils.py` | Manifest, `build_messages`, image/svg resolve |
| `train/base_infer.py` | Base @ 0% generation |
| `train/modal_app.py` | Modal smoke + train (train is stub) |
| `configs/train_e4b_broad.yaml` | Broad 2k train config |
| `configs/train_e4b_structsvg.yaml` | StructSVG 2k train config |
| `configs/train_e2b_qlora_smoke.yaml` | Local 8GB smoke |
| `notes/setup_checklist.md` | HF + Modal setup |
| `notes/canonical_svg.md` | SVG output contract |

---

## Success criteria (training done)

- [ ] Broad 2k bf16 LoRA SFT complete with 8 checkpoint saves + final
- [ ] Base @ 0% preds saved (prompt + svg_prefix)
- [ ] Training loss curves exist
- [ ] At least one checkpoint generates parseable SVG on 10 examples
- [ ] Repro commands documented in `outputs/TRAIN_LOG.md`

**The paper needs checkpoint **timing** curves, not SOTA scores.** Get the saves right.

Start by reading `AGENTS.md` and running dry-run + gold_recovery. Then fix collator/loss mask before any Modal spend.
