"""LoRA / QLoRA SFT entrypoint for Gemma 4 image→SVG."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.checkpoint_utils import CheckpointPctCallback, SampleProgressCallback, estimate_total_steps, pct_to_steps
from train.data_utils import PROMPT, build_train_example, load_manifest, resolve_image, resolve_svg


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_manifest(cfg: dict) -> Path:
    man = Path(cfg["data"]["manifest"])
    if man.exists():
        return man
    alt = ROOT / "data" / "fixtures" / "smoke_manifest.jsonl"
    if alt.exists():
        print(f"manifest missing; using {alt}")
        return alt
    raise FileNotFoundError(man)


def verify_loss_mask(processor, examples: list[dict], *, max_length: int | None) -> dict:
    """Print one batch label mask stats for gate 4."""
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    collator = DataCollatorForVisionLanguageModeling(
        processor=processor,
        max_length=max_length,
        completion_only_loss=True,
    )
    batch = collator(examples[:1])
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    n_loss = int((labels != -100).sum().item())
    n_masked = int((labels == -100).sum().item())
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    supervised = labels[labels != -100]
    preview = tok.decode(supervised[: min(80, len(supervised))], skip_special_tokens=False)
    return {
        "seq_len": int(len(labels)),
        "supervised_tokens": n_loss,
        "masked_tokens": n_masked,
        "supervised_preview": preview[:500],
    }


def _print_loss_mask_stats(stats: dict) -> None:
    print(f"seq_len={stats['seq_len']}")
    print(f"supervised_tokens={stats['supervised_tokens']} masked_tokens={stats['masked_tokens']}")
    print("supervised_preview (first SVG tokens):")
    print(stats["supervised_preview"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="validate data only")
    ap.add_argument("--verify-loss-mask", action="store_true", help="load model and print label mask on one batch")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (probe)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    man = _resolve_manifest(cfg)
    rows = load_manifest(man, cfg["data"].get("max_samples"))
    print(f"loaded {len(rows)} rows from {man}")

    samples_meta = []
    for r in rows[: min(8, len(rows))]:
        svg = resolve_svg(r)
        try:
            img = resolve_image(r)
            size = img.size
        except Exception as e:  # noqa: BLE001
            size = None
            print(f"warn image {r.get('id')}: {e}")
        samples_meta.append({"id": r.get("id"), "svg_len": len(svg), "image_size": size})
    out_meta = Path(cfg["train"]["output_dir"])
    out_meta.mkdir(parents=True, exist_ok=True)
    (out_meta / "data_preview.json").write_text(json.dumps(samples_meta, indent=2), encoding="utf-8")

    if args.dry_run:
        print("dry-run OK")
        return

    # UTF-8 required for TRL chat templates on Windows
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    from train.model_load import load_vlm

    model_id = cfg["model_id"]
    if cfg.get("use_base_not_it") and model_id.endswith("-it"):
        raise ValueError("Refusing instruct checkpoint while use_base_not_it=true")

    processor, model, loader = load_vlm(
        model_id,
        load_in_4bit=bool(cfg.get("load_in_4bit")),
        dtype_name=cfg.get("torch_dtype", "bfloat16"),
        trust_remote_code=cfg.get("trust_remote_code", True),
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    print(f"loaded {model_id} via {loader}")

    if cfg.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model)

    lcfg = cfg["lora"]
    peft_cfg = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg.get("lora_dropout", 0.05),
        target_modules=lcfg.get("target_modules", "all-linear"),
        bias=lcfg.get("bias", "none"),
        task_type=lcfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    prompt = cfg.get("prompt_template", PROMPT).strip()
    train_records = None
    if args.verify_loss_mask:
        train_records = [build_train_example(r, prompt=prompt) for r in rows[:8]]
    else:

        def _examples():
            for r in rows:
                yield build_train_example(r, prompt=prompt)

        train_dataset = Dataset.from_generator(_examples)

    tcfg = cfg["train"]
    max_length = tcfg.get("max_seq_length", 4096)

    if args.verify_loss_mask:
        stats = verify_loss_mask(processor, train_records, max_length=max_length)
        _print_loss_mask_stats(stats)
        if stats["supervised_tokens"] <= 0:
            raise SystemExit("loss mask gate FAILED: no supervised tokens")
        print("loss mask gate OK")
        return

    total_steps = estimate_total_steps(
        len(train_dataset),
        num_train_epochs=float(tcfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 8)),
    )
    checkpoints_pct = cfg.get("checkpoints_pct", [])
    step_map = pct_to_steps(checkpoints_pct, total_steps)
    print(f"estimated total_steps={total_steps} checkpoint map={step_map}")

    sft_args = SFTConfig(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg.get("num_train_epochs", 1),
        max_steps=args.max_steps if args.max_steps is not None else -1,
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(tcfg.get("learning_rate", 1e-4)),
        logging_steps=tcfg.get("logging_steps", 10),
        save_strategy="no",
        bf16=tcfg.get("bf16", True),
        remove_unused_columns=tcfg.get("remove_unused_columns", False),
        report_to=tcfg.get("report_to", "none"),
        optim=tcfg.get("optim", "adamw_torch"),
        seed=tcfg.get("seed", 42),
        max_length=max_length,
        packing=False,
        padding_free=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        dataloader_pin_memory=tcfg.get("dataloader_pin_memory", False),
    )

    n_samples = len(train_dataset)
    progress_interval = max(1, n_samples // 20)  # ~20 status lines per epoch on 2k data
    ckpt_cb = CheckpointPctCallback(
        checkpoints_pct=checkpoints_pct,
        total_steps=total_steps,
        output_dir=tcfg["output_dir"],
    )
    progress_cb = SampleProgressCallback(
        n_samples=n_samples,
        num_epochs=float(tcfg.get("num_train_epochs", 1)),
        interval=progress_interval,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        processing_class=processor,
        callbacks=[ckpt_cb, progress_cb],
    )
    ckpt_cb.bind_trainer(trainer)

    meta = {
        "checkpoints_pct": checkpoints_pct,
        "total_steps": total_steps,
        "step_map": {str(k): v for k, v in step_map.items()},
        "manifest": str(cfg["data"]["manifest"]),
        "n_samples": len(train_dataset),
        "load_in_4bit": bool(cfg.get("load_in_4bit")),
    }
    (Path(tcfg["output_dir"]) / "checkpoints_pct.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    trainer.train()
    final_dir = Path(tcfg["output_dir"]) / "final"
    trainer.save_model(str(final_dir))
    print(f"training complete; final adapter at {final_dir}")
    import torch

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"PEAK_VRAM_GB={peak_gb:.2f}", flush=True)


if __name__ == "__main__":
    main()
