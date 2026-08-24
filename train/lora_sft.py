"""LoRA / QLoRA SFT entrypoint for Gemma 4 image→SVG."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.checkpoint_utils import (
    CheckpointPctCallback,
    SampleProgressCallback,
    StepTimingCallback,
    TrainMetricsCallback,
    VramPeakCallback,
    estimate_total_steps,
    pct_to_steps,
)
from train.data_utils import (
    PROMPT,
    load_manifest,
    materialize_train_examples,
    resolve_image,
    resolve_svg,
)


def _status(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


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


def _setup_train_env() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


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
    _status(f"loaded {len(rows)} rows from {man}")

    _status("validating sample preview (first 8 rows)...")
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
        _status("dry-run OK")
        return

    _setup_train_env()

    prompt = cfg.get("prompt_template", PROMPT).strip()
    tcfg = cfg["train"]

    # Cache PNG+SVG in RAM *before* loading the 8B model (single disk read; reused all epochs).
    if args.verify_loss_mask:
        _status("building 8 examples for loss-mask gate...")
        train_records = materialize_train_examples(rows[:8], prompt=prompt, log_fn=_status, log_every=8)
        train_dataset = None
    else:
        train_records = materialize_train_examples(rows, prompt=prompt, log_fn=_status, log_every=100)
        from datasets import Dataset

        train_dataset = Dataset.from_list(train_records)
        _status(f"dataset cached in RAM ({len(train_records)} rows; no per-epoch disk reload)")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    from train.model_load import load_vlm

    model_id = cfg["model_id"]
    if cfg.get("use_base_not_it") and model_id.endswith("-it"):
        raise ValueError("Refusing instruct checkpoint while use_base_not_it=true")

    _status(f"loading model {model_id} once (bf16={not cfg.get('load_in_4bit', False)})...")
    processor, model, loader = load_vlm(
        model_id,
        load_in_4bit=bool(cfg.get("load_in_4bit")),
        dtype_name=cfg.get("torch_dtype", "bfloat16"),
        trust_remote_code=cfg.get("trust_remote_code", True),
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    _status(f"model+processor loaded via {loader} (single load for entire run)")

    if cfg.get("load_in_4bit"):
        _status("preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)

    _status("applying LoRA adapters...")
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

    max_length = tcfg.get("max_seq_length", 4096)
    grad_ckpt = bool(tcfg.get("gradient_checkpointing", True))
    num_workers = int(tcfg.get("dataloader_num_workers", 0))
    if torch.cuda.is_available() and num_workers > 0:
        _status("forcing dataloader_num_workers=0 (CUDA init before DataLoader; forked workers crash)")
        num_workers = 0
    _status(
        f"gradient_checkpointing={grad_ckpt} effective_batch="
        f"{tcfg.get('per_device_train_batch_size', 1) * tcfg.get('gradient_accumulation_steps', 8)} "
        f"dataloader_workers={num_workers}"
    )

    if args.verify_loss_mask:
        _status("verifying loss mask on one batch...")
        stats = verify_loss_mask(processor, train_records, max_length=max_length)
        _print_loss_mask_stats(stats)
        if stats["supervised_tokens"] <= 0:
            raise SystemExit("loss mask gate FAILED: no supervised tokens")
        _status("loss mask gate OK")
        return

    total_steps = estimate_total_steps(
        len(train_dataset),
        num_train_epochs=float(tcfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 8)),
    )
    checkpoints_pct = cfg.get("checkpoints_pct", [])
    step_map = pct_to_steps(checkpoints_pct, total_steps)
    _status(f"estimated total_steps={total_steps} epochs={tcfg.get('num_train_epochs', 1)}")
    _status(f"checkpoint schedule: {step_map}")

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
        gradient_checkpointing=grad_ckpt,
        dataloader_pin_memory=tcfg.get("dataloader_pin_memory", False),
        dataloader_num_workers=num_workers,
        dataloader_persistent_workers=False,
    )

    n_samples = len(train_dataset)
    progress_interval = max(100, n_samples // 20)
    out_dir = Path(tcfg["output_dir"])
    ckpt_cb = CheckpointPctCallback(
        checkpoints_pct=checkpoints_pct,
        total_steps=total_steps,
        output_dir=out_dir,
    )
    metrics_cb = TrainMetricsCallback(out_dir)
    progress_cb = SampleProgressCallback(
        n_samples=n_samples,
        num_epochs=float(tcfg.get("num_train_epochs", 1)),
        interval=progress_interval,
    )
    callbacks: list = [ckpt_cb, metrics_cb, progress_cb]
    if args.max_steps is not None:
        callbacks.extend([VramPeakCallback(log_first_n_steps=3), StepTimingCallback()])

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        processing_class=processor,
        callbacks=callbacks,
    )
    ckpt_cb.bind_trainer(trainer)
    metrics_cb.bind_trainer(trainer)

    meta = {
        "checkpoints_pct": checkpoints_pct,
        "total_steps": total_steps,
        "step_map": {str(k): v for k, v in step_map.items()},
        "manifest": str(cfg["data"]["manifest"]),
        "n_samples": len(train_dataset),
        "load_in_4bit": bool(cfg.get("load_in_4bit")),
        "gradient_checkpointing": grad_ckpt,
        "per_device_train_batch_size": int(tcfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(tcfg.get("gradient_accumulation_steps", 8)),
        "effective_batch": int(tcfg.get("per_device_train_batch_size", 1)) * int(tcfg.get("gradient_accumulation_steps", 8)),
    }
    (out_dir / "checkpoints_pct.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _status(f"metrics → {out_dir}/train_log.jsonl (streaming) + train_log.json (final)")

    _status("starting trainer.train() — first step may take 1–2 min (cuda warmup)...")
    if 0 in step_map:
        _status("checkpoint 0% will save after step 1 (avoids pre-step VRAM spike)")
    trainer.train()
    final_dir = Path(tcfg["output_dir"]) / "final"
    trainer.save_model(str(final_dir))
    _status(f"training complete; final adapter at {final_dir}")


if __name__ == "__main__":
    main()
