"""LoRA / QLoRA SFT entrypoint for Gemma 4 image→SVG."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.data_utils import PROMPT, build_messages, load_manifest, resolve_image, resolve_svg


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="validate data only")
    args = ap.parse_args()
    cfg = load_config(args.config)

    man = Path(cfg["data"]["manifest"])
    if not man.exists():
        # fallback fixtures smoke
        alt = ROOT / "data" / "fixtures" / "smoke_manifest.jsonl"
        if alt.exists():
            print(f"manifest missing; using {alt}")
            man = alt
        else:
            raise FileNotFoundError(man)

    rows = load_manifest(man, cfg["data"].get("max_samples"))
    print(f"loaded {len(rows)} rows from {man}")

    # dry-run path always works without GPU
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

    # Heavy imports after dry-run gate
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainingArguments
    from trl import SFTTrainer

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
    target = lcfg.get("target_modules", "all-linear")
    peft_cfg = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg.get("lora_dropout", 0.05),
        target_modules=target,
        bias=lcfg.get("bias", "none"),
        task_type=lcfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    prompt = cfg.get("prompt_template", PROMPT).strip()

    def to_record(row):
        img = resolve_image(row)
        svg = resolve_svg(row)
        messages = build_messages(img, svg, prompt=prompt)
        # Chat template text — image handling depends on processor version
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"messages": messages, "text": text, "image": img, "id": row.get("id")}

    train_records = [to_record(r) for r in rows]

    tcfg = cfg["train"]
    training_args = TrainingArguments(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg.get("num_train_epochs", 1),
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(tcfg.get("learning_rate", 1e-4)),
        logging_steps=tcfg.get("logging_steps", 10),
        save_strategy=tcfg.get("save_strategy", "steps"),
        save_steps=tcfg.get("save_steps", 50),
        bf16=tcfg.get("bf16", True),
        remove_unused_columns=tcfg.get("remove_unused_columns", False),
        report_to=tcfg.get("report_to", "none"),
        optim=tcfg.get("optim", "adamw_torch"),
        seed=tcfg.get("seed", 42),
    )

    # Minimal trainer — multimodal collators vary by transformers version.
    # Users should verify loss mask covers assistant SVG tokens only on first smoke.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_records,
        processing_class=processor.tokenizer if hasattr(processor, "tokenizer") else processor,
    )
    # Save checkpoint schedule metadata
    (Path(tcfg["output_dir"]) / "checkpoints_pct.json").write_text(
        json.dumps({"checkpoints_pct": cfg.get("checkpoints_pct", []), "note": "map pct→step after trainer.estimate"}, indent=2),
        encoding="utf-8",
    )
    trainer.train()
    trainer.save_model(tcfg["output_dir"] + "/final")
    print("training complete")


if __name__ == "__main__":
    main()
