"""Generate predictions for each checkpoint on eval benches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.data_utils import load_manifest, resolve_image


def generate(
    *,
    manifest: Path,
    out: Path,
    config: Path,
    adapter: Path | None,
    protocol: str,
    max_samples: int | None,
) -> int:
    import torch
    from peft import PeftModel

    from train.model_load import load_vlm

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    rows = load_manifest(manifest, max_samples)
    prompt = cfg.get("prompt_template", "").strip()
    prefix = cfg.get("svg_prefix_scaffold", "")
    gen_cfg = cfg.get("generation", {})

    processor, model, loader = load_vlm(
        cfg["model_id"],
        load_in_4bit=bool(cfg.get("load_in_4bit", False)),
        dtype_name=cfg.get("torch_dtype", "bfloat16"),
        trust_remote_code=True,
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    print(f"checkpoint_infer: {loader} adapter={adapter} n={len(rows)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            image = resolve_image(r)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if protocol == "svg_prefix":
                text = text + prefix
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
            input_len = int(inputs["input_ids"].shape[-1])
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=gen_cfg.get("max_new_tokens", 2048),
                    do_sample=gen_cfg.get("do_sample", False),
                    max_time=gen_cfg.get("max_time", 600),
                )
            decoded = processor.decode(out_ids[0][input_len:], skip_special_tokens=True)
            f.write(json.dumps({"id": r["id"], "pred_text": decoded, "protocol": protocol}) + "\n")
            f.flush()
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "model_e4b.yaml")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir; omit for base")
    ap.add_argument("--protocol", choices=["prompt", "svg_prefix"], default="prompt")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()
    n = generate(
        manifest=args.manifest,
        out=args.out,
        config=args.config,
        adapter=args.adapter,
        protocol=args.protocol,
        max_samples=args.max_samples,
    )
    print(f"wrote {n} preds → {args.out}")


if __name__ == "__main__":
    main()
