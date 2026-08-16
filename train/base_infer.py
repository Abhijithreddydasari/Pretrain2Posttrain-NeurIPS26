"""Base-model zero-shot / scaffolded inference over a manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.data_utils import load_manifest, resolve_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "model_e4b.yaml")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/generations/base_preds.jsonl"))
    ap.add_argument("--protocol", choices=["prompt", "svg_prefix"], default="prompt")
    ap.add_argument("--max-samples", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = load_manifest(args.manifest, args.max_samples)
    prompt = cfg.get("prompt_template", "").strip()
    prefix = cfg.get("svg_prefix_scaffold", "")

    if args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for r in rows:
                pred = prefix if args.protocol == "svg_prefix" else ""
                f.write(json.dumps({"id": r["id"], "pred_text": pred, "protocol": args.protocol, "dry_run": True}) + "\n")
        print(f"dry-run wrote {args.out}")
        return

    import torch

    from train.model_load import load_vlm

    model_id = cfg["model_id"]
    processor, model, loader = load_vlm(
        model_id,
        load_in_4bit=bool(cfg.get("load_in_4bit", False)),
        dtype_name=cfg.get("torch_dtype", "bfloat16"),
        trust_remote_code=True,
    )
    print(f"loaded {model_id} via {loader}")
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gen_cfg = cfg.get("generation", {})
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            image = resolve_image(r)
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
            messages = [{"role": "user", "content": user_content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.protocol == "svg_prefix":
                text = text + prefix
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=gen_cfg.get("max_new_tokens", 2048),
                    do_sample=gen_cfg.get("do_sample", False),
                )
            # decode newly generated portion when possible
            decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
            f.write(json.dumps({"id": r["id"], "pred_text": decoded, "protocol": args.protocol}) + "\n")
            print(f"generated {r['id']}")


if __name__ == "__main__":
    main()
