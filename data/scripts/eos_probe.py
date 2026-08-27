"""Probe: does the supervised completion end with a stop token, and what does
truncation at a given max_length do to the tail?"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.data_utils import PROMPT, build_train_example, load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processor", default="google/gemma-3-4b-it")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--index", type=int, default=0, help="row index to probe")
    ap.add_argument("--longest", action="store_true", help="probe the longest-SVG row instead")
    args = ap.parse_args()

    from transformers import AutoProcessor
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    proc = AutoProcessor.from_pretrained(args.processor, trust_remote_code=True)
    tok = proc.tokenizer

    rows = load_manifest(ROOT / "data/processed/svg_diagrams/train_manifest.jsonl", 400)
    if args.longest:
        from train.data_utils import longest_rows

        row = longest_rows(rows, 1)[0]
    else:
        row = rows[args.index]
    print(f"probing id={row.get('id')}")
    ex = build_train_example(row, prompt=PROMPT)

    collator = DataCollatorForVisionLanguageModeling(
        processor=proc,
        max_length=args.max_length,
        completion_only_loss=True,
    )
    batch = collator([ex])
    labels = batch["labels"][0]
    sup = labels[labels != -100]
    print(f"seq_len={len(labels)} supervised={len(sup)}")
    tail = tok.decode(sup[-25:], skip_special_tokens=False)
    print("--- supervised TAIL (last 25 tokens, specials shown) ---")
    print(repr(tail))
    print(f"eos_token={tok.eos_token!r} eos_id={tok.eos_token_id}")
    ids = sup.tolist()
    print(f"last 5 ids={ids[-5:]}")
    print(f"contains eos_id: {tok.eos_token_id in ids}")
    for name in ("<end_of_turn>", "<eos>"):
        try:
            tid = tok.convert_tokens_to_ids(name)
            print(f"{name} id={tid} present={tid in ids}")
        except Exception:
            pass
    print(f"tail has '</svg>': {'</svg>' in tok.decode(sup[-60:], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
