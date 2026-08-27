"""Audit target-SVG token lengths + viewBox reality across train/eval manifests.

Answers: what max_seq_length / max_new_tokens do we actually need, and how many
examples would a given budget truncate?
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.data_utils import _resolve_path, load_manifest, resolve_svg

MANIFESTS = {
    "broad_train": "data/processed/svg_diagrams/train_manifest.jsonl",
    "vfig_id": "data/processed/vfig_bench/id_manifest.jsonl",
    "svg_diagrams_test": "data/processed/svg_diagrams_test/test_manifest.jsonl",
}


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return sorted_vals[idx]


def viewbox_of(svg: str):
    m = re.search(r'viewBox="([^"]+)"', svg[:3000])
    if not m:
        return None
    parts = re.split(r"[\s,]+", m.group(1).strip())
    if len(parts) != 4:
        return None
    try:
        return float(parts[2]), float(parts[3])
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="google/gemma-3-4b-it")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--image-tokens", type=int, default=300,
                    help="approx image+prompt+chat-template overhead tokens")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "metrics" / "token_audit.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    report = {"tokenizer": args.tokenizer, "image_prompt_overhead": args.image_tokens, "benches": {}}

    for name, rel in MANIFESTS.items():
        man = ROOT / rel
        if not man.exists():
            print(f"skip missing {man}")
            continue
        rows = load_manifest(man, args.limit)
        tgt_tokens = []
        vb_512 = 0
        vb_seen = 0
        aspects = []
        for r in rows:
            try:
                svg = resolve_svg(r)
            except Exception:
                continue
            n = len(tok.encode(svg, add_special_tokens=False))
            tgt_tokens.append(n)
            vb = viewbox_of(svg)
            if vb:
                vb_seen += 1
                w, h = vb
                if abs(w - 512) < 1 and abs(h - 512) < 1:
                    vb_512 += 1
                if h > 0:
                    aspects.append(w / h)

        tgt_tokens.sort()
        aspects.sort()
        total = [t + args.image_tokens for t in tgt_tokens]

        def trunc_at(budget):
            return sum(1 for t in total if t > budget)

        entry = {
            "n": len(tgt_tokens),
            "target_svg_tokens": {
                "p50": pct(tgt_tokens, 0.50),
                "p90": pct(tgt_tokens, 0.90),
                "p95": pct(tgt_tokens, 0.95),
                "p99": pct(tgt_tokens, 0.99),
                "max": tgt_tokens[-1] if tgt_tokens else 0,
            },
            "total_seq_tokens": {
                "p50": pct(total, 0.50),
                "p90": pct(total, 0.90),
                "p95": pct(total, 0.95),
                "p99": pct(total, 0.99),
                "max": total[-1] if total else 0,
            },
            "truncated_at": {
                str(b): trunc_at(b) for b in (2048, 3072, 4096, 6144, 8192)
            },
            "viewbox": {
                "seen": vb_seen,
                "is_512_square": vb_512,
                "aspect_p10": round(pct(aspects, 0.10), 3) if aspects else None,
                "aspect_p50": round(pct(aspects, 0.50), 3) if aspects else None,
                "aspect_p90": round(pct(aspects, 0.90), 3) if aspects else None,
            },
        }
        report["benches"][name] = entry
        print(f"\n=== {name} (n={entry['n']}) ===")
        print(f"  target SVG tokens: {entry['target_svg_tokens']}")
        print(f"  total seq tokens : {entry['total_seq_tokens']}")
        print(f"  truncated at     : {entry['truncated_at']}")
        print(f"  viewBox 512sq    : {vb_512}/{vb_seen}  aspect p10/p50/p90="
              f"{entry['viewbox']['aspect_p10']}/{entry['viewbox']['aspect_p50']}/{entry['viewbox']['aspect_p90']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
