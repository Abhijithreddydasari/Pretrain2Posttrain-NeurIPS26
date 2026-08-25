"""Generate predictions for a manifest (base or LoRA adapter)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.data_utils import load_manifest
from train.infer_engine import InferEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "model_e4b.yaml")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir; omit for base")
    ap.add_argument("--protocol", choices=["prompt", "svg_prefix"], default="prompt")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    rows = load_manifest(args.manifest, args.max_samples)
    engine = InferEngine(args.config)
    tag = args.adapter.name if args.adapter else "base_0pct"
    engine.set_adapter(args.adapter, tag=tag)
    cached = InferEngine.preload_rows(rows)
    n = engine.generate_manifest(rows, out=args.out, protocol=args.protocol, cached=cached)
    print(f"wrote {n} preds → {args.out}")


if __name__ == "__main__":
    main()
