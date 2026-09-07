"""Generate correct/shuffled/blank image-control predictions on a fixed subset."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.sweep_checkpoints import BENCHES
from train.infer_engine import load_bench_manifests
from train.vllm_infer import VllmInferEngine


def _control_pairs(pairs, mode: str, seed: int):
    if mode == "correct":
        return pairs
    if mode == "blank":
        return [(row, Image.new("RGB", image.size, "white")) for row, image in pairs]
    if mode != "shuffled":
        raise ValueError(mode)
    if len(pairs) < 2:
        raise ValueError("shuffled control needs at least two examples")
    rng = random.Random(seed)
    offset = rng.randrange(1, len(pairs))
    images = [image for _, image in pairs]
    return [(row, images[(i + offset) % len(images)]) for i, (row, _) in enumerate(pairs)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "model_e4b.yaml")
    ap.add_argument("--bench", choices=sorted(BENCHES), default="vfig_id")
    ap.add_argument("--pcts", default="0,20,80,100")
    ap.add_argument("--modes", default="shuffled,blank")
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data_root = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
    bench_map = {args.bench: BENCHES[args.bench]}
    loaded = load_bench_manifests(
        bench_map,
        data_root=data_root,
        repo_root=ROOT,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
        require_loadable_image=True,
    )
    rows, manifest, split = loaded[args.bench]
    engine = VllmInferEngine(
        args.config,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    correct_pairs = engine.preload_rows(rows, log_prefix=f"{args.bench} controls ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    subset = args.out_dir / f"subset_{args.bench}_{len(rows)}_seed{args.sample_seed}.jsonl"
    with subset.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    for pct in [int(x) for x in args.pcts.split(",") if x.strip()]:
        adapter = None if pct == 0 else args.adapter_root / f"checkpoint_pct_{pct:03d}"
        tag = "base_0pct" if pct == 0 else f"pct_{pct:03d}"
        engine.set_adapter(adapter, tag=tag)
        for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
            pairs = _control_pairs(correct_pairs, mode, args.sample_seed)
            out = args.out_dir / f"{args.bench}_{tag}_{mode}.jsonl"
            engine.generate_manifest(rows, out=out, protocol="prompt", cached=pairs, resume=args.resume)

    meta = {
        "bench": args.bench,
        "manifest": str(manifest),
        "split": split,
        "n": len(rows),
        "sample_seed": args.sample_seed,
        "pcts": args.pcts,
        "modes": args.modes,
        "max_new_tokens": args.max_new_tokens,
        "subset": str(subset),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
