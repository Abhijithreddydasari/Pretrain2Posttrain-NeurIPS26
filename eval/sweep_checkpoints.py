"""Sweep all checkpoints across eval benches; aggregate curves JSON."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_OUT = Path("/vol/out") if Path("/vol/out").exists() else ROOT / "outputs"
_DATA = Path(os.environ.get("DATA_ROOT", ROOT / "data"))


BENCHES = {
    "vfig_id": ("data/processed/vfig_bench/id_manifest.jsonl", "id"),
    "vfig_ood": ("data/processed/vfig_bench/ood_manifest.jsonl", "ood"),
    "svg_diagrams": ("data/processed/svg_diagrams_test/test_manifest.jsonl", "test"),
}


def _run_eval(manifest: Path, preds: Path, out: Path, split: str) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "eval.run_bench_eval",
        "--manifest",
        str(manifest),
        "--preds",
        str(preds),
        "--out",
        str(out),
        "--split",
        split if split != "test" else "all",
    ]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return json.loads(out.read_text(encoding="utf-8"))


def _write_subset_manifest(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-root", type=Path, default=_OUT / "e4b_broad")
    ap.add_argument("--protocol", default="prompt")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--sample-seed", type=int, default=42, help="fixed seed so all checkpoints use same eval IDs")
    ap.add_argument("--pcts", default="0,5,10,20,40,60,80,100")
    ap.add_argument("--out-dir", type=Path, default=_OUT / "metrics" / "sweep")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "model_e4b.yaml")
    ap.add_argument("--skip-existing", action="store_true", help="skip gen if preds file exists")
    ap.add_argument("--gen-only", action="store_true", help="skip per-bench scoring (run eval later)")
    ap.add_argument(
        "--benches",
        default="",
        help="comma-separated bench keys (default: all). e.g. vfig_id or vfig_id,vfig_ood",
    )
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--batch-size", type=int, default=8, help="vLLM batch size (ignored for hf)")
    args = ap.parse_args()

    bench_map = BENCHES
    if args.benches.strip():
        wanted = {x.strip() for x in args.benches.split(",") if x.strip()}
        unknown = wanted - set(BENCHES)
        if unknown:
            raise SystemExit(f"unknown benches: {sorted(unknown)}; valid: {sorted(BENCHES)}")
        bench_map = {k: v for k, v in BENCHES.items() if k in wanted}

    from train.infer_engine import InferEngine, load_bench_manifests

    if args.backend == "vllm":
        from train.vllm_infer import VllmInferEngine

        engine = VllmInferEngine(args.config, batch_size=args.batch_size)
        preload_rows = VllmInferEngine.preload_rows
    else:
        engine = InferEngine(args.config)
        preload_rows = InferEngine.preload_rows
    pcts = [int(x) for x in args.pcts.split(",")]
    print(f"sweep backend={args.backend} adapter_root={args.adapter_root}", flush=True)
    curves = {
        "pcts": pcts,
        "sample_seed": args.sample_seed,
        "max_samples": args.max_samples,
        "validity": [],
        "ssim": [],
        "dino_cosine": [],
        "by_bench": {},
    }

    benches = load_bench_manifests(
        bench_map,
        data_root=_DATA,
        repo_root=ROOT,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
        require_loadable_image=True,
    )
    subset_manifests: dict[str, Path] = {}
    for bench_name, (rows, _man, _split) in benches.items():
        seed_tag = f"seed{args.sample_seed}" if args.sample_seed is not None else "noseed"
        n_tag = args.max_samples if args.max_samples is not None else "all"
        subset_path = args.out_dir / f"eval_subset_{bench_name}_{n_tag}_{seed_tag}.jsonl"
        _write_subset_manifest(rows, subset_path)
        subset_manifests[bench_name] = subset_path
        print(f"wrote eval subset {subset_path} ({len(rows)} rows)", flush=True)
    image_cache = {
        name: preload_rows(rows, log_prefix=f"{name} ")
        for name, (rows, _man, _split) in benches.items()
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pct in pcts:
        if pct == 0:
            adapter = None
            tag = "base_0pct"
        else:
            adapter = args.adapter_root / f"checkpoint_pct_{pct:03d}"
            tag = f"pct_{pct:03d}"
        if adapter is not None and not adapter.exists():
            print(f"skip missing {adapter}", flush=True)
            continue

        engine.set_adapter(adapter, tag=tag)

        pct_validity = []
        pct_ssim = []
        pct_dino = []
        curves["by_bench"][tag] = {}

        for bench_name, (_rows, man, split) in benches.items():
            gen_out = _OUT / "generations" / "sweep" / f"{bench_name}_{tag}_{args.protocol}.jsonl"
            gen_out.parent.mkdir(parents=True, exist_ok=True)
            if args.skip_existing and gen_out.exists() and gen_out.stat().st_size > 0:
                print(f"skip existing {gen_out}", flush=True)
            else:
                rows, _, _ = benches[bench_name]
                engine.generate_manifest(
                    rows,
                    out=gen_out,
                    protocol=args.protocol,
                    cached=image_cache[bench_name],
                )

            if args.gen_only:
                continue

            met_out = args.out_dir / f"{bench_name}_{tag}.json"
            rep = _run_eval(subset_manifests[bench_name], gen_out, met_out, split)
            curves["by_bench"][tag][bench_name] = rep["aggregate"]
            if bench_name == "vfig_id":
                pct_validity.append(rep["aggregate"].get("validity"))
                pct_ssim.append(rep["aggregate"].get("ssim"))
                pct_dino.append(rep["aggregate"].get("dino_cosine"))

        curves["validity"].append(mean_or_nan(pct_validity))
        curves["ssim"].append(mean_or_nan(pct_ssim))
        curves["dino_cosine"].append(mean_or_nan(pct_dino))

    if args.gen_only:
        print("gen-only: skipped scoring and curves aggregation", flush=True)
        return

    curves_path = args.out_dir / f"curves_{args.protocol}.json"
    curves_for_plot = {
        "pcts": pcts,
        "syntax": curves["validity"],
        "structure": curves["ssim"],
        "dino_cosine": curves["dino_cosine"],
        "by_bench": curves["by_bench"],
    }
    curves_path.write_text(json.dumps(curves_for_plot, indent=2), encoding="utf-8")
    print(f"wrote {curves_path}", flush=True)

    cmd = [
        sys.executable,
        "-m",
        "eval.checkpoint_curves",
        "--curves",
        str(curves_path),
        "--out",
        str(args.out_dir / f"emergence_{args.protocol}.json"),
    ]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def mean_or_nan(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


if __name__ == "__main__":
    main()
