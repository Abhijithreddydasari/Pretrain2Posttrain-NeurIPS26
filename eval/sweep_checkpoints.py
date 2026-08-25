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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-root", type=Path, default=_OUT / "e4b_broad")
    ap.add_argument("--protocol", default="prompt")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--pcts", default="0,5,10,20,40,60,80,100")
    ap.add_argument("--out-dir", type=Path, default=_OUT / "metrics" / "sweep")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "model_e4b.yaml")
    ap.add_argument("--skip-existing", action="store_true", help="skip gen if preds file exists")
    args = ap.parse_args()

    from train.infer_engine import InferEngine, load_bench_manifests

    pcts = [int(x) for x in args.pcts.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves = {"pcts": pcts, "validity": [], "ssim": [], "dino_cosine": [], "by_bench": {}}

    benches = load_bench_manifests(BENCHES, data_root=_DATA, repo_root=ROOT, max_samples=args.max_samples)
    image_cache = {
        name: InferEngine.preload_rows(rows, log_prefix=f"{name} ")
        for name, (rows, _man, _split) in benches.items()
    }

    engine = InferEngine(args.config)

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

            met_out = args.out_dir / f"{bench_name}_{tag}.json"
            rep = _run_eval(man, gen_out, met_out, split)
            curves["by_bench"][tag][bench_name] = rep["aggregate"]
            if bench_name == "vfig_id":
                pct_validity.append(rep["aggregate"].get("validity"))
                pct_ssim.append(rep["aggregate"].get("ssim"))
                pct_dino.append(rep["aggregate"].get("dino_cosine"))

        curves["validity"].append(mean_or_nan(pct_validity))
        curves["ssim"].append(mean_or_nan(pct_ssim))
        curves["dino_cosine"].append(mean_or_nan(pct_dino))

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
