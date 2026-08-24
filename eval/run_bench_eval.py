"""Score cached generations on VFIG / SVG-Diagrams benches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structsvg_lib.broad_features import vfig_metrics, vfig_code_filter
from structsvg_lib.metrics import aggregate_scores, bootstrap_ci, mean
from structsvg_lib.svg_ops import extract_svg_blob, render_pil, validate_svg
from train.data_utils import _resolve_path


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_gold_svg(row: dict) -> str | None:
    if row.get("svg"):
        return row["svg"]
    if row.get("svg_path"):
        p = _resolve_path(row["svg_path"])
        if p.exists():
            return p.read_text(encoding="utf-8")
    return None


def _ssim(img_a, img_b) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim_fn

        a = np.asarray(img_a.convert("L"), dtype=np.float64)
        b = np.asarray(img_b.convert("L"), dtype=np.float64)
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
        return float(ssim_fn(a, b, data_range=255.0))
    except Exception:
        a = np.asarray(img_a.resize((256, 256)).convert("RGB"), dtype=np.float32) / 255.0
        b = np.asarray(img_b.resize((256, 256)).convert("RGB"), dtype=np.float32) / 255.0
        mse = float(np.mean((a - b) ** 2))
        return float(max(0.0, 1.0 - mse))


def _dino_cosine(img_a, img_b, model_cache: dict) -> float | None:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if "model" not in model_cache:
            name = "facebook/dinov2-small"
            model_cache["processor"] = AutoImageProcessor.from_pretrained(name)
            model_cache["model"] = AutoModel.from_pretrained(name)
            model_cache["model"].eval()
        proc = model_cache["processor"]
        model = model_cache["model"]
        inputs_a = proc(images=img_a, return_tensors="pt")
        inputs_b = proc(images=img_b, return_tensors="pt")
        with torch.no_grad():
            fa = model(**inputs_a).last_hidden_state.mean(dim=1)
            fb = model(**inputs_b).last_hidden_state.mean(dim=1)
            fa = fa / fa.norm(dim=-1, keepdim=True)
            fb = fb / fb.norm(dim=-1, keepdim=True)
            return float((fa * fb).sum().item())
    except Exception:
        return None


def score_pred(pred_text: str, gold_svg: str | None, *, dino_cache: dict) -> dict:
    blob = extract_svg_blob(pred_text) or pred_text
    val = validate_svg(blob, try_render=True)
    out: dict = {
        "parse_ok": val.parse_ok,
        "render_ok": val.render_ok,
        "validity": float(val.ok),
        "n_drawable": val.n_drawable,
    }
    ok_vfig, _, vfig_m = vfig_code_filter(blob)
    out["vfig_clean"] = vfig_m.get("vfig_clean")
    out["vfig_pass"] = float(ok_vfig)

    if gold_svg and val.ok:
        try:
            pred_img = render_pil(blob, size=960)
            gold_img = render_pil(gold_svg, size=960)
            out["ssim"] = _ssim(pred_img, gold_img)
            dino = _dino_cosine(pred_img, gold_img, dino_cache)
            if dino is not None:
                out["dino_cosine"] = dino
        except Exception as e:  # noqa: BLE001
            out["metric_error"] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/metrics/bench_eval.json"))
    ap.add_argument("--split", type=str, default="all")
    ap.add_argument("--bench", type=str, default="")
    args = ap.parse_args()

    gold = {r["id"]: r for r in _load_jsonl(args.manifest)}
    preds = {r["id"]: r for r in _load_jsonl(args.preds)}
    dino_cache: dict = {}
    rows = []
    for eid, grow in gold.items():
        if args.split != "all" and grow.get("split") not in {None, args.split}:
            continue
        if args.bench and grow.get("bench") != args.bench:
            continue
        prow = preds.get(eid)
        if not prow:
            continue
        gold_svg = _read_gold_svg(grow)
        sc = score_pred(prow.get("pred_text") or prow.get("svg") or "", gold_svg, dino_cache=dino_cache)
        sc["id"] = eid
        rows.append(sc)

    keys = ["validity", "ssim", "dino_cosine", "vfig_clean", "vfig_pass"]
    agg = {k: mean(r.get(k) for r in rows) for k in keys}
    cis = {k: bootstrap_ci([r[k] for r in rows if r.get(k) is not None]) for k in keys if any(r.get(k) is not None for r in rows)}
    report = {
        "manifest": str(args.manifest),
        "preds": str(args.preds),
        "n": len(rows),
        "aggregate": agg,
        "bootstrap": cis,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
