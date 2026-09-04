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
from structsvg_lib.metrics import bootstrap_ci, mean
from structsvg_lib.svg_ops import extract_svg_blob, render_pil, validate_svg
from train.data_utils import _resolve_path
from train.vllm_infer import recover_svg_prefix


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


def _read_gold_image(row: dict):
    image_path = row.get("image_path")
    if not image_path:
        return None
    path = _resolve_path(image_path)
    if not path.exists():
        return None
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _letterbox_image(image, size: int = 960):
    from PIL import Image

    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


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


def _fidelity(
    svg: str,
    gold_svg: str | None,
    gold_image,
    *,
    dino_cache: dict | None,
) -> tuple[float, float | None]:
    pred_img = render_pil(svg, size=960)
    gold_img = render_pil(gold_svg, size=960) if gold_svg else _letterbox_image(gold_image, size=960)
    dino = _dino_cosine(pred_img, gold_img, dino_cache) if dino_cache is not None else None
    return _ssim(pred_img, gold_img), dino


def score_pred(
    pred_text: str,
    gold_svg: str | None,
    gold_image=None,
    *,
    dino_cache: dict | None,
    recovered_svg: str | None = None,
    hit_length_limit: bool = False,
) -> dict:
    blob = extract_svg_blob(pred_text) or pred_text
    val = validate_svg(blob, try_render=True)
    out: dict = {
        "parse_ok": val.parse_ok,
        "render_ok": val.render_ok,
        "validity": float(val.ok),
        "n_drawable": val.n_drawable,
        "svg_open": float("<svg" in pred_text.lower()),
        "svg_close": float("</svg>" in pred_text.lower()),
        "hit_length_limit": float(hit_length_limit),
    }
    ok_vfig, _, vfig_m = vfig_code_filter(blob)
    out["vfig_clean"] = vfig_m.get("vfig_clean")
    out["vfig_pass"] = float(ok_vfig)

    has_gold = bool(gold_svg) or gold_image is not None
    if has_gold and val.ok:
        try:
            out["ssim"], dino = _fidelity(blob, gold_svg, gold_image, dino_cache=dino_cache)
            if dino is not None:
                out["dino_cosine"] = dino
        except Exception as e:  # noqa: BLE001
            out["metric_error"] = str(e)

    # Primary fidelity includes invalid outputs as zero. Conditional metrics are
    # retained to diagnose image reconstruction among only renderable outputs.
    if has_gold:
        out["ssim_all"] = out.get("ssim", 0.0)
        if dino_cache is not None:
            if "dino_cosine" in out:
                out["dino_cosine_all"] = out["dino_cosine"]
            elif not val.ok:
                out["dino_cosine_all"] = 0.0

    recovered_blob = extract_svg_blob(recovered_svg or "") if recovered_svg else None
    recovered_val = validate_svg(recovered_blob, try_render=True) if recovered_blob else val
    out["recovered_validity"] = float(recovered_val.ok)
    out["salvage_gain"] = max(0.0, out["recovered_validity"] - out["validity"])
    if has_gold:
        if recovered_blob and recovered_val.ok and not val.ok:
            try:
                recovered_ssim, recovered_dino = _fidelity(
                    recovered_blob,
                    gold_svg,
                    gold_image,
                    dino_cache=dino_cache,
                )
                out["recovered_ssim_all"] = recovered_ssim
                if recovered_dino is not None:
                    out["recovered_dino_cosine_all"] = recovered_dino
            except Exception as e:  # noqa: BLE001
                out["recovery_metric_error"] = str(e)
        else:
            out["recovered_ssim_all"] = out.get("ssim_all", 0.0)
            if dino_cache is not None and "dino_cosine_all" in out:
                out["recovered_dino_cosine_all"] = out["dino_cosine_all"]
        out.setdefault("recovered_ssim_all", 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/metrics/bench_eval.json"))
    ap.add_argument("--split", type=str, default="all")
    ap.add_argument("--bench", type=str, default="")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--skip-dino", action="store_true", help="score syntax and SSIM without loading DINOv2")
    args = ap.parse_args()

    gold_rows = [
        r
        for r in _load_jsonl(args.manifest)
        if (args.split == "all" or r.get("split") in {None, args.split})
        and (not args.bench or r.get("bench") == args.bench)
    ]
    gold = {r["id"]: r for r in gold_rows}
    pred_rows = _load_jsonl(args.preds)
    preds = {r["id"]: r for r in pred_rows}
    if len(preds) != len(pred_rows):
        raise SystemExit(f"duplicate prediction ids in {args.preds}")
    missing_ids = sorted(set(gold) - set(preds))
    if missing_ids and not args.allow_missing:
        raise SystemExit(
            f"incomplete predictions: missing {len(missing_ids)}/{len(gold)} ids; "
            "use --allow-missing only for diagnostics"
        )
    dino_cache: dict | None = None if args.skip_dino else {}
    rows = []
    for eid, grow in gold.items():
        prow = preds.get(eid)
        if not prow:
            continue
        gold_svg = _read_gold_svg(grow)
        gold_image = None if gold_svg else _read_gold_image(grow)
        pred_text = prow.get("pred_text") or prow.get("svg") or ""
        recovered_svg = prow.get("recovered_svg")
        if not recovered_svg:
            recovered_svg, _ = recover_svg_prefix(pred_text)
        sc = score_pred(
            pred_text,
            gold_svg,
            gold_image,
            dino_cache=dino_cache,
            recovered_svg=recovered_svg,
            hit_length_limit=bool(prow.get("hit_length_limit")),
        )
        sc["id"] = eid
        rows.append(sc)

    keys = [
        "parse_ok",
        "render_ok",
        "validity",
        "svg_open",
        "svg_close",
        "hit_length_limit",
        "recovered_validity",
        "salvage_gain",
        "ssim",
        "ssim_all",
        "recovered_ssim_all",
        "dino_cosine",
        "dino_cosine_all",
        "recovered_dino_cosine_all",
        "vfig_clean",
        "vfig_pass",
    ]
    agg = {k: mean(r.get(k) for r in rows) for k in keys}
    cis = {k: bootstrap_ci([r[k] for r in rows if r.get(k) is not None]) for k in keys if any(r.get(k) is not None for r in rows)}
    report = {
        "manifest": str(args.manifest),
        "preds": str(args.preds),
        "n_expected": len(gold),
        "n": len(rows),
        "n_missing": len(missing_ids),
        "aggregate": agg,
        "bootstrap": cis,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
