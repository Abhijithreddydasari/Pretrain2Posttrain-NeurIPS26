"""Save PNG previews + lightweight HTML (file refs, not base64)."""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structsvg_lib.svg_ops import extract_svg_blob, render_pil, validate_svg
from train.data_utils import load_manifest, resolve_image, resolve_svg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    title = args.title or args.preds.stem
    out_dir = args.out_dir or Path("outputs/gallery") / args.preds.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = {json.loads(l)["id"]: json.loads(l) for l in args.preds.read_text().splitlines() if l.strip()}
    gold = {r["id"]: r for r in load_manifest(args.manifest)}

    cards = []
    valid = closed = 0
    for eid, prow in preds.items():
        grow = gold.get(eid)
        if not grow:
            continue
        safe = eid.replace("/", "_").replace("\\", "_")[:80]
        pred_text = prow.get("pred_text", "")
        pred_blob = extract_svg_blob(pred_text) or pred_text
        val = validate_svg(pred_blob, try_render=False)
        valid += int(val.ok)
        closed += int("</svg>" in pred_blob.lower())

        input_png = out_dir / f"{safe}_input.png"
        gold_png = out_dir / f"{safe}_gold.png"
        pred_png = out_dir / f"{safe}_pred.png"

        try:
            resolve_image(grow).save(input_png)
            input_ref = input_png.name
        except Exception as e:  # noqa: BLE001
            input_ref = ""
            input_err = str(e)
        else:
            input_err = ""

        gold_ref = ""
        try:
            gs = resolve_svg(grow)
            if gs:
                render_pil(gs, size=512).save(gold_png)
                gold_ref = gold_png.name
        except Exception:
            pass

        pred_ref = ""
        (out_dir / f"{safe}_pred.txt").write_text(pred_text[:50000], encoding="utf-8")
        render_blob = pred_blob
        if "</svg>" not in render_blob.lower() and render_blob.lstrip().lower().startswith("<svg"):
            render_blob = render_blob.rstrip() + "</svg>"
        if len(render_blob) < 15000:
            try:
                render_pil(render_blob, size=512).save(pred_png)
                pred_ref = pred_png.name
            except Exception:
                pass

        cards.append((eid, input_ref, gold_ref, pred_ref, val.ok, "</svg>" in pred_blob.lower(), len(pred_text), input_err))

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:Segoe UI,sans-serif;background:#111;color:#eee;margin:20px}",
        ".card{background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:12px;margin:16px 0}",
        ".row{display:flex;gap:12px;flex-wrap:wrap}.panel{background:#fff;padding:6px;border-radius:4px}",
        ".panel img{max-height:280px}.meta{font-size:12px;color:#aaa;margin-bottom:8px}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p>valid={valid}/{len(cards)} closed={closed}/{len(cards)}</p>",
    ]
    for eid, inp, gld, prd, ok, cl, nchar, ierr in cards:
        parts.append(f"<div class='card'><div class='meta'><b>{html.escape(eid)}</b> · valid={ok} · closed={cl} · {nchar} chars</div><div class='row'>")
        if inp:
            parts.append(f"<div class='panel'><div>Input</div><img src='{html.escape(inp)}'/></div>")
        elif ierr:
            parts.append(f"<div class='panel'><div>Input error</div>{html.escape(ierr)}</div>")
        if gld:
            parts.append(f"<div class='panel'><div>Gold</div><img src='{html.escape(gld)}'/></div>")
        if prd:
            parts.append(f"<div class='panel'><div>Pred</div><img src='{html.escape(prd)}'/></div>")
        else:
            parts.append("<div class='panel'><div>Pred</div>no render (truncated/invalid)</div>")
        parts.append("</div></div>")
    parts.append("</body></html>")
    (out_dir / "index.html").write_text("".join(parts), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "n": len(cards), "valid": valid, "closed": closed}, indent=2))


if __name__ == "__main__":
    main()
