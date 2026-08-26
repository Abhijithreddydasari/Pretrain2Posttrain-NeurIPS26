"""HTML gallery: input image | gold render | pred render for sweep generations."""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structsvg_lib.svg_ops import extract_svg_blob, render_pil, validate_svg
from train.data_utils import load_manifest, resolve_image, resolve_svg


def _b64_png(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _pred_svg(text: str) -> str | None:
    return extract_svg_blob(text) or None


def _render_or_none(svg: str | None):
    if not svg:
        return None
    try:
        return render_pil(svg, size=512)
    except Exception:
        return None


def _load_preds(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["id"]] = row
    return out


def build_gallery(
    *,
    preds_path: Path,
    manifest_path: Path,
    out_html: Path,
    title: str,
) -> dict:
    preds = _load_preds(preds_path)
    gold_rows = {r["id"]: r for r in load_manifest(manifest_path)}
    cards: list[str] = []
    stats = {"n": 0, "valid": 0, "render_ok": 0, "closed_svg": 0}

    for eid, prow in preds.items():
        grow = gold_rows.get(eid)
        if not grow:
            continue
        stats["n"] += 1
        pred_text = prow.get("pred_text", "")
        pred_blob = _pred_svg(pred_text)
        val = validate_svg(pred_blob or pred_text, try_render=True)
        if val.ok:
            stats["valid"] += 1
        if val.render_ok:
            stats["render_ok"] += 1
        if pred_blob and "</svg>" in pred_blob.lower():
            stats["closed_svg"] += 1

        try:
            input_img = resolve_image(grow)
            input_b64 = _b64_png(input_img)
        except Exception as e:  # noqa: BLE001
            input_b64 = ""
            input_err = str(e)
        else:
            input_err = ""

        try:
            gold_svg = resolve_svg(grow)
        except Exception:
            gold_svg = grow.get("svg") or ""
        gold_img = _render_or_none(gold_svg)
        pred_img = _render_or_none(pred_blob)

        gold_b64 = _b64_png(gold_img) if gold_img else ""
        pred_b64 = _b64_png(pred_img) if pred_img else ""

        status = []
        status.append("valid" if val.ok else "invalid")
        status.append("render" if val.render_ok else "no-render")
        status.append("closed" if pred_blob and "</svg>" in pred_blob.lower() else "truncated")
        status.append(f"{len(pred_text)} chars")

        cards.append(
            f"""
<div class="card">
  <div class="meta"><b>{html.escape(eid)}</b><br/>
    <span class="tag">{' · '.join(status)}</span></div>
  <div class="row">
    <div class="panel"><div class="label">Input</div>
      {'<img src="data:image/png;base64,' + input_b64 + '"/>' if input_b64 else f'<div class="err">{html.escape(input_err)}</div>'}
    </div>
    <div class="panel"><div class="label">Gold</div>
      {'<img src="data:image/png;base64,' + gold_b64 + '"/>' if gold_b64 else '<div class="err">render failed</div>'}
    </div>
    <div class="panel"><div class="label">Pred @ {html.escape(title)}</div>
      {'<img src="data:image/png;base64,' + pred_b64 + '"/>' if pred_b64 else '<div class="err">no renderable pred SVG</div>'}
    </div>
  </div>
</div>"""
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Segoe UI, sans-serif; margin: 20px; background:#0f1115; color:#e8e8e8; }}
h1 {{ font-size: 22px; }}
.summary {{ color:#9aa4b2; margin-bottom: 20px; }}
.grid {{ display:flex; flex-direction:column; gap:20px; }}
.card {{ background:#1a1f29; border:1px solid #2a3344; border-radius:10px; padding:12px; }}
.meta {{ font-size:12px; margin-bottom:10px; word-break:break-all; }}
.tag {{ color:#7ec8e3; }}
.row {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; }}
.panel {{ background:#fff; border-radius:6px; padding:6px; min-height:180px; }}
.panel img {{ width:100%; height:auto; display:block; }}
.label {{ font-size:11px; color:#333; margin-bottom:4px; font-weight:600; }}
.err {{ color:#b00020; font-size:12px; padding:8px; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="summary">n={stats['n']} · valid={stats['valid']} · render_ok={stats['render_ok']} · closed_svg={stats['closed_svg']}</p>
<div class="grid">{''.join(cards)}</div>
</body></html>"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_doc, encoding="utf-8")
    stats["html"] = str(out_html)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    title = args.title or args.preds.stem
    out = args.out or Path("outputs/gallery") / f"{args.preds.stem}.html"
    stats = build_gallery(preds_path=args.preds, manifest_path=args.manifest, out_html=out, title=title)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
