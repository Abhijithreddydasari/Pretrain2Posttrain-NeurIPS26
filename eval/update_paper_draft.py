"""Fill paper/draft.md results section from sweep metrics JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "paper" / "draft.md"
MARKER = "## 7. Results"


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _table_base_vs_sft(curves: dict) -> str:
  by = curves.get("by_bench", {})
  base = by.get("base_0pct", {}).get("vfig_id", {})
  final = by.get("pct_100", {}).get("vfig_id", {})
  rows = [
      ("Validity", base.get("validity"), final.get("validity")),
      ("SSIM", base.get("ssim"), final.get("ssim")),
      ("DINO cosine", base.get("dino_cosine"), final.get("dino_cosine")),
  ]
  lines = [
      "### Table 1 — Base @ 0% vs 100% SFT (VFIG-Bench ID, prompt)",
      "",
      "| Metric | Base @ 0% | SFT @ 100% |",
      "|--------|-----------|------------|",
  ]
  for name, b, f in rows:
      lines.append(f"| {name} | {_fmt(b)} | {_fmt(f)} |")
  return "\n".join(lines)


def _table_id_ood(curves: dict) -> str:
  by = curves.get("by_bench", {})
  id100 = by.get("pct_100", {}).get("vfig_id", {})
  ood100 = by.get("pct_100", {}).get("vfig_ood", {})
  lines = [
      "### Table 2 — VFIG ID vs OOD @ 100% SFT (validity)",
      "",
      "| Bench | Validity |",
      "|-------|----------|",
      f"| VFIG ID (400) | {_fmt(id100.get('validity'))} |",
      f"| VFIG OOD (198) | {_fmt(ood100.get('validity'))} |",
  ]
  return "\n".join(lines)


def _table_svg_diagrams(curves: dict) -> str:
  by = curves.get("by_bench", {})
  base = by.get("base_0pct", {}).get("svg_diagrams", {})
  final = by.get("pct_100", {}).get("svg_diagrams", {})
  lines = [
      "### Table 3 — SVG-Diagrams test (secondary, in-domain held-out)",
      "",
      "| Metric | Base @ 0% | SFT @ 100% |",
      "|--------|-----------|------------|",
      f"| Validity | {_fmt(base.get('validity'))} | {_fmt(final.get('validity'))} |",
      f"| SSIM | {_fmt(base.get('ssim'))} | {_fmt(final.get('ssim'))} |",
      f"| DINO | {_fmt(base.get('dino_cosine'))} | {_fmt(final.get('dino_cosine'))} |",
  ]
  return "\n".join(lines)


def _curves_note(curves: dict, emergence: dict | None) -> str:
  pcts = curves.get("pcts", [])
  syn = curves.get("syntax", [])
  struct = curves.get("structure", [])
  lines = [
      "### Figure 1 — Checkpoint curves (VFIG ID primary)",
      "",
      f"- Checkpoints: {pcts}",
      f"- Validity curve: {[round(x, 3) if x == x else None for x in syn]}",
      f"- SSIM curve: {[round(x, 3) if x == x else None for x in struct]}",
  ]
  if emergence:
      lines.append(f"- Emergence: `{json.dumps(emergence)}`")
  lines.append("- Plot: `outputs/metrics/sweep/emergence_prompt.png`")
  return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", type=Path, default=ROOT / "outputs/metrics/sweep/curves_prompt.json")
    ap.add_argument("--emergence", type=Path, default=ROOT / "outputs/metrics/sweep/emergence_prompt.json")
    ap.add_argument("--draft", type=Path, default=DRAFT)
    args = ap.parse_args()

    if not args.curves.exists():
        raise SystemExit(f"missing curves JSON: {args.curves}")
    curves = json.loads(args.curves.read_text(encoding="utf-8"))
    emergence = json.loads(args.emergence.read_text(encoding="utf-8")) if args.emergence.exists() else None

    block = "\n\n".join(
        [
            _table_base_vs_sft(curves),
            _table_id_ood(curves),
            _table_svg_diagrams(curves),
            _curves_note(curves, emergence),
            "### Figure 2 — Qualitative failures",
            "",
            "See `outputs/generations/sweep/` and render failure grid manually.",
        ]
    )

    text = args.draft.read_text(encoding="utf-8")
    if MARKER not in text:
        raise SystemExit(f"marker {MARKER!r} not found in {args.draft}")
    head, _ = text.split(MARKER, 1)
    tail_idx = text.find("## 8. Discussion", text.index(MARKER))
    tail = text[tail_idx:] if tail_idx != -1 else ""
    out = head + MARKER + "\n\n" + block + "\n\n" + tail
    args.draft.write_text(out, encoding="utf-8")
    print(f"updated {args.draft}")


if __name__ == "__main__":
    main()
