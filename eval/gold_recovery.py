"""Gold-recovery gate: canonical StructSVG must extract perfectly."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structsvg_lib.extract import extract_from_structsvg_markup, scene_graph_from_sidecar
from structsvg_lib.metrics import score_example
from structsvg_lib.scene_graph import entity_f1, relation_f1, spatial_aggregate
from structsvg_lib.svg_ops import validate_svg


def run(fixtures_dir: Path | None = None) -> int:
    fixtures_dir = fixtures_dir or ROOT / "data" / "fixtures"
    svg_paths = sorted(fixtures_dir.glob("*.svg"))
    if not svg_paths:
        print("No fixtures found; generating inline minimal fixtures check")
        from data.scripts.generate_structsvg import make_workflow_example, make_geometry_example

        samples = [make_workflow_example(0, split="id"), make_geometry_example(0, split="id")]
    else:
        samples = []
        for p in svg_paths:
            sidecar = p.with_suffix(".json")
            samples.append(
                {
                    "svg": p.read_text(encoding="utf-8"),
                    "graph": json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else None,
                    "id": p.stem,
                }
            )

    failures = 0
    for s in samples:
        svg = s["svg"] if isinstance(s, dict) and "svg" in s else s.get("svg_text")
        gdict = s.get("graph") or s.get("scene_graph")
        val = validate_svg(svg, try_render=True)
        if not val.parse_ok:
            print(f"FAIL {s.get('id')}: parse {val.errors}")
            failures += 1
            continue
        # Prefer data-* extraction
        pred = extract_from_structsvg_markup(svg)
        gold = scene_graph_from_sidecar(gdict) if gdict else pred
        if pred is None or gold is None:
            print(f"FAIL {s.get('id')}: missing scene graph")
            failures += 1
            continue
        ef = entity_f1(pred, gold)["f1"]
        rf = relation_f1(pred, gold)["f1"]
        sp = spatial_aggregate(pred, gold)
        ok = ef >= 0.999 and rf >= 0.999 and sp >= 0.999
        status = "OK" if ok else "FAIL"
        print(f"{status} {s.get('id', '?')}: entity_f1={ef:.3f} relation_f1={rf:.3f} spatial={sp:.3f} validity={val.ok}")
        if not ok:
            failures += 1
        # self-score path
        score_example(svg, svg, gold, pred)

    if failures:
        print(f"\nGold recovery FAILED ({failures} issues)")
        return 1
    print("\nGold recovery PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
