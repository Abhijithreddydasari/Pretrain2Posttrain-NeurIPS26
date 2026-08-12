"""Evaluate cached generations against gold manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structsvg_lib.extract import (
    extract_from_structsvg_markup,
    heuristic_extract_workflow,
    scene_graph_from_sidecar,
)
from structsvg_lib.metrics import aggregate_scores, bootstrap_ci, id_ood_gap, score_example


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_graph(row: dict):
    if "scene_graph" in row:
        return scene_graph_from_sidecar(row["scene_graph"])
    if "scene_graph_path" in row:
        return scene_graph_from_sidecar(row["scene_graph_path"])
    # try extract from gold svg
    svg = row.get("svg") or Path(row["svg_path"]).read_text(encoding="utf-8")
    return extract_from_structsvg_markup(svg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="gold manifest jsonl")
    ap.add_argument("--preds", type=Path, required=True, help="predictions jsonl id->pred_text")
    ap.add_argument("--out", type=Path, default=Path("outputs/metrics/eval.json"))
    ap.add_argument("--split", type=str, default="id")
    args = ap.parse_args()

    gold = {r["id"]: r for r in load_jsonl(args.manifest)}
    preds = {r["id"]: r for r in load_jsonl(args.preds)}

    rows = []
    for eid, grow in gold.items():
        if args.split != "all" and grow.get("split") not in {None, args.split}:
            continue
        prow = preds.get(eid)
        if not prow:
            continue
        gold_graph = resolve_graph(grow)
        pred_text = prow.get("pred_text") or prow.get("svg") or ""
        pred_graph = extract_from_structsvg_markup(pred_text) or heuristic_extract_workflow(pred_text)
        gold_svg = grow.get("svg")
        if not gold_svg and grow.get("svg_path"):
            gold_svg = Path(grow["svg_path"]).read_text(encoding="utf-8")
        sc = score_example(pred_text, gold_svg, gold_graph, pred_graph)
        sc["id"] = eid
        rows.append(sc)

    agg = aggregate_scores(rows)
    cis = {
        k: bootstrap_ci([r[k] for r in rows if r.get(k) is not None])
        for k in ["validity", "entity_f1", "relation_f1", "spatial"]
    }
    report = {"split": args.split, "n": len(rows), "aggregate": agg, "bootstrap": cis}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
