"""FlowGen adapter: convert render-code + triplets into eval rows (SVG targets optional)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_triplets_file(text: str) -> list[tuple[str, str, str]]:
    """Best-effort parse of FlowGen-style node-label-node lines."""
    trips = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # common patterns: a\tlabel\tb  or  (a, label, b)
        m = re.match(r"\(?\s*([^,|\t]+)\s*[,|\t]\s*([^,|\t]*)\s*[,|\t]\s*([^)\s]+)\s*\)?", line)
        if m:
            trips.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
    return trips


def triplets_to_scene_graph(trips: list[tuple[str, str, str]]) -> dict:
    nodes = {}
    rels = []
    for a, lab, b in trips:
        if a not in nodes:
            nodes[a] = {"id": a, "type": "node", "label": a}
        if b not in nodes:
            nodes[b] = {"id": b, "type": "node", "label": b}
        rels.append({"src": a, "dst": b, "type": "edge", "label": lab})
    return {
        "grammar": "workflow",
        "entities": list(nodes.values()),
        "relations": rels,
        "meta": {"source": "flowgen"},
    }


def main():
    ap = argparse.ArgumentParser(description="Convert a folder of FlowGen-like files to manifest jsonl")
    ap.add_argument("--root", type=Path, required=True, help="directory with .png + .txt/.mmd/.dot")
    ap.add_argument("--out", type=Path, default=Path("data/processed/flowgen/eval_manifest.jsonl"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    pngs = list(args.root.rglob("*.png"))
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for png in pngs:
            stem = png.with_suffix("")
            trip_path = None
            for ext in (".txt", ".triples", ".triplet"):
                cand = Path(str(stem) + ext)
                if cand.exists():
                    trip_path = cand
                    break
            trips = parse_triplets_file(trip_path.read_text(encoding="utf-8", errors="ignore")) if trip_path else []
            row = {
                "id": png.stem,
                "image_path": str(png.as_posix()),
                "scene_graph": triplets_to_scene_graph(trips),
                "source": "flowgen",
                "code_path": None,
            }
            for ext in (".mmd", ".dot", ".puml"):
                c = Path(str(stem) + ext)
                if c.exists():
                    row["code_path"] = str(c.as_posix())
                    break
            f.write(json.dumps(row) + "\n")
            n += 1
    print(f"wrote {n} rows → {args.out}")
    print("Note: native SVG targets require a deterministic Graphviz/Mermaid render step (Phase 3).")


if __name__ == "__main__":
    main()
