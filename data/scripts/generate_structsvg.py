"""Generate StructSVG workflow + geometry examples with scene-graph sidecars."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from structsvg_lib.scene_graph import Entity, Relation, SceneGraph
from structsvg_lib.svg_ops import normalize_svg, parse_svg, render_pil, validate_svg

WORKFLOW_LABELS = [
    "Start",
    "Ingest",
    "Validate",
    "Transform",
    "Branch",
    "Merge",
    "Store",
    "Notify",
    "End",
    "Review",
    "Compute",
    "Filter",
]


def _svg_wrap(inner: str, grammar: str, scene: SceneGraph) -> str:
    meta = json.dumps(scene.as_dict(), separators=(",", ":"))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" data-grammar="{grammar}">
<desc data-scenegraph="1">{meta}</desc>
{inner}
</svg>'''
    root, err = parse_svg(svg)
    assert root is not None, err
    return normalize_svg(root)


def make_workflow_example(idx: int, split: str = "train", seed: int = 0) -> dict:
    rng = random.Random(seed + idx * 17)
    ood = split == "ood"

    if ood:
        # branch then merge, longer path, nested group
        n = rng.randint(7, 9)
        pattern = "branch_merge"
    else:
        n = rng.randint(3, 6)
        pattern = rng.choice(["chain", "single_branch", "single_merge"])

    labels = rng.sample(WORKFLOW_LABELS, k=min(n, len(WORKFLOW_LABELS)))
    while len(labels) < n:
        labels.append(f"N{len(labels)}")

    entities: list[Entity] = []
    relations: list[Relation] = []
    parts: list[str] = []

    # layout
    positions: list[tuple[float, float]] = []
    if pattern == "chain":
        for i in range(n):
            positions.append((60 + i * (400 / max(n - 1, 1)), 220.0))
    elif pattern == "single_branch":
        positions.append((80, 240))
        positions.append((220, 140))
        positions.append((220, 340))
        for i in range(3, n):
            positions.append((360 + (i - 3) * 50, 240.0))
    elif pattern == "single_merge":
        positions.append((80, 140))
        positions.append((80, 340))
        positions.append((240, 240))
        for i in range(3, n):
            positions.append((360 + (i - 3) * 50, 240.0))
    else:  # branch_merge OOD
        positions = [(60, 240), (180, 120), (180, 360), (320, 240), (420, 240)]
        while len(positions) < n:
            positions.append((420, 120 + 40 * len(positions)))

    for i, (lab, (x, y)) in enumerate(zip(labels, positions)):
        eid = f"n{i}"
        typ = "decision" if lab in {"Branch", "Validate", "Filter"} else "node"
        w, h = 100.0, 40.0
        entities.append(Entity(id=eid, type=typ, label=lab, bbox=(x - w / 2, y - h / 2, w, h)))
        fill = "#fde68a" if typ == "decision" else "#bfdbfe"
        parts.append(
            f'<rect data-entity-id="{eid}" data-entity-type="{typ}" data-label="{lab}" '
            f'x="{x - w/2:.1f}" y="{y - h/2:.1f}" width="{w}" height="{h}" rx="6" '
            f'fill="{fill}" stroke="#1e3a8a" stroke-width="2"/>'
        )
        parts.append(
            f'<text data-entity-id="{eid}_t" data-entity-type="label" data-label="{lab}" '
            f'x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" font-size="14" '
            f'font-family="Arial">{lab}</text>'
        )

    def edge(i: int, j: int):
        a, b = entities[i], entities[j]
        ax, ay = a.bbox[0] + a.bbox[2] / 2, a.bbox[1] + a.bbox[3] / 2
        bx, by = b.bbox[0] + b.bbox[2] / 2, b.bbox[1] + b.bbox[3] / 2
        relations.append(Relation(src=a.id, dst=b.id, type="edge"))
        parts.append(
            f'<line data-edge="{a.id}>{b.id}:edge:" x1="{ax:.1f}" y1="{ay:.1f}" '
            f'x2="{bx:.1f}" y2="{by:.1f}" stroke="#111827" stroke-width="2" marker-end="url(#arrow)"/>'
        )

    if pattern == "chain":
        for i in range(n - 1):
            edge(i, i + 1)
    elif pattern == "single_branch":
        edge(0, 1)
        edge(0, 2)
        for i in range(3, n):
            edge(1 if i % 2 else 2, i)
    elif pattern == "single_merge":
        edge(0, 2)
        edge(1, 2)
        for i in range(3, n):
            edge(i - 1, i)
    else:
        edge(0, 1)
        edge(0, 2)
        edge(1, 3)
        edge(2, 3)
        for i in range(4, n):
            edge(3, i)

    if ood:
        # nested group visual
        parts.insert(
            0,
            '<g data-entity-id="grp0" data-entity-type="group" data-label="cluster">'
            '<rect x="140" y="80" width="120" height="320" fill="none" stroke="#9ca3af" '
            'stroke-dasharray="6 4"/></g>',
        )
        entities.append(Entity(id="grp0", type="group", label="cluster", bbox=(140, 80, 120, 320)))

    defs = (
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" '
        'orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#111827"/></marker></defs>'
    )
    scene = SceneGraph(
        grammar="workflow",
        entities=entities,
        relations=relations,
        meta={"pattern": pattern, "split": split, "idx": idx},
    )
    svg = _svg_wrap(defs + "\n".join(parts), "workflow", scene)
    return {
        "id": f"wf_{split}_{idx:04d}",
        "split": split,
        "grammar": "workflow",
        "svg": svg,
        "scene_graph": scene.as_dict(),
        "pattern": pattern,
    }


def make_geometry_example(idx: int, split: str = "train", seed: int = 0) -> dict:
    rng = random.Random(seed + 1000 + idx * 13)
    ood = split == "ood"

    entities: list[Entity] = []
    relations: list[Relation] = []
    parts: list[str] = []

    # base points
    pts = [
        ("A", 120 + rng.randint(-20, 20), 360),
        ("B", 380 + rng.randint(-20, 20), 360),
        ("C", 250 + rng.randint(-30, 30), 140 + rng.randint(-20, 20)),
    ]
    for name, x, y in pts:
        eid = f"p{name}"
        entities.append(Entity(id=eid, type="point", label=name, bbox=(x - 4, y - 4, 8, 8)))
        parts.append(
            f'<circle data-entity-id="{eid}" data-entity-type="point" data-label="{name}" '
            f'cx="{x}" cy="{y}" r="4" fill="#111827"/>'
        )
        parts.append(
            f'<text data-entity-id="{eid}_t" data-entity-type="label" data-label="{name}" '
            f'x="{x + 8}" y="{y - 8}" font-size="16" font-family="Arial">{name}</text>'
        )

    def add_segment(a: str, b: str):
        pa = next(e for e in entities if e.label == a)
        pb = next(e for e in entities if e.label == b)
        ax, ay = pa.bbox[0] + 4, pa.bbox[1] + 4
        bx, by = pb.bbox[0] + 4, pb.bbox[1] + 4
        sid = f"s{a}{b}"
        entities.append(Entity(id=sid, type="segment", label=f"{a}{b}", bbox=(min(ax, bx), min(ay, by), abs(bx - ax), abs(by - ay)))
        )
        relations.append(Relation(src=pa.id, dst=sid, type="incident"))
        relations.append(Relation(src=pb.id, dst=sid, type="incident"))
        parts.append(
            f'<line data-entity-id="{sid}" data-entity-type="segment" data-label="{a}{b}" '
            f'data-edge="{pa.id}>{pb.id}:connects:" x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" '
            f'stroke="#1d4ed8" stroke-width="2"/>'
        )

    # always triangle edges for visibility
    add_segment("A", "B")
    add_segment("B", "C")
    add_segment("C", "A")

    # single relation for ID/train; combinations for OOD
    relations_pool = ["left_of", "above", "contains_circle"]
    if ood:
        chosen = relations_pool  # combination
    else:
        chosen = [rng.choice(relations_pool)]

    pa = next(e for e in entities if e.label == "A")
    pb = next(e for e in entities if e.label == "B")
    pc = next(e for e in entities if e.label == "C")

    if "left_of" in chosen:
        relations.append(Relation(src=pa.id, dst=pb.id, type="left_of"))
    if "above" in chosen:
        relations.append(Relation(src=pc.id, dst=pa.id, type="above"))
    if "contains_circle" in chosen:
        # circum-ish circle around C
        cx, cy = pc.bbox[0] + 4, pc.bbox[1] + 4
        r = 36
        eid = "circ0"
        entities.append(Entity(id=eid, type="circle", label="ω", bbox=(cx - r, cy - r, 2 * r, 2 * r)))
        relations.append(Relation(src=eid, dst=pc.id, type="contains"))
        parts.append(
            f'<circle data-entity-id="{eid}" data-entity-type="circle" data-label="ω" '
            f'cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#dc2626" stroke-width="2"/>'
        )

    scene = SceneGraph(
        grammar="geometry",
        entities=entities,
        relations=relations,
        meta={"split": split, "idx": idx, "relations": chosen},
    )
    svg = _svg_wrap("\n".join(parts), "geometry", scene)
    return {
        "id": f"geo_{split}_{idx:04d}",
        "split": split,
        "grammar": "geometry",
        "svg": svg,
        "scene_graph": scene.as_dict(),
        "pattern": "+".join(chosen),
    }


def write_example(ex: dict, out_dir: Path, render: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{ex['id']}.svg"
    json_path = out_dir / f"{ex['id']}.json"
    svg_path.write_text(ex["svg"], encoding="utf-8")
    json_path.write_text(json.dumps(ex["scene_graph"], indent=2), encoding="utf-8")
    row = {
        "id": ex["id"],
        "split": ex["split"],
        "grammar": ex["grammar"],
        "svg_path": str(svg_path.as_posix()),
        "scene_graph_path": str(json_path.as_posix()),
        "scene_graph": ex["scene_graph"],
        "svg": ex["svg"],
        "pattern": ex.get("pattern"),
    }
    if render:
        try:
            img = render_pil(ex["svg"], size=256)
            png_path = out_dir / f"{ex['id']}.png"
            img.save(png_path)
            row["image_path"] = str(png_path.as_posix())
        except Exception:
            row["image_path"] = None
    val = validate_svg(ex["svg"], try_render=False)
    row["svg_sha256"] = val.sha256
    return row


def generate(
    out_root: Path,
    train_n: int = 2000,
    id_n: int = 250,
    ood_n: int = 250,
    seed: int = 42,
    pilot: bool = False,
) -> None:
    if pilot:
        train_n, id_n, ood_n = 20, 8, 8

    manifests = {"train": [], "id": [], "ood": []}
    half_train = train_n // 2
    half_id = id_n // 2
    half_ood = ood_n // 2

    specs = [
        ("train", half_train, make_workflow_example),
        ("train", train_n - half_train, make_geometry_example),
        ("id", half_id, make_workflow_example),
        ("id", id_n - half_id, make_geometry_example),
        ("ood", half_ood, make_workflow_example),
        ("ood", ood_n - half_ood, make_geometry_example),
    ]

    # rebuild with proper counts per grammar
    plans: list[tuple[str, int, callable]] = [
        ("train", half_train, make_workflow_example),
        ("train", train_n - half_train, make_geometry_example),
        ("id", half_id, make_workflow_example),
        ("id", id_n - half_id, make_geometry_example),
        ("ood", half_ood, make_workflow_example),
        ("ood", ood_n - half_ood, make_geometry_example),
    ]

    counters = {"train": 0, "id": 0, "ood": 0}
    for split, count, fn in plans:
        for _ in range(count):
            i = counters[split]
            counters[split] += 1
            ex = fn(i, split=split, seed=seed)
            sub = out_root / split / ex["grammar"]
            row = write_example(ex, sub, render=True)
            manifests[split].append(row)

    man_dir = out_root
    man_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in manifests.items():
        path = man_dir / f"{split}_manifest.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                # keep sidecar paths; drop bulky svg from manifest optional — keep for fixtures
                slim = {k: v for k, v in r.items() if k != "svg"}
                # retain svg for small pilots
                if pilot:
                    slim["svg"] = r["svg"]
                f.write(json.dumps(slim) + "\n")
        print(f"wrote {path} ({len(rows)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "structsvg")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--train-n", type=int, default=2000)
    ap.add_argument("--id-n", type=int, default=250)
    ap.add_argument("--ood-n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixtures", action="store_true", help="write a few examples into data/fixtures")
    args = ap.parse_args()

    if args.fixtures:
        fix = ROOT / "data" / "fixtures"
        for i, fn in enumerate([make_workflow_example, make_geometry_example]):
            ex = fn(i, split="id", seed=42)
            write_example(ex, fix, render=False)
            (fix / f"{ex['id']}.svg").write_text(ex["svg"], encoding="utf-8")
            (fix / f"{ex['id']}.json").write_text(json.dumps(ex["scene_graph"], indent=2), encoding="utf-8")
        print(f"fixtures in {fix}")
        return

    generate(args.out, args.train_n, args.id_n, args.ood_n, args.seed, pilot=args.pilot)


if __name__ == "__main__":
    main()
