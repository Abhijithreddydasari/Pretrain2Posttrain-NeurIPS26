"""Extract scene graphs from StructSVG-style markup (gold path) and heuristic recovery."""
from __future__ import annotations

import json
import re
from xml.etree import ElementTree as ET

from structsvg_lib.scene_graph import Entity, Relation, SceneGraph
from structsvg_lib.svg_ops import _local, parse_svg


def scene_graph_from_sidecar(path_or_dict) -> SceneGraph:
    if isinstance(path_or_dict, dict):
        return SceneGraph.from_dict(path_or_dict)
    with open(path_or_dict, encoding="utf-8") as f:
        return SceneGraph.from_dict(json.load(f))


def extract_from_structsvg_markup(svg: str, grammar: str | None = None) -> SceneGraph | None:
    """
    StructSVG generators embed data-entity-id / data-entity-type / data-label
    and data-edge attributes so gold recovery is deterministic.
    """
    root, err = parse_svg(svg)
    if err or root is None:
        return None

    g = grammar or root.attrib.get("data-grammar", "workflow")
    entities: list[Entity] = []
    relations: list[Relation] = []

    for e in root.iter():
        eid = e.attrib.get("data-entity-id")
        if eid:
            typ = e.attrib.get("data-entity-type", _local(e.tag))
            label = e.attrib.get("data-label", (e.text or "").strip())
            bbox = _bbox_from_elem(e)
            entities.append(Entity(id=eid, type=typ, label=label, bbox=bbox))
        # edges encoded as data-edge="src>dst:type:label"
        edge = e.attrib.get("data-edge")
        if edge:
            m = re.match(r"([^>]+)>([^:]+):([^:]*):(.*)", edge)
            if m:
                relations.append(
                    Relation(src=m.group(1), dst=m.group(2), type=m.group(3) or "edge", label=m.group(4))
                )

    # also allow <metadata> JSON block
    for e in root.iter():
        if _local(e.tag) == "desc" and e.attrib.get("data-scenegraph") == "1" and e.text:
            try:
                return SceneGraph.from_dict(json.loads(e.text))
            except json.JSONDecodeError:
                pass

    if not entities and not relations:
        return None
    return SceneGraph(grammar=g, entities=entities, relations=relations)  # type: ignore[arg-type]


def _bbox_from_elem(e: ET.Element) -> tuple[float, float, float, float] | None:
    tag = _local(e.tag)
    try:
        if tag == "rect":
            return float(e.get("x", 0)), float(e.get("y", 0)), float(e.get("width", 0)), float(e.get("height", 0))
        if tag == "circle":
            cx, cy, r = float(e.get("cx", 0)), float(e.get("cy", 0)), float(e.get("r", 0))
            return cx - r, cy - r, 2 * r, 2 * r
        if tag == "text":
            x, y = float(e.get("x", 0)), float(e.get("y", 0))
            return x, y - 12, 40, 16
    except ValueError:
        return None
    return None


def heuristic_extract_workflow(svg: str) -> SceneGraph:
    """Best-effort extraction for predicted SVG without data-* attrs."""
    root, err = parse_svg(svg)
    entities: list[Entity] = []
    relations: list[Relation] = []
    if err or root is None:
        return SceneGraph(grammar="workflow", entities=[], relations=[])

    texts: list[tuple[str, float, float]] = []
    boxes: list[tuple[str, float, float, float, float]] = []
    idx = 0
    for e in root.iter():
        tag = _local(e.tag)
        if tag == "text" and (e.text or "").strip():
            label = e.text.strip()
            x, y = float(e.get("x", 0)), float(e.get("y", 0))
            eid = f"t{idx}"
            idx += 1
            entities.append(Entity(id=eid, type="node", label=label, bbox=(x, y - 12, 60, 20)))
            texts.append((eid, x, y))
        if tag == "rect":
            x, y = float(e.get("x", 0)), float(e.get("y", 0))
            w, h = float(e.get("width", 0)), float(e.get("height", 0))
            eid = f"b{idx}"
            idx += 1
            # label = nearest text center
            label = ""
            cx, cy = x + w / 2, y + h / 2
            best = None
            for tid, tx, ty in texts:
                d = (tx - cx) ** 2 + (ty - cy) ** 2
                if best is None or d < best[0]:
                    best = (d, tid)
            if best and best[0] < 80**2:
                # use that text label
                te = next(en for en in entities if en.id == best[1])
                label = te.label
            entities.append(Entity(id=eid, type="box", label=label, bbox=(x, y, w, h)))
            boxes.append((eid, x, y, w, h))

    # lines as edges between nearest box centers
    for e in root.iter():
        if _local(e.tag) != "line":
            continue
        x1, y1 = float(e.get("x1", 0)), float(e.get("y1", 0))
        x2, y2 = float(e.get("x2", 0)), float(e.get("y2", 0))

        def nearest(px, py):
            best = None
            for bid, x, y, w, h in boxes:
                cx, cy = x + w / 2, y + h / 2
                d = (cx - px) ** 2 + (cy - py) ** 2
                if best is None or d < best[0]:
                    best = (d, bid)
            return best[1] if best else None

        s, d = nearest(x1, y1), nearest(x2, y2)
        if s and d and s != d:
            relations.append(Relation(src=s, dst=d, type="edge"))

    return SceneGraph(grammar="workflow", entities=entities, relations=relations)
