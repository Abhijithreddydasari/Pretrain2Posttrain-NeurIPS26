"""Scene-graph schema and matching metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Grammar = Literal["workflow", "geometry"]


@dataclass
class Entity:
    id: str
    type: str
    label: str = ""
    # axis-aligned box in viewBox coords
    bbox: tuple[float, float, float, float] | None = None  # x, y, w, h
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    src: str
    dst: str
    type: str
    label: str = ""


@dataclass
class SceneGraph:
    grammar: Grammar
    entities: list[Entity]
    relations: list[Relation]
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "grammar": self.grammar,
            "entities": [asdict(e) for e in self.entities],
            "relations": [asdict(r) for r in self.relations],
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SceneGraph":
        ents = []
        for e in d.get("entities", []):
            ee = dict(e)
            if ee.get("bbox") is not None:
                ee["bbox"] = tuple(ee["bbox"])
            ents.append(Entity(**ee))
        rels = [Relation(**r) for r in d.get("relations", [])]
        return SceneGraph(grammar=d["grammar"], entities=ents, relations=rels, meta=d.get("meta", {}))


def _entity_key(e: Entity) -> tuple[str, str]:
    return (e.type, (e.label or "").strip().lower())


def _relation_key(r: Relation, id_to_key: dict[str, tuple[str, str]]) -> tuple:
    return (
        r.type,
        id_to_key.get(r.src, ("?", "")),
        id_to_key.get(r.dst, ("?", "")),
        (r.label or "").strip().lower(),
    )


def f1_from_sets(pred: set, gold: set) -> dict[str, float]:
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred & gold)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gold) if gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def entity_f1(pred: SceneGraph, gold: SceneGraph) -> dict[str, float]:
    p = {_entity_key(e) for e in pred.entities}
    g = {_entity_key(e) for e in gold.entities}
    return f1_from_sets(p, g)


def relation_f1(pred: SceneGraph, gold: SceneGraph) -> dict[str, float]:
    pmap = {e.id: _entity_key(e) for e in pred.entities}
    gmap = {e.id: _entity_key(e) for e in gold.entities}
    p = {_relation_key(r, pmap) for r in pred.relations}
    g = {_relation_key(r, gmap) for r in gold.relations}
    return f1_from_sets(p, g)


def reachability_accuracy(pred: SceneGraph, gold: SceneGraph, max_hops: int = 3) -> float:
    """Fraction of gold directed adjacency paths (up to max_hops) present in pred via label keys."""

    def adj(sg: SceneGraph) -> dict[tuple[str, str], set[tuple[str, str]]]:
        key = {e.id: _entity_key(e) for e in sg.entities}
        a: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for r in sg.relations:
            if r.type not in {"edge", "flow", "connects"}:
                continue
            s, d = key.get(r.src), key.get(r.dst)
            if s and d:
                a.setdefault(s, set()).add(d)
        return a

    g_adj = adj(gold)
    p_adj = adj(pred)

    def paths(a: dict, hops: int) -> set[tuple]:
        out: set[tuple] = set()

        def dfs(node, trail):
            if len(trail) - 1 > hops:
                return
            if len(trail) >= 2:
                out.add(tuple(trail))
            if len(trail) - 1 == hops:
                return
            for nxt in a.get(node, ()):
                if nxt in trail:
                    continue
                dfs(nxt, trail + [nxt])

        for start in a:
            dfs(start, [start])
        return out

    gold_paths = paths(g_adj, max_hops)
    if not gold_paths:
        return 1.0
    pred_paths = paths(p_adj, max_hops)
    return len(gold_paths & pred_paths) / len(gold_paths)


def geometry_relation_accuracy(pred: SceneGraph, gold: SceneGraph) -> float:
    gmap = {e.id: _entity_key(e) for e in gold.entities}
    pmap = {e.id: _entity_key(e) for e in pred.entities}
    gold_set = {_relation_key(r, gmap) for r in gold.relations}
    pred_set = {_relation_key(r, pmap) for r in pred.relations}
    if not gold_set:
        return 1.0
    return len(gold_set & pred_set) / len(gold_set)


def spatial_aggregate(pred: SceneGraph, gold: SceneGraph) -> float:
    if gold.grammar == "workflow":
        return reachability_accuracy(pred, gold)
    return geometry_relation_accuracy(pred, gold)
