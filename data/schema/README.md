# Scene graph (StructSVG)

```json
{
  "grammar": "workflow" | "geometry",
  "entities": [{"id": "n0", "type": "node|decision|box|point|segment|circle|group|label", "label": "...", "bbox": [x,y,w,h]}],
  "relations": [{"src": "n0", "dst": "n1", "type": "edge|incident|left_of|above|contains|connects", "label": ""}],
  "meta": {}
}
```

Gold SVGs embed the same JSON in `<desc data-scenegraph="1">` and duplicate ids on elements via `data-entity-*` / `data-edge`.
