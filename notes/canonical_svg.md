# Canonical SVG Profile (locked)

Target format for all training and evaluation in this project.

## Hard rules

1. **Native SVG only** — XML SVG markup, not a primitive protocol and not raster embeddings.
2. **Root element** — single `<svg>` with explicit `xmlns="http://www.w3.org/2000/svg"`.
3. **Viewport** — fixed `viewBox="0 0 512 512"` and `width="512" height="512"` unless a dataset explicitly documents otherwise.
4. **Safe subset** — allowed elements: `svg`, `g`, `rect`, `circle`, `ellipse`, `line`, `polyline`, `polygon`, `path`, `text`, `tspan`, `defs`, `marker`, `title`, `desc`.
5. **Forbidden** — `<script>`, `<foreignObject>`, `<image>`, external `xlink:href` / `href` to non-fragment URLs, CSS `@import`, embedded base64 rasters, animation elements (`animate*`), event handlers (`onclick`, etc.).
6. **Deterministic normalization** — strip comments; collapse insignificant whitespace; sort attributes within an element in a stable order; normalize numeric floats to 3 decimal places; prefer explicit fill/stroke over CSS classes when generating StructSVG.
7. **Bounds** — target length ≤ 4096 tokenizer tokens after processor encoding; shape/text element count ≤ 200; reject empty or near-empty drawings (< 2 drawable elements).
8. **Text** — labels use `<text>` with plain character data; no HTML inside text.
9. **Coordinates** — all geometry expressed in the root viewBox coordinate system; nested transforms allowed but StructSVG generators should prefer absolute coordinates.
10. **One sample = one complete diagram** — no multi-page SVGs, no spritesheets.

## Validation checklist (must pass)

- XML parse success
- Root is `svg` with required namespace + viewBox
- No forbidden elements/attributes
- Renders with the project rasterizer without exception
- Element/token bounds satisfied

## Why this profile

Supports the scientific split between **syntax/format competence** (valid, renderable SVG) and **structural competence** (recoverable typed scene graph), without path-soup icons or unsafe web SVG artifacts.
