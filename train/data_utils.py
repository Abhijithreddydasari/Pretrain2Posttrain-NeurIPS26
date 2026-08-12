"""Dataset utilities for image→SVG SFT."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def load_manifest(path: Path, max_samples: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def resolve_image(row: dict) -> Image.Image:
    if row.get("image_path") and Path(row["image_path"]).exists():
        return Image.open(row["image_path"]).convert("RGB")
    # render from svg if needed
    from structsvg_lib.svg_ops import render_pil

    svg = row.get("svg")
    if not svg and row.get("svg_path"):
        svg = Path(row["svg_path"]).read_text(encoding="utf-8")
    if not svg:
        raise FileNotFoundError(f"no image/svg for {row.get('id')}")
    return render_pil(svg, size=448)


def resolve_svg(row: dict) -> str:
    if row.get("svg"):
        return row["svg"]
    return Path(row["svg_path"]).read_text(encoding="utf-8")


PROMPT = (
    "Reconstruct the diagram as a single canonical SVG. "
    'Use viewBox="0 0 512 512". Output only SVG markup.'
)


def build_messages(image: Image.Image, svg: str, prompt: str = PROMPT) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": svg}],
        },
    ]
