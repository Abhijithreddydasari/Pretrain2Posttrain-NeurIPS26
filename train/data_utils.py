"""Dataset utilities for image→SVG SFT."""
from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

from structsvg_lib.svg_ops import TRAIN_RENDER_LONG_EDGE, render_pil

# Optional prefix when manifests use repo-relative paths on Modal volume.
_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "")).expanduser() if os.environ.get("DATA_ROOT") else None
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROCESSED_MARKER = "data/processed/"


def _repo_relative_processed_path(p: str | Path) -> Path | None:
    """Map any manifest path (repo-relative or Windows absolute) to data/processed/..."""
    s = str(p).replace("\\", "/")
    if _PROCESSED_MARKER in s:
        rel = s.split(_PROCESSED_MARKER, 1)[1]
        return Path("data") / "processed" / rel
    path = Path(p)
    if path.parts and path.parts[0] == "data":
        return path
    return None


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.exists():
        return path
    rel = _repo_relative_processed_path(p)
    if rel is not None:
        if _DATA_ROOT is not None:
            alt = _DATA_ROOT / Path(*rel.parts[1:])  # processed/...
            if alt.exists():
                return alt
            alt2 = _DATA_ROOT / rel.as_posix().lstrip("/")
            if alt2.exists():
                return alt2
        alt3 = _REPO_ROOT / rel
        if alt3.exists():
            return alt3
    if _DATA_ROOT is not None:
        alt = _DATA_ROOT / path.as_posix().lstrip("/")
        if alt.exists():
            return alt
        if path.parts and path.parts[0] == "data":
            alt2 = _DATA_ROOT / Path(*path.parts[1:])
            if alt2.exists():
                return alt2
    return path


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
    if row.get("image_path"):
        p = _resolve_path(row["image_path"])
        if p.exists():
            return Image.open(p).convert("RGB")
    # render from svg if needed
    svg = row.get("svg")
    if not svg and row.get("svg_path"):
        svg = _resolve_path(row["svg_path"]).read_text(encoding="utf-8")
    if not svg:
        raise FileNotFoundError(f"no image/svg for {row.get('id')}")
    return render_pil(svg, size=TRAIN_RENDER_LONG_EDGE)


def resolve_svg(row: dict) -> str:
    if row.get("svg"):
        return row["svg"]
    return _resolve_path(row["svg_path"]).read_text(encoding="utf-8")


PROMPT = (
    "Reconstruct the diagram as a single canonical SVG. "
    'Use viewBox="0 0 512 512". Output only SVG markup.'
)


def build_train_example(row: dict, *, prompt: str = PROMPT) -> dict:
    """Prompt-completion record for TRL VLM SFT (completion_only_loss masks prompt)."""
    img = resolve_image(row)
    svg = resolve_svg(row)
    # TRL collator injects images from `images`; prompt uses placeholders only.
    return {
        "id": row.get("id"),
        "images": [img],
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "completion": [{"role": "assistant", "content": [{"type": "text", "text": svg}]}],
    }


def longest_rows(rows: list[dict], n: int) -> list[dict]:
    """Return n rows with longest SVG text (worst-case VRAM for probe)."""
    return sorted(rows, key=lambda r: len(resolve_svg(r)), reverse=True)[:n]


def materialize_train_examples(
    rows: list[dict],
    *,
    prompt: str = PROMPT,
    log_fn=None,
    log_every: int = 100,
) -> list[dict]:
    """Load every PNG+SVG once into RAM; reused across all epochs (no per-epoch disk reload)."""
    n = len(rows)
    out: list[dict] = []
    svg_chars = 0
    img_bytes = 0
    for i, row in enumerate(rows):
        if log_fn and (i == 0 or (i + 1) % log_every == 0 or i + 1 == n):
            log_fn(f"caching examples {i + 1}/{n} (disk → RAM, once per run)")
        ex = build_train_example(row, prompt=prompt)
        svg_chars += len(ex["completion"][0]["content"][0]["text"])
        img_bytes += len(ex["images"][0].tobytes())
        out.append(ex)
    if log_fn and n:
        ram_mb = (svg_chars + img_bytes) / (1024 * 1024)
        log_fn(
            f"RAM cache ~{ram_mb:.0f} MiB ({n} rows: {svg_chars / 1024:.0f} KiB SVG text + "
            f"{img_bytes / (1024 * 1024):.0f} MiB decoded PNGs; disk pngs+svgs ~75 MiB compressed)"
        )
    return out
