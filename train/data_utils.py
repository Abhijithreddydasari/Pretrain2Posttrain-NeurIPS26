"""Dataset utilities for image→SVG SFT."""
from __future__ import annotations

import json
import os
import random
import re
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
        # Modal processed-data volume layout:
        # /vol/data/processed/... (manifest rows remain repo-relative data/processed/...).
        vol_alt = Path("/vol/data") / Path(*rel.parts[1:])
        if vol_alt.exists():
            return vol_alt
        vol_repo_alt = Path("/vol/data") / rel
        if vol_repo_alt.exists():
            return vol_repo_alt
    if _DATA_ROOT is not None:
        alt = _DATA_ROOT / path.as_posix().lstrip("/")
        if alt.exists():
            return alt
        if path.parts and path.parts[0] == "data":
            alt2 = _DATA_ROOT / Path(*path.parts[1:])
            if alt2.exists():
                return alt2
    return path


def row_has_loadable_image(row: dict) -> bool:
    """True if resolve_image would succeed (PNG on disk or renderable SVG)."""
    if row.get("image_path"):
        p = _resolve_path(row["image_path"])
        if p.is_file():
            return True
    svg = row.get("svg")
    if not svg and row.get("svg_path"):
        p = _resolve_path(row["svg_path"])
        if not p.is_file():
            return False
        svg = p.read_text(encoding="utf-8")
    if not svg:
        return False
    try:
        render_pil(svg, size=TRAIN_RENDER_LONG_EDGE)
        return True
    except Exception:
        return False


def load_manifest(
    path: Path,
    max_samples: int | None = None,
    *,
    sample_seed: int | None = None,
    require_loadable_image: bool = False,
) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if require_loadable_image:
        before = len(rows)
        rows = [r for r in rows if row_has_loadable_image(r)]
        print(
            f"load_manifest: {path.name} loadable {len(rows)}/{before}",
            flush=True,
        )

    if max_samples is not None and sample_seed is not None:
        rows.sort(key=lambda r: str(r.get("id", "")))
        rng = random.Random(sample_seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        pick = sorted(indices[:max_samples], key=lambda i: rows[i].get("id", ""))
        rows = [rows[i] for i in pick]
    elif max_samples:
        rows = rows[:max_samples]

    if max_samples is not None and sample_seed is not None:
        print(
            f"load_manifest: {path.name} selected n={len(rows)} seed={sample_seed}"
            + (f" ids={[r.get('id') for r in rows[:3]]}..." if rows else ""),
            flush=True,
        )

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
    "Reconstruct the diagram as one complete native SVG. "
    'Use viewBox="{viewbox}" and preserve the diagram\'s aspect ratio, layout, text, '
    "shapes, and connections. Output only SVG markup. End with </svg>."
)


def canvas_viewbox(row: dict, *, image: Image.Image | None = None) -> str:
    """Native gold viewBox; image dimensions only when no gold SVG exists."""
    try:
        svg = resolve_svg(row)
    except (FileNotFoundError, KeyError):
        svg = ""
    match = re.search(r'\bviewBox\s*=\s*["\']([^"\']+)["\']', svg[:4000], re.IGNORECASE)
    if match:
        parts = re.split(r"[\s,]+", match.group(1).strip())
        if len(parts) == 4:
            return " ".join(parts)
    if image is None:
        image = resolve_image(row)
    width, height = image.size
    return f"0 0 {width} {height}"


def prompt_for_row(row: dict, template: str = PROMPT, *, image: Image.Image | None = None) -> str:
    """Fill a canvas-conditioned prompt without changing target coordinates."""
    viewbox = canvas_viewbox(row, image=image)
    return template.format(viewbox=viewbox)


def build_train_example(row: dict, *, prompt: str = PROMPT) -> dict:
    """Prompt-completion record for TRL VLM SFT (completion_only_loss masks prompt)."""
    img = resolve_image(row)
    svg = resolve_svg(row)
    row_prompt = prompt_for_row(row, prompt, image=img)
    # TRL collator injects images from `images`; prompt uses placeholders only.
    return {
        "id": row.get("id"),
        "images": [img],
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": row_prompt},
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
