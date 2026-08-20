"""Structural features and bucketing for broad SVG-Diagrams coreset selection."""
from __future__ import annotations

import math
import re
from xml.etree import ElementTree as ET

import numpy as np

from structsvg_lib.svg_ops import _local, parse_svg

FEATURE_TAGS = [
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "text",
    "g",
]

STRUCTURAL_FEATURE_NAMES = [
    *[f"n_{t}" for t in FEATURE_TAGS],
    "n_elements",
    "max_depth",
    "n_colors",
    "text_chars",
    "path_segments",
    "aspect_ratio",
    "n_groups",
    "n_markers",
    "difficulty",
]

# VFIG (arXiv:2603.24575) SVG code filter — replaces tag-count path_soup rejection.
VFIG_MIN_CLEAN = 0.40  # (B + K) / N
VFIG_MAX_COMPLEX = 50  # max path + polygon count

_VFIG_BASIC = frozenset({"rect", "circle", "ellipse"})
_VFIG_CONNECTOR = frozenset({"line", "polyline"})
_VFIG_COMPLEX = frozenset({"path", "polygon"})


def count_vfig_elements(root: ET.Element) -> dict[str, int]:
    """Count VFIG element groups: B (basic), K (connector), C (complex), T (text)."""
    counts = {"B": 0, "K": 0, "C": 0, "T": 0}
    for node in root.iter():
        tag = _local(node.tag)
        if tag in _VFIG_BASIC:
            counts["B"] += 1
        elif tag in _VFIG_CONNECTOR:
            counts["K"] += 1
        elif tag in _VFIG_COMPLEX:
            counts["C"] += 1
        elif tag == "text":
            counts["T"] += 1
    return counts


def vfig_metrics(counts: dict[str, int]) -> dict[str, float]:
    """Derived VFIG stats: N, Clean, PD, EC (log-scaled element complexity)."""
    b, k, c, t = counts["B"], counts["K"], counts["C"], counts["T"]
    n = b + k + c
    clean = (b + k) / n if n > 0 else 0.0
    return {
        "vfig_B": float(b),
        "vfig_K": float(k),
        "vfig_C": float(c),
        "vfig_T": float(t),
        "vfig_N": float(n),
        "vfig_clean": float(clean),
        "vfig_pd": float(c / n if n > 0 else 0.0),
        "vfig_ec": float(math.log1p(n + t)),
    }


def vfig_code_filter(svg: str) -> tuple[bool, str | None, dict[str, float]]:
    """VFIG code filter: Clean >= 0.40 and C <= 50 (N = B+K+C > 0).

    Returns (pass, rejection_reason, metrics_dict).
    """
    root, err = parse_svg(svg)
    if err or root is None:
        return False, "vfig_parse", {}
    metrics = vfig_metrics(count_vfig_elements(root))
    n = int(metrics["vfig_N"])
    c = int(metrics["vfig_C"])
    if n == 0:
        return False, "vfig_no_geometry", metrics
    if metrics["vfig_clean"] < VFIG_MIN_CLEAN:
        return False, "vfig_low_clean", metrics
    if c > VFIG_MAX_COMPLEX:
        return False, "vfig_too_many_complex", metrics
    return True, None, metrics


def feature_bucket(svg: str) -> str:
    """Crude structural bucket for stratification / reporting."""
    low = svg.lower()
    n_text = low.count("<text")
    n_rect = low.count("<rect")
    n_line = low.count("<line") + low.count("<polyline")
    n_path = low.count("<path")
    if n_rect >= 2 and n_text >= 1 and n_line >= 1:
        return "workflow_like"
    if n_line >= 3 and n_text <= 3 and n_path <= 2:
        return "geometry_like"
    if n_text >= 1:
        return "labeled"
    return "other"


def phash_hamming(a: str | None, b: str | None) -> int:
    """Hamming distance between two 16-char hex perceptual hashes."""
    if not a or not b or len(a) != len(b):
        return 999
    ai = int(a, 16)
    bi = int(b, 16)
    return (ai ^ bi).bit_count()


def _path_segment_estimate(d: str | None) -> int:
    if not d:
        return 0
    return max(1, len(re.findall(r"[MLHVCSQTAZmlhvcsqtaz]", d)))


def _parse_viewbox_aspect(root: ET.Element) -> float:
    vb = root.get("viewBox") or root.get("viewbox")
    if vb:
        parts = re.split(r"[\s,]+", vb.strip())
        if len(parts) >= 4:
            try:
                w = float(parts[2])
                h = float(parts[3])
                if h > 0:
                    return w / h
            except ValueError:
                pass
    try:
        w = float(re.sub(r"[^\d.]", "", root.get("width", "1") or "1") or "1")
        h = float(re.sub(r"[^\d.]", "", root.get("height", "1") or "1") or "1")
        if h > 0:
            return w / h
    except ValueError:
        pass
    return 1.0


def extract_structural_features(svg: str) -> tuple[np.ndarray, dict[str, float]]:
    """Return (feature_vector, named_dict) for one normalized SVG."""
    root, err = parse_svg(svg)
    if err or root is None:
        raise ValueError(err or "parse failed")

    counts = {t: 0 for t in FEATURE_TAGS}
    depths: list[int] = []
    colors: set[str] = set()
    text_chars = 0
    path_segments = 0
    n_groups = 0
    n_markers = 0
    n_elements = 0

    def walk(node: ET.Element, depth: int = 0) -> None:
        nonlocal text_chars, path_segments, n_groups, n_markers, n_elements
        n_elements += 1
        depths.append(depth)
        tag = _local(node.tag)
        if tag in counts:
            counts[tag] += 1
        if tag == "g":
            n_groups += 1
        if tag == "marker":
            n_markers += 1
        if tag == "path":
            path_segments += _path_segment_estimate(node.get("d"))
        if tag == "text" and node.text:
            text_chars += len(node.text.strip())
        for attr in ("fill", "stroke"):
            val = node.get(attr)
            if val and val not in ("none", "transparent"):
                colors.add(val)
        for child in node:
            walk(child, depth + 1)

    walk(root)
    aspect = _parse_viewbox_aspect(root)

    named = {
        **{f"n_{t}": float(counts[t]) for t in FEATURE_TAGS},
        "n_elements": float(n_elements),
        "max_depth": float(max(depths) if depths else 0),
        "n_colors": float(len(colors)),
        "text_chars": float(text_chars),
        "path_segments": float(path_segments),
        "aspect_ratio": float(aspect),
        "n_groups": float(n_groups),
        "n_markers": float(n_markers),
    }
    named["difficulty"] = difficulty_score(named)
    vec = np.array([named[k] for k in STRUCTURAL_FEATURE_NAMES], dtype=np.float32)
    return vec, named


def difficulty_score(features: dict[str, float]) -> float:
    """Scalar difficulty from structural stats (higher = more complex)."""
    return (
        0.25 * features.get("n_elements", 0)
        + 0.20 * features.get("max_depth", 0)
        + 0.15 * features.get("path_segments", 0)
        + 0.15 * features.get("text_chars", 0) ** 0.5
        + 0.10 * features.get("n_groups", 0)
        + 0.10 * features.get("n_colors", 0)
        + 0.05 * features.get("n_markers", 0)
    )


def dedup_by_phash(
    rows: list[dict],
    *,
    phash_field: str = "phash",
    max_hamming: int = 3,
    prefix_len: int = 4,
) -> tuple[list[dict], int]:
    """Batch near-duplicate removal via prefix buckets — O(n·b) not O(n²)."""
    from collections import defaultdict

    buckets: dict[str, list[str]] = defaultdict(list)
    kept: list[dict] = []
    removed = 0

    for row in rows:
        ph = row.get(phash_field)
        if not ph:
            kept.append(row)
            continue
        prefix = ph[:prefix_len]
        is_dup = False
        for existing in buckets[prefix]:
            if phash_hamming(ph, existing) <= max_hamming:
                is_dup = True
                break
        if is_dup:
            removed += 1
            continue
        kept.append(row)
        buckets[prefix].append(ph)

    return kept, removed
