"""Tests for broad SVG feature extraction and pipeline helpers."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.scripts.broad_io import ErrorLogger, RejectionCounter, tqdm_enabled  # noqa: E402
from structsvg_lib.broad_features import (  # noqa: E402
    STRUCTURAL_FEATURE_NAMES,
    difficulty_score,
    extract_structural_features,
    feature_bucket,
    phash_hamming,
)

MINIMAL_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
<rect x="10" y="10" width="40" height="20" fill="#fff" stroke="#000"/>
<text x="15" y="25">A</text>
</svg>'''

WORKFLOW_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
<rect x="1" y="1" width="10" height="10"/><rect x="20" y="1" width="10" height="10"/>
<line x1="5" y1="5" x2="25" y2="5"/><text x="2" y="8">A</text>
</svg>'''


def test_feature_bucket_workflow():
    assert feature_bucket(WORKFLOW_SVG) == "workflow_like"


def test_structural_features_shape_stable():
    vec, named = extract_structural_features(MINIMAL_SVG)
    assert vec.shape == (len(STRUCTURAL_FEATURE_NAMES),)
    assert vec.dtype == np.float32
    assert named["n_rect"] >= 1
    assert named["difficulty"] == difficulty_score(named)


def test_phash_hamming_identical():
    h = "a" * 16
    assert phash_hamming(h, h) == 0
    assert phash_hamming(None, h) == 999


def test_rejection_counter():
    c = RejectionCounter()
    c.reject("parse", "bad xml")
    c.reject("parse", "another")
    assert c.counts["parse"] == 2


def test_error_logger(tmp_path: Path):
    path = tmp_path / "errors.jsonl"
    el = ErrorLogger(path)
    el.log("test", "id1", "sha", "ValueError", "msg")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["stage"] == "test"
    assert el.count == 1


def test_tqdm_disabled_in_pytest():
    os.environ["BROAD_TQDM"] = "0"
    assert tqdm_enabled() is False
    os.environ["BROAD_TQDM"] = "1"


def test_preprocess_named_colors():
    from structsvg_lib.svg_ops import preprocess_svg_for_render

    svg = '<svg><rect fill="lightgray" stroke="inherit"/></svg>'
    out = preprocess_svg_for_render(svg)
    assert 'fill="#d3d3d3"' in out
    assert 'stroke="none"' in out


def test_dedup_by_phash_bucketed():
    from structsvg_lib.broad_features import dedup_by_phash

    rows = [
        {"id": "a", "phash": "0000000000000000"},
        {"id": "b", "phash": "0000000000000001"},
        {"id": "c", "phash": "ffffffffffffffff"},
    ]
    kept, removed = dedup_by_phash(rows, max_hamming=3)
    assert removed == 1
    assert len(kept) == 2
    ids = {r["id"] for r in kept}
    assert ids == {"a", "c"}


def test_medoid_pick_deterministic():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 8)).astype(np.float32)
    center = x.mean(axis=0)
    dists = np.linalg.norm(x - center, axis=1)
    pick_a = int(np.argmin(dists))
    pick_b = int(np.argmin(dists))
    assert pick_a == pick_b
