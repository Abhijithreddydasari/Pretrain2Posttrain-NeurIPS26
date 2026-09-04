"""Tests for SVG validation, rendering, and checkpoint metrics."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structsvg_lib.metrics import area_between, emergence_times
from structsvg_lib.svg_ops import preprocess_svg_for_render, render_pil, validate_svg
from train.vllm_infer import recover_svg_prefix


def test_validate_minimal_svg():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
    <rect x="10" y="10" width="40" height="20" fill="#fff" stroke="#000"/>
    <text x="15" y="25">A</text>
    </svg>'''
    v = validate_svg(svg, try_render=False)
    assert v.parse_ok
    assert v.n_drawable >= 2
    assert v.sha256


def test_forbidden_script():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
    <rect x="1" y="1" width="2" height="2"/>
    <text x="1" y="1">a</text>
    <script>alert(1)</script>
    </svg>'''
    v = validate_svg(svg, try_render=False)
    assert any("forbidden" in e for e in v.errors)


def test_emergence():
    pcts = [0, 50, 100]
    s = [0.0, 1.0, 1.0]
    assert emergence_times(pcts, s)["t50"] == 50
    assert area_between([0, 1, 1], [0, 0, 1], pcts) > 0


def test_graphviz_text_renders_on_white():
    """Graphviz SVGs: black text on transparent bg must not disappear after rasterize."""
    svg = preprocess_svg_for_render(
        '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 80" width="200" height="80">
        <polygon fill="none" points="10,10 90,10 90,50 10,50" stroke="#000"/>
        <text fill="#000000" font-family="FreeSans" font-size="12" x="50" y="35" text-anchor="middle">NodeA</text>
        </svg>'''
    )
    img = render_pil(svg, size=128)
    px = img.load()
    dark = sum(1 for y in range(img.size[1]) for x in range(img.size[0]) if sum(px[x, y]) < 700)
    assert dark > 50, f"expected visible text strokes, got {dark} dark pixels"


def test_recover_unterminated_svg_closes_open_elements():
    raw = '<svg viewBox="0 0 10 10"><g><rect width="4" height="4"/>'
    recovered, reason = recover_svg_prefix(raw)
    assert reason == "closed_open_tags"
    assert recovered == raw + "</g></svg>"
    assert validate_svg(recovered, try_render=False).parse_ok


def test_recover_svg_discards_repeated_incomplete_tail():
    raw = '<svg viewBox="0 0 10 10"><rect width="4" height="4"/><rect width="'
    recovered, reason = recover_svg_prefix(raw)
    assert reason == "closed_open_tags"
    assert recovered == '<svg viewBox="0 0 10 10"><rect width="4" height="4"/></svg>'


def test_recover_svg_does_not_invent_root():
    assert recover_svg_prefix('<rect width="4"/>') == (None, None)


def test_recover_svg_trims_exact_repetition_after_first_copy():
    tag = '<path d="M 0 0 L 10 10" stroke="black"/>'
    raw = '<svg viewBox="0 0 10 10">' + tag * 3
    recovered, reason = recover_svg_prefix(raw)
    assert reason == "trimmed_exact_repetition_and_closed_open_tags"
    assert recovered == '<svg viewBox="0 0 10 10">' + tag + "</svg>"
