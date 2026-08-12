"""pytest for svg validate + gold recovery."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structsvg_lib.metrics import area_between, emergence_times
from structsvg_lib.svg_ops import validate_svg


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
