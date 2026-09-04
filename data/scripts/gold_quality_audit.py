"""Audit degenerate patterns in gold SVG targets (nested <g> chains, style blobs)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.data_utils import load_manifest, resolve_svg


def max_g_run(svg: str) -> int:
    """Longest run of consecutive '<g ...>' or '<g>' opens with nothing between."""
    runs = re.findall(r"(?:<g>){2,}", svg)
    return max((len(r) // 3 for r in runs), default=0)


def main():
    man = ROOT / "data/processed/svg_diagrams/train_manifest.jsonl"
    rows = load_manifest(man, 2000)
    g_runs = []
    style_chars = []
    has_style = 0
    for r in rows:
        try:
            svg = resolve_svg(r)
        except Exception:
            continue
        g_runs.append(max_g_run(svg))
        st = re.findall(r"<style[^>]*>([\s\S]*?)</style>", svg)
        if st:
            has_style += 1
            style_chars.append(sum(len(s) for s in st))
        else:
            style_chars.append(0)

    g_runs.sort()
    style_chars.sort()
    n = len(g_runs)
    print(f"n={n}")
    print(f"max consecutive <g> run: p50={g_runs[n//2]} p90={g_runs[9*n//10]} p99={g_runs[int(n*0.99)]} max={g_runs[-1]}")
    print(f"examples with >=5 nested <g>: {sum(1 for x in g_runs if x >= 5)}")
    print(f"examples with >=20 nested <g>: {sum(1 for x in g_runs if x >= 20)}")
    print(f"examples with <style> block: {has_style}/{n}")
    print(f"style chars: p50={style_chars[n//2]} p90={style_chars[9*n//10]} max={style_chars[-1]}")


if __name__ == "__main__":
    main()
