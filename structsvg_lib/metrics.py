"""Metric helpers for checkpoint emergence and bootstrap intervals."""
from __future__ import annotations

import math
import random
from typing import Iterable


def mean(xs: Iterable[float | None]) -> float:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(vals) / len(vals) if vals else float("nan")


def bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = random.Random(seed)
    means = []
    n = len(vals)
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return {"mean": sum(vals) / n, "lo": lo, "hi": hi}


def normalized_gain(series: list[float], eps: float = 1e-8) -> list[float]:
    if not series:
        return []
    m0, m1 = series[0], series[-1]
    denom = m1 - m0
    if abs(denom) < eps:
        return [0.0 for _ in series]
    return [(m - m0) / denom for m in series]


def emergence_times(pcts: list[float], series: list[float]) -> dict[str, float | None]:
    gains = normalized_gain(series)

    def first_at(thr: float) -> float | None:
        for p, g in zip(pcts, gains):
            if g >= thr:
                return p
        return None

    return {"t50": first_at(0.5), "t90": first_at(0.9)}


def area_between(syntax: list[float], structure: list[float], pcts: list[float]) -> float:
    """Trapezoid area of (syntax_gain - structure_gain) over pct axis."""
    sg = normalized_gain(syntax)
    st = normalized_gain(structure)
    if len(pcts) < 2:
        return 0.0
    area = 0.0
    for i in range(len(pcts) - 1):
        dx = pcts[i + 1] - pcts[i]
        y0 = sg[i] - st[i]
        y1 = sg[i + 1] - st[i + 1]
        area += 0.5 * (y0 + y1) * dx
    return area
