"""Shared package root helpers."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)
