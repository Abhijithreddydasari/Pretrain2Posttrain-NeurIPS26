"""Shared I/O, progress bars, and exception helpers for broad SVG pipeline."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

T = TypeVar("T")


def repo_relative(out_dir: Path, rel: str) -> str:
    """Repo-relative posix path for manifests/parquet (Modal-safe: no resolve())."""
    p = out_dir / rel
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return f"data/processed/broad/{rel}".replace("\\", "/")


def resolve_asset_path(path: str | Path, *, root: Path | None = None) -> Path:
    """Resolve manifest/parquet asset path (repo-relative or absolute)."""
    p = Path(path)
    if p.is_file():
        return p
    base = root or ROOT
    cand = base / p
    if cand.is_file():
        return cand
    if p.is_absolute() and p.exists():
        return p
    return cand


def tqdm_enabled() -> bool:
    if os.environ.get("BROAD_TQDM", "1").strip() in ("0", "false", "False"):
        return False
    return sys.stderr.isatty() or os.environ.get("BROAD_TQDM", "1") == "1"


def progress_bar(
    iterable=None,
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    initial: int = 0,
    max_updates: int = 20,
    **kwargs,
):
    """tqdm wrapper; refreshes at most ~max_updates times for large totals."""
    miniters = 1
    if total and total > 0:
        miniters = max(1, total // max_updates)
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        initial=initial,
        miniters=miniters,
        mininterval=0.3,
        disable=not tqdm_enabled(),
        **kwargs,
    )


class ProgressTracker:
    """Manual counter with throttled postfix updates (~max_updates refreshes)."""

    def __init__(self, total: int, desc: str, unit: str = "row", max_updates: int = 20) -> None:
        self._step = max(1, total // max_updates) if total > 0 else 1
        self._bar = progress_bar(total=total, desc=desc, unit=unit, max_updates=max_updates)
        self._n = 0

    def tick(self, *, kept: int = 0, rejected: int = 0, **extra: Any) -> None:
        self._n += 1
        self._bar.update(1)
        if self._n % self._step == 0 or self._n >= (self._bar.total or self._n):
            postfix = {"kept": kept, "rejected": rejected, **extra}
            self._bar.set_postfix(**postfix)

    def close(self) -> None:
        self._bar.close()

    @property
    def n(self) -> int:
        return self._n


class RejectionCounter:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self._logged: Counter[str] = Counter()
        self._log_limit = 20

    def reject(self, reason: str, detail: str = "") -> None:
        self.counts[reason] += 1
        if self._logged[reason] < self._log_limit:
            logger.warning("rejected [%s]: %s", reason, detail[:200] if detail else "")
            self._logged[reason] += 1

    def as_dict(self) -> dict[str, int]:
        return dict(self.counts)


class ErrorLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def log(self, stage: str, row_id: str, sha256: str | None, error_type: str, message: str) -> None:
        rec = {
            "stage": stage,
            "row_id": row_id,
            "sha256": sha256,
            "error_type": error_type,
            "message": message[:500],
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.count += 1


def print_summary(stage: str, *, scanned: int = 0, kept: int = 0, rejected: int = 0, errors_logged: int = 0, **extra: Any) -> None:
    parts = [f"[{stage}] scanned={scanned}", f"kept={kept}", f"rejected={rejected}", f"errors_logged={errors_logged}"]
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts))


def retry_hf(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 2.0) -> T:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if i < attempts - 1:
                delay = base_delay * (2**i)
                logger.warning("HF retry %d/%d after %s: %s", i + 1, attempts, delay, e)
                time.sleep(delay)
    raise RuntimeError(f"HF operation failed after {attempts} attempts: {last_exc}") from last_exc


def load_test_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    if not path.exists():
        return hashes
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        hashes.add(json.loads(line)["sha256"])
    return hashes


def load_jsonl_ids(path: Path, key: str = "id") -> list[str]:
    ids: list[str] = []
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(json.loads(line)[key])
    return ids


def configure_broad_logging() -> None:
    """Quiet third-party noise; keep stage summaries readable."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in ("httpx", "httpcore", "filelock", "datasets", "urllib3", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def safe_row(stage: str, row_id: str, sha256: str | None, fn: Callable[[], T], errors: ErrorLogger, counter: RejectionCounter | None = None, reject_reason: str = "exception") -> T | None:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        errors.log(stage, row_id, sha256, type(e).__name__, str(e))
        if counter:
            counter.reject(reject_reason, str(e))
        return None
