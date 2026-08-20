"""Stage 1: scan starvector/svg-diagrams train pool into filtered candidate index.

Three-pass pipeline for speed:
  A) cheap validate (no render) — streaming
  B) render once + phash + cache PNG — optional multiprocessing
  C) batch bucketed phash dedup — O(n·bucket_size)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.scripts.broad_io import (  # noqa: E402
    ErrorLogger,
    ProgressTracker,
    RejectionCounter,
    load_test_hashes,
    print_summary,
    repo_relative,
    resolve_asset_path,
    retry_hf,
    write_json,
)
from structsvg_lib.broad_features import (  # noqa: E402
    STRUCTURAL_FEATURE_NAMES,
    dedup_by_phash,
    extract_structural_features,
    feature_bucket,
    vfig_code_filter,
)
from structsvg_lib.svg_ops import extract_svg_blob, validate_svg

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

HF_DATASET = "starvector/svg-diagrams"
DEFAULT_TOTAL = 182_618
MAX_SVG_CHARS = 20_000
PHASH_NEAR_DUP = 3
SCAN_RENDER_SIZE = 224


def _default_workers() -> int:
    """Process pool on Linux/Modal; single worker on Windows (svglib fallback is not MP-safe)."""
    if os.name == "nt":
        return 1
    return min(8, os.cpu_count() or 4)


def _load_dataset_stream(revision: str | None):
    from datasets import load_dataset

    kwargs: dict = {"split": "train", "streaming": True}
    if revision:
        kwargs["revision"] = revision
    return retry_hf(lambda: load_dataset(HF_DATASET, **kwargs))


def _pin_revision_if_needed(revision: str | None, out_dir: Path) -> str | None:
    if revision:
        return revision
    try:
        from huggingface_hub import dataset_info

        info = dataset_info(HF_DATASET)
        rev = info.sha
        write_json(out_dir / "hf_revision.json", {"hf_id": HF_DATASET, "revision": rev})
        return rev
    except Exception:  # noqa: BLE001
        return None


def _pass_a_cheap_filter(
    ds,
    *,
    max_rows: int | None,
    test_hashes: set[str],
    counter: RejectionCounter,
    errors: ErrorLogger,
    total: int,
) -> tuple[list[dict], int]:
    """Parse + validate without render; collect staging records."""
    seen_sha: set[str] = set()
    staging: list[dict] = []
    scanned = 0
    progress = ProgressTracker(total=total, desc="pass A (cheap)", unit="row", max_updates=20)

    try:
        for row in ds:
            if max_rows and scanned >= max_rows:
                break
            scanned += 1
            filename = row.get("Filename") or f"row_{scanned}"
            try:
                raw = row.get("Svg") or row.get("svg") or ""
                svg_blob = extract_svg_blob(raw) or raw
                if not svg_blob:
                    counter.reject("no_svg")
                elif len(svg_blob) > MAX_SVG_CHARS:
                    counter.reject("too_long")
                else:
                    val = validate_svg(svg_blob, try_render=False)
                    if not val.parse_ok or not val.normalized or val.errors:
                        reason = val.errors[0] if val.errors else "validate_fail"
                        counter.reject("validate", reason)
                    elif val.sha256 in test_hashes:
                        counter.reject("test_leak")
                    elif val.sha256 in seen_sha:
                        counter.reject("exact_dup")
                    else:
                        vfig_ok, vfig_reason, vfig_m = vfig_code_filter(val.normalized)
                        if not vfig_ok:
                            counter.reject(vfig_reason or "vfig_filter")
                        else:
                            bucket = feature_bucket(val.normalized)
                            assert val.sha256 is not None
                            feat_vec, feat_named = extract_structural_features(val.normalized)
                            seen_sha.add(val.sha256)
                            rec: dict = {
                                "id": str(filename).replace(".svg", "")[:64],
                                "filename": filename,
                                "sha256": val.sha256,
                                "bucket": bucket,
                                "n_drawable": val.n_drawable,
                                "n_elements": val.n_elements,
                                "difficulty": feat_named["difficulty"],
                                "source": HF_DATASET,
                                "normalized": val.normalized,
                                **vfig_m,
                            }
                            for i, name in enumerate(STRUCTURAL_FEATURE_NAMES):
                                rec[f"feat_{name}"] = float(feat_vec[i])
                            staging.append(rec)
            except Exception as e:  # noqa: BLE001
                errors.log("scan_a", str(filename), None, type(e).__name__, str(e))
                counter.reject("exception", str(e))

            progress.tick(kept=len(staging), rejected=sum(counter.counts.values()))
    except Exception as e:  # noqa: BLE001
        progress.close()
        raise RuntimeError(f"pass A interrupted after {scanned} rows: {e}") from e

    progress.close()
    return staging, scanned


def _render_worker(task: dict) -> dict:
    """Pass B worker: single render → phash + write SVG/PNG."""
    # ProcessPoolExecutor children may not inherit repo layout; Modal sets PYTHONPATH=/root.
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    root = Path(task["root"])
    try:
        from structsvg_lib.svg_ops import perceptual_hash_from_image, render_pil

        normalized = task["normalized"]
        svg_path = root / task["svg_rel"]
        png_path = root / task["png_rel"]
        img = render_pil(normalized, size=task.get("render_size", SCAN_RENDER_SIZE))
        phash = perceptual_hash_from_image(img)
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(normalized, encoding="utf-8")
        img.save(png_path)
        return {
            **{k: v for k, v in task.items() if k not in ("normalized", "svg_store", "png_store")},
            "phash": phash,
            "render_ok": True,
            "svg_path": task["svg_store"],
            "png_path": task["png_store"],
        }
    except Exception as e:  # noqa: BLE001
        return {
            **{k: v for k, v in task.items() if k not in ("normalized", "svg_store", "png_store")},
            "phash": None,
            "render_ok": False,
            "error": str(e),
        }


def _pass_b_render(
    staging: list[dict],
    *,
    out_dir: Path,
    workers: int,
    counter: RejectionCounter,
    errors: ErrorLogger,
) -> list[dict]:
    """Render survivors once; parallel when workers > 1."""
    tasks: list[dict] = []
    for rec in staging:
        sha_key = rec["sha256"][:16]
        svg_rel = f"pool_svgs/{sha_key}.svg"
        png_rel = f"pool_pngs/{sha_key}.png"
        tasks.append(
            {
                **{k: v for k, v in rec.items() if k != "normalized"},
                "normalized": rec["normalized"],
                "root": str(out_dir),
                "svg_rel": svg_rel,
                "png_rel": png_rel,
                "svg_store": repo_relative(out_dir, svg_rel),
                "png_store": repo_relative(out_dir, png_rel),
                "render_size": SCAN_RENDER_SIZE,
            }
        )

    rendered: list[dict] = []
    total = len(tasks)
    progress = ProgressTracker(total=total, desc="pass B (render)", unit="row", max_updates=20)

    if workers <= 1:
        for task in tasks:
            out = _render_worker(task)
            if out.get("render_ok"):
                rendered.append(out)
            else:
                counter.reject("render_fail", out.get("error", ""))
                errors.log("scan_b", out.get("id", "?"), out.get("sha256"), "RenderError", out.get("error", ""))
            progress.tick(kept=len(rendered), rejected=counter.counts.get("render_fail", 0))
    else:
        executor_cls = ProcessPoolExecutor if os.name != "nt" else ThreadPoolExecutor
        logging.info("pass B: rendering with %s workers=%d", executor_cls.__name__, workers)
        with executor_cls(max_workers=workers) as pool:
            futures = {pool.submit(_render_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                out = fut.result()
                if out.get("render_ok"):
                    rendered.append(out)
                else:
                    counter.reject("render_fail", out.get("error", ""))
                    errors.log("scan_b", out.get("id", "?"), out.get("sha256"), "RenderError", out.get("error", ""))
                progress.tick(kept=len(rendered), rejected=counter.counts.get("render_fail", 0))

    progress.close()
    return rendered


def _pass_c_phash_dedup(rows: list[dict], counter: RejectionCounter) -> list[dict]:
    kept, removed = dedup_by_phash(rows, max_hamming=PHASH_NEAR_DUP)
    if removed:
        counter.counts["phash_near_dup"] += removed
    return kept


def scan_pool(
    out_dir: Path,
    *,
    max_rows: int | None = None,
    pilot: bool = False,
    seed: int = 42,
    revision: str | None = None,
    test_hashes_path: Path | None = None,
    workers: int | None = None,
) -> dict:
    if pilot:
        max_rows = max_rows or 5_000

    if workers is None:
        workers = _default_workers()

    out_dir.mkdir(parents=True, exist_ok=True)
    # Do not resolve() out_dir — Modal volume mounts symlink to /__modal/volumes/...
    svg_dir = out_dir / "pool_svgs"
    png_dir = out_dir / "pool_pngs"
    svg_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    errors = ErrorLogger(out_dir / "errors.jsonl")
    counter = RejectionCounter()

    test_hashes = load_test_hashes(test_hashes_path or out_dir / "test_hashes.jsonl")
    revision = _pin_revision_if_needed(revision, out_dir)
    ds = _load_dataset_stream(revision)

    total = max_rows or DEFAULT_TOTAL
    logging.info("pass A: cheap filter (no render)...")
    staging, scanned = _pass_a_cheap_filter(
        ds,
        max_rows=max_rows,
        test_hashes=test_hashes,
        counter=counter,
        errors=errors,
        total=total,
    )

    if not staging:
        raise RuntimeError("pool empty after pass A; check filters and dataset access")

    logging.info("pass B: render + phash (%d survivors, workers=%d)...", len(staging), workers)
    rendered = _pass_b_render(
        staging,
        out_dir=out_dir,
        workers=workers,
        counter=counter,
        errors=errors,
    )

    if not rendered:
        raise RuntimeError("pool empty after pass B render; check renderer install")

    logging.info("pass C: batch phash dedup (%d rows)...", len(rendered))
    rows = _pass_c_phash_dedup(rendered, counter)

    if not rows:
        raise RuntimeError("pool empty after phash dedup")

    pool_path = out_dir / "pool_index.parquet"
    _write_pool_index(rows, pool_path)

    bucket_hist: dict[str, int] = {}
    for r in rows:
        bucket_hist[r["bucket"]] = bucket_hist.get(r["bucket"], 0) + 1

    stats = {
        "scanned": scanned,
        "staging_after_pass_a": len(staging),
        "after_render_pass_b": len(rendered),
        "kept": len(rows),
        "rejected": sum(counter.counts.values()),
        "rejected_by_reason": counter.as_dict(),
        "bucket_histogram": bucket_hist,
        "hf_id": HF_DATASET,
        "revision": revision,
        "seed": seed,
        "max_rows": max_rows,
        "pilot": pilot,
        "workers": workers,
        "errors_logged": errors.count,
        "pipeline": "three_pass",
    }
    write_json(out_dir / "scan_stats.json", stats)
    return stats


def _write_pool_index(rows: list[dict], path: Path) -> None:
    import pandas as pd

    slim = []
    for r in rows:
        slim.append({k: v for k, v in r.items() if k not in ("normalized", "root", "svg_rel", "png_rel", "svg_store", "png_store", "render_size", "render_ok", "error")})
    df = pd.DataFrame(slim)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_pool_index(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def structural_matrix(df) -> np.ndarray:
    cols = [f"feat_{n}" for n in STRUCTURAL_FEATURE_NAMES]
    return df[cols].to_numpy(dtype=np.float32)


def load_pool_png(row: dict, *, out_dir: Path | None = None, render_size: int = 224):
    """Load cached pool PNG only — never re-render SVG during embed."""
    from PIL import Image

    candidates: list[Path] = []
    if row.get("png_path"):
        candidates.append(resolve_asset_path(row["png_path"]))
    if out_dir is not None and row.get("sha256"):
        candidates.append(out_dir / "pool_pngs" / f"{row['sha256'][:16]}.png")

    for p in candidates:
        if p.is_file():
            with Image.open(p) as im:
                im.load()
                img = im.convert("RGB") if im.mode != "RGB" else im.copy()
            if img.size != (render_size, render_size):
                img = img.resize((render_size, render_size), Image.Resampling.LANCZOS)
            return img
    raise FileNotFoundError(f"pool PNG missing for {row.get('id', '?')} (checked {candidates})")


def resolve_pool_image(row: dict, render_size: int = 224, *, out_dir: Path | None = None):
    """Load cached pool PNG or fall back to SVG render (scan/visualize only)."""
    from structsvg_lib.svg_ops import render_pil

    try:
        return load_pool_png(row, out_dir=out_dir, render_size=render_size)
    except FileNotFoundError:
        pass
    svg_path = resolve_asset_path(row["svg_path"])
    return render_pil(svg_path.read_text(encoding="utf-8"), size=render_size)


def main():
    ap = argparse.ArgumentParser(description="Scan SVG-Diagrams train into filtered pool")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "broad")
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--test-dedup", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None, help="pass B render workers (default: 8 on Linux, 1 on Windows)")
    args = ap.parse_args()

    stats = scan_pool(
        args.out,
        max_rows=args.max_rows,
        pilot=args.pilot,
        seed=args.seed,
        revision=args.revision,
        test_hashes_path=args.test_dedup,
        workers=args.workers,
    )
    print_summary(
        "scan",
        scanned=stats["scanned"],
        kept=stats["kept"],
        rejected=stats["rejected"],
        errors_logged=stats["errors_logged"],
        workers=stats["workers"],
    )
    print(f"wrote {args.out / 'pool_index.parquet'} n={stats['kept']}")


if __name__ == "__main__":
    main()
