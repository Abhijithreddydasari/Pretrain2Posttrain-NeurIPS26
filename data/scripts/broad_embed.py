"""Stage 2: SigLIP visual embeddings for broad pool candidates."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.scripts.broad_io import ErrorLogger, print_summary, progress_bar, write_json  # noqa: E402
from data.scripts.broad_scan_pool import load_pool_index, load_pool_png  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

SIGLIP_MODEL = "google/siglip-base-patch16-224"
DEFAULT_BATCH = 64
RENDER_SIZE = 224
DEFAULT_IO_WORKERS = min(16, (os.cpu_count() or 4) * 2)


def _shard_dir(out_dir: Path) -> Path:
    d = out_dir / "embeddings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _completed_shards(shard_dir: Path) -> int:
    return len(list(shard_dir.glob("embeddings_shard_*.npy")))


def _load_model(device: str):
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model = AutoModel.from_pretrained(SIGLIP_MODEL).to(device)
    model.eval()
    return processor, model, torch


def _clear_embedding_cache(out_dir: Path) -> None:
    emb_dir = out_dir / "embeddings"
    if not emb_dir.exists():
        return
    for pattern in ("embeddings_shard_*.npy", "embeddings_shard_*.json"):
        for path in emb_dir.glob(pattern):
            path.unlink()
    visual = emb_dir / "visual_fp16.npy"
    if visual.exists():
        visual.unlink()


def _existing_embedding_rows(out_dir: Path) -> int | None:
    visual_path = out_dir / "embeddings" / "visual_fp16.npy"
    meta_path = out_dir / "embed_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "rows" in meta:
                return int(meta["rows"])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if visual_path.exists():
        return int(np.load(visual_path).shape[0])
    shard_dir = _shard_dir(out_dir)
    if not any(shard_dir.glob("embeddings_shard_*.npy")):
        return None
    max_idx = -1
    for meta_path in shard_dir.glob("embeddings_shard_*.json"):
        try:
            indices = json.loads(meta_path.read_text(encoding="utf-8"))["indices"]
            if indices:
                max_idx = max(max_idx, max(indices))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return max_idx + 1 if max_idx >= 0 else 0


def _load_one_png(df, idx: int, *, out_dir: Path, render_size: int, errors: ErrorLogger):
    r = df.iloc[idx]
    try:
        return idx, load_pool_png(r.to_dict(), out_dir=out_dir, render_size=render_size)
    except Exception as e:  # noqa: BLE001
        errors.log("embed", str(r.get("id", idx)), r.get("sha256"), type(e).__name__, str(e))
        return idx, None


def _load_batch(
    df,
    batch_indices: list[int],
    *,
    out_dir: Path,
    render_size: int,
    errors: ErrorLogger,
    workers: int,
) -> tuple[list[object], list[int]]:
    """Load one embed batch; returns (images, valid_indices) in index order."""
    if not batch_indices:
        return [], []
    if len(batch_indices) == 1 or workers <= 1:
        pairs = [_load_one_png(df, i, out_dir=out_dir, render_size=render_size, errors=errors) for i in batch_indices]
    else:
        pairs = [None] * len(batch_indices)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(_load_one_png, df, idx, out_dir=out_dir, render_size=render_size, errors=errors)
                for idx in batch_indices
            ]
            for j, fut in enumerate(futs):
                pairs[j] = fut.result()
        pairs = [p for p in pairs if p is not None]

    images: list[object] = []
    valid_indices: list[int] = []
    for idx, img in pairs:
        if img is not None:
            valid_indices.append(idx)
            images.append(img)
    return images, valid_indices


class _BatchPrefetcher:
    """Overlap PNG reads for batch N+1 while GPU embeds batch N (one batch in flight)."""

    def __init__(
        self,
        df,
        *,
        out_dir: Path,
        render_size: int,
        errors: ErrorLogger,
        workers: int,
    ) -> None:
        self._df = df
        self._out_dir = out_dir
        self._render_size = render_size
        self._errors = errors
        self._workers = workers
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._pending: Future | None = None

    def submit(self, batch_indices: list[int]) -> None:
        self._pending = self._pool.submit(
            _load_batch,
            self._df,
            batch_indices,
            out_dir=self._out_dir,
            render_size=self._render_size,
            errors=self._errors,
            workers=self._workers,
        )

    def result(self) -> tuple[list[object], list[int]]:
        if self._pending is None:
            return [], []
        images, valid = self._pending.result()
        self._pending = None
        return images, valid

    def close(self) -> None:
        self._pool.shutdown(wait=False)


def embed_pool(
    out_dir: Path,
    *,
    batch_size: int = DEFAULT_BATCH,
    pilot: bool = False,
    device: str | None = None,
    fresh: bool = False,
    preload: bool | None = None,
    io_workers: int | None = None,
) -> dict:
    pool_path = out_dir / "pool_index.parquet"
    if not pool_path.exists():
        raise FileNotFoundError(f"missing {pool_path}; run broad_scan_pool first")

    df = load_pool_index(pool_path)
    pool_n = len(df)
    existing_rows = _existing_embedding_rows(out_dir)
    if fresh or (existing_rows is not None and existing_rows != pool_n):
        if existing_rows is not None and existing_rows != pool_n:
            logging.warning(
                "pool has %d rows but embeddings cover %d; clearing stale embedding cache",
                pool_n,
                existing_rows,
            )
        _clear_embedding_cache(out_dir)

    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    processor, model, torch_mod = _load_model(device)
    model_load_s = time.perf_counter() - t0

    shard_dir = _shard_dir(out_dir)
    errors = ErrorLogger(out_dir / "errors.jsonl")
    n = len(df)
    bs = batch_size
    shard_idx = _completed_shards(shard_dir)
    start_row = shard_idx * bs

    io_workers = io_workers or DEFAULT_IO_WORKERS
    do_preload = preload is True  # off by default — stream batches (O(batch) RAM, scales to full pool)

    image_cache = None
    preload_s = 0.0
    if do_preload and start_row < n:
        t_pre = time.perf_counter()
        logging.warning(
            "preload loads all %d PNGs into RAM (~%.0f MB); prefer default stream mode for full pool",
            n,
            n * RENDER_SIZE * RENDER_SIZE * 3 / 1e6,
        )
        cache: list = [None] * n
        for row in range(0, n, bs):
            batch_indices = list(range(row, min(row + bs, n)))
            imgs, valid = _load_batch(
                df, batch_indices, out_dir=out_dir, render_size=RENDER_SIZE, errors=errors, workers=io_workers
            )
            for idx, img in zip(valid, imgs):
                cache[idx] = img
        image_cache = cache
        preload_s = time.perf_counter() - t_pre
    elif start_row < n:
        logging.info("streaming pool PNGs with batch prefetch (%d io workers, n=%d)", io_workers, n)

    if start_row >= n:
        visual = _concat_shards(shard_dir, n)
        _write_final(visual, out_dir, n, bs, device)
        return {"rows": n, "device": device, "resumed": True}

    n_batches = (n + bs - 1) // bs
    bar = progress_bar(total=n_batches, desc="embedding", unit="batch", initial=shard_idx)

    row = start_row
    batch_i = shard_idx
    infer_start = time.perf_counter()
    prefetcher: _BatchPrefetcher | None = None

    with torch_mod.inference_mode():
        if image_cache is None and start_row < n:
            prefetcher = _BatchPrefetcher(
                df, out_dir=out_dir, render_size=RENDER_SIZE, errors=errors, workers=io_workers
            )
            # Prime pipeline: start loading first batch before model warm-up completes.
            first_end = min(start_row + bs, n)
            prefetcher.submit(list(range(start_row, first_end)))

        while row < n:
            end = min(row + bs, n)

            if image_cache is not None:
                images = []
                valid_indices = []
                for idx in range(row, end):
                    img = image_cache[idx]
                    if img is not None:
                        images.append(img)
                        valid_indices.append(idx)
            elif prefetcher is not None:
                images, valid_indices = prefetcher.result()
                if end < n:
                    next_end = min(end + bs, n)
                    prefetcher.submit(list(range(end, next_end)))
            else:
                images, valid_indices = _load_batch(
                    df,
                    list(range(row, end)),
                    out_dir=out_dir,
                    render_size=RENDER_SIZE,
                    errors=errors,
                    workers=io_workers,
                )

            if not images:
                row = end
                batch_i += 1
                bar.update(1)
                continue

            try:
                inputs = processor(images=images, return_tensors="pt").to(device, non_blocking=True)
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                emb = feats.cpu().numpy().astype(np.float16)
                np.save(shard_dir / f"embeddings_shard_{batch_i:04d}.npy", emb)
                meta_path = shard_dir / f"embeddings_shard_{batch_i:04d}.json"
                meta_path.write_text(json.dumps({"indices": valid_indices}), encoding="utf-8")
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs > 1:
                    bs = max(1, bs // 2)
                    logging.warning("OOM — halving batch size to %d", bs)
                    continue
                errors.log("embed", f"batch_{batch_i}", None, "OOM", str(e))
                row = end
                batch_i += 1
                bar.update(1)
                continue
            except Exception as e:  # noqa: BLE001
                errors.log("embed", f"batch_{batch_i}", None, type(e).__name__, str(e))
                row = end
                batch_i += 1
                bar.update(1)
                continue

            row = end
            batch_i += 1
            bar.update(1)
            bar.set_postfix(row=row, batch_size=bs)

    bar.close()
    if prefetcher is not None:
        prefetcher.close()
    infer_s = time.perf_counter() - infer_start

    visual = _build_aligned_matrix(df, shard_dir, n)
    _write_final(visual, out_dir, n, bs, device)
    stats = {
        "rows": n,
        "pool_rows": pool_n,
        "device": device,
        "batch_size": bs,
        "model": SIGLIP_MODEL,
        "pilot": pilot,
        "errors_logged": errors.count,
        "preload": do_preload,
        "io_workers": io_workers,
        "timing_s": {
            "model_load": round(model_load_s, 2),
            "preload": round(preload_s, 2),
            "inference": round(infer_s, 2),
            "total": round(time.perf_counter() - t0, 2),
        },
    }
    write_json(out_dir / "embed_meta.json", stats)
    return stats


def _concat_shards(shard_dir: Path, n: int) -> np.ndarray:
    return _build_aligned_matrix(None, shard_dir, n)


def _build_aligned_matrix(df, shard_dir: Path, n: int) -> np.ndarray:
    dim = None
    for p in sorted(shard_dir.glob("embeddings_shard_*.npy")):
        arr = np.load(p)
        dim = arr.shape[1]
        break
    if dim is None:
        raise RuntimeError("no embedding shards found")

    out = np.zeros((n, dim), dtype=np.float16)

    for npy_path in sorted(shard_dir.glob("embeddings_shard_*.npy")):
        emb = np.load(npy_path)
        meta_path = npy_path.with_suffix(".json")
        if meta_path.exists():
            indices = json.loads(meta_path.read_text(encoding="utf-8"))["indices"]
            for j, idx in enumerate(indices):
                if idx < n and j < len(emb):
                    out[idx] = emb[j]
        else:
            batch_i = int(npy_path.stem.split("_")[-1])
            start = batch_i * len(emb)
            for j in range(len(emb)):
                if start + j < n:
                    out[start + j] = emb[j]

    return out


def _write_final(visual: np.ndarray, out_dir: Path, n: int, batch_size: int, device: str) -> None:
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / "visual_fp16.npy", visual)
    write_json(
        out_dir / "embed_meta.json",
        {
            "rows": n,
            "dim": int(visual.shape[1]),
            "model": SIGLIP_MODEL,
            "batch_size": batch_size,
            "device": device,
            "alpha_default": 2.0,
        },
    )


def main():
    ap = argparse.ArgumentParser(description="Embed broad pool with SigLIP")
    ap.add_argument("--pilot", action="store_true", help="metadata only; embeds the full pool")
    ap.add_argument("--fresh", action="store_true", help="drop cached shards/embeddings and re-embed")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "broad")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--preload", action="store_true", help="load all PNGs into RAM first (not for full ~70k pool)")
    ap.add_argument("--io-workers", type=int, default=None, help="parallel PNG loaders per batch (default 16)")
    args = ap.parse_args()

    stats = embed_pool(
        args.out,
        batch_size=args.batch_size,
        pilot=args.pilot,
        device=args.device,
        fresh=args.fresh,
        preload=args.preload,
        io_workers=args.io_workers,
    )
    print_summary(
        "embed",
        scanned=stats["rows"],
        kept=stats["rows"],
        errors_logged=stats.get("errors_logged", 0),
        device=stats["device"],
    )
    print(f"wrote {args.out / 'embeddings' / 'visual_fp16.npy'}")


if __name__ == "__main__":
    main()
