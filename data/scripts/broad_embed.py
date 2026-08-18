"""Stage 2: SigLIP visual embeddings for broad pool candidates."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.scripts.broad_io import ErrorLogger, print_summary, progress_bar, write_json  # noqa: E402
from data.scripts.broad_scan_pool import load_pool_index, resolve_pool_image  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

SIGLIP_MODEL = "google/siglip-base-patch16-224"
DEFAULT_BATCH = 64
RENDER_SIZE = 224


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


def embed_pool(
    out_dir: Path,
    *,
    batch_size: int = DEFAULT_BATCH,
    pilot: bool = False,
    device: str | None = None,
) -> dict:
    pool_path = out_dir / "pool_index.parquet"
    if not pool_path.exists():
        raise FileNotFoundError(f"missing {pool_path}; run broad_scan_pool first")

    df = load_pool_index(pool_path)
    if pilot:
        df = df.head(min(200, len(df)))

    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor, model, torch_mod = _load_model(device)

    shard_dir = _shard_dir(out_dir)
    errors = ErrorLogger(out_dir / "errors.jsonl")
    n = len(df)
    bs = batch_size
    shard_idx = _completed_shards(shard_dir)
    start_row = shard_idx * bs

    if start_row >= n:
        visual = _concat_shards(shard_dir, n)
        _write_final(visual, out_dir, n, bs, device)
        return {"rows": n, "device": device, "resumed": True}

    indices = list(range(n))
    n_batches = (n + bs - 1) // bs
    bar = progress_bar(total=n_batches, desc="embedding", unit="batch", initial=shard_idx)

    row = start_row
    batch_i = shard_idx

    while row < n:
        end = min(row + bs, n)
        batch_indices = indices[row:end]
        images = []
        valid_indices: list[int] = []

        for idx in batch_indices:
            r = df.iloc[idx]
            try:
                img = resolve_pool_image(r.to_dict(), render_size=RENDER_SIZE)
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:  # noqa: BLE001
                errors.log("embed", str(r.get("id", idx)), r.get("sha256"), type(e).__name__, str(e))

        if not images:
            row = end
            batch_i += 1
            bar.update(1)
            continue

        try:
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch_mod.no_grad():
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

    visual = _build_aligned_matrix(df, shard_dir, n)
    _write_final(visual, out_dir, n, bs, device)
    stats = {
        "rows": n,
        "device": device,
        "batch_size": bs,
        "model": SIGLIP_MODEL,
        "pilot": pilot,
        "errors_logged": errors.count,
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
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "broad")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    stats = embed_pool(args.out, batch_size=args.batch_size, pilot=args.pilot, device=args.device)
    print_summary("embed", kept=stats["rows"], errors_logged=stats.get("errors_logged", 0), device=stats["device"])
    print(f"wrote {args.out / 'embeddings' / 'visual_fp16.npy'}")


if __name__ == "__main__":
    main()
