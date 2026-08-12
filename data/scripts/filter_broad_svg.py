"""Stream-filter starvector/svg-diagrams into candidate manifests (no manual 182k browse)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from structsvg_lib.svg_ops import extract_svg_blob, perceptual_hash, validate_svg


def feature_bucket(svg: str, val) -> str:
    # crude buckets for stratification
    low = svg.lower()
    n_text = low.count("<text")
    n_rect = low.count("<rect")
    n_line = low.count("<line") + low.count("<polyline")
    n_path = low.count("<path")
    if n_rect >= 2 and n_text >= 1 and n_line >= 1:
        return "workflow_like"
    if n_line >= 3 and n_text <= 3 and n_path <= 2:
        return "geometry_like"
    if n_path >= 5 and n_text == 0:
        return "path_soup"
    if n_text >= 1:
        return "labeled"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="only scan a few hundred streaming rows")
    ap.add_argument("--scan-n", type=int, default=5000)
    ap.add_argument("--target-n", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "broad")
    ap.add_argument("--test-dedup", type=Path, default=None, help="optional jsonl of test svg hashes")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    scan_n = 400 if args.pilot else args.scan_n
    target_n = 40 if args.pilot else args.target_n
    args.out.mkdir(parents=True, exist_ok=True)

    test_hashes: set[str] = set()
    if args.test_dedup and args.test_dedup.exists():
        for line in args.test_dedup.read_text(encoding="utf-8").splitlines():
            if line.strip():
                test_hashes.add(json.loads(line)["sha256"])

    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed; writing empty pilot stub")
        stub = args.out / "candidates_pilot.jsonl"
        stub.write_text("", encoding="utf-8")
        return

    print(f"streaming starvector/svg-diagrams (scan_n={scan_n})...")
    ds = load_dataset("starvector/svg-diagrams", split="train", streaming=True)

    buckets: dict[str, list[dict]] = defaultdict(list)
    kept = 0
    seen = 0
    for row in ds:
        seen += 1
        if seen > scan_n:
            break
        raw = row.get("Svg") or row.get("svg") or ""
        svg = extract_svg_blob(raw) or raw
        val = validate_svg(svg, try_render=False)
        if not val.parse_ok or not val.normalized:
            continue
        if val.errors:
            continue
        if val.sha256 in test_hashes:
            continue
        if val.n_drawable < 2 or val.n_elements > 200:
            continue
        if len(val.normalized) > 20000:
            continue
        b = feature_bucket(val.normalized, val)
        if b == "path_soup":
            continue
        rec = {
            "id": row.get("Filename") or hashlib.sha256(val.sha256.encode()).hexdigest()[:16],
            "sha256": val.sha256,
            "bucket": b,
            "n_drawable": val.n_drawable,
            "n_elements": val.n_elements,
            "svg": val.normalized,
            "source": "starvector/svg-diagrams",
        }
        ph = perceptual_hash(val.normalized) if kept < 200 else None
        if ph:
            rec["phash"] = ph
        buckets[b].append(rec)
        kept += 1
        if kept % 100 == 0:
            print(f"  scanned={seen} kept={kept} buckets={ {k: len(v) for k, v in buckets.items()} }")

    # stratified sample
    import random

    rng = random.Random(args.seed)
    preferred = ["workflow_like", "geometry_like", "labeled", "other"]
    selected: list[dict] = []
    per = max(1, target_n // max(1, len(preferred)))
    for b in preferred:
        pool = buckets.get(b, [])
        rng.shuffle(pool)
        selected.extend(pool[:per])
    # fill
    rest = [r for b, rs in buckets.items() for r in rs if r not in selected]
    rng.shuffle(rest)
    while len(selected) < target_n and rest:
        selected.append(rest.pop())
    selected = selected[:target_n]

    out_path = args.out / ("candidates_pilot.jsonl" if args.pilot else "train_manifest.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in selected:
            # store svg only in pilot; full train may write files separately
            f.write(json.dumps(r) + "\n")
    stats = {k: len(v) for k, v in buckets.items()}
    (args.out / "filter_stats.json").write_text(
        json.dumps({"scanned": seen, "kept_pre_sample": kept, "buckets": stats, "selected": len(selected)}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_path} n={len(selected)}")
    print("bucket stats:", stats)


if __name__ == "__main__":
    main()
