"""Stream-filter starvector/svg-diagrams — thin wrapper around broad coreset pipeline (pilot)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.scripts.broad_scan_pool import scan_pool  # noqa: E402
from data.scripts.broad_select_coreset import select_coreset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="pilot mode (~5k scan, ~40 select)")
    ap.add_argument("--scan-n", type=int, default=None, help="override max rows to scan")
    ap.add_argument("--target-n", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "broad")
    ap.add_argument("--test-dedup", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pilot = args.pilot or args.scan_n is not None
    max_rows = args.scan_n if args.scan_n else (5000 if args.pilot else None)
    target_n = 40 if args.pilot else args.target_n

    scan_pool(
        args.out,
        max_rows=max_rows,
        pilot=pilot,
        seed=args.seed,
        test_hashes_path=args.test_dedup,
    )
    print("Note: full pipeline also requires broad_embed then broad_select_coreset.")
    print("For pilot end-to-end, run:")
    print("  python -m data.scripts.broad_embed --pilot")
    print(f"  python -m data.scripts.broad_select_coreset --pilot --target-n {target_n}")


if __name__ == "__main__":
    main()
