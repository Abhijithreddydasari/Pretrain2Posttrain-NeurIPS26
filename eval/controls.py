"""Image-control evaluation: correct vs shuffled vs blank inputs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-correct", type=Path, required=True)
    ap.add_argument("--preds-shuffled", type=Path, required=True)
    ap.add_argument("--preds-blank", type=Path, required=True)
    ap.add_argument("--metrics-correct", type=Path, required=True)
    ap.add_argument("--metrics-shuffled", type=Path, required=True)
    ap.add_argument("--metrics-blank", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/metrics/controls.json"))
    args = ap.parse_args()

    def load(p: Path):
        return json.loads(p.read_text(encoding="utf-8"))

    c, s, b = load(args.metrics_correct), load(args.metrics_shuffled), load(args.metrics_blank)
    report = {
        "correct": c.get("aggregate"),
        "shuffled": s.get("aggregate"),
        "blank": b.get("aggregate"),
        "flags": {
            "vision_ignored_if": "shuffled structure ≈ correct structure",
            "template_prior_if": "blank validity high with non-trivial SVG",
        },
    }
    # heuristic warning
    try:
        if abs(c["aggregate"]["relation_f1"] - s["aggregate"]["relation_f1"]) < 0.05:
            report["warning"] = "shuffled≈correct structure — check that the model uses the image"
    except Exception:
        pass
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
