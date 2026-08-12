"""Plot / summarize checkpoint curves from metrics JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structsvg_lib.metrics import area_between, emergence_times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--curves",
        type=Path,
        required=True,
        help="JSON: {pcts:[], syntax:[], structure:[], ...}",
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/metrics/emergence.json"))
    args = ap.parse_args()
    data = json.loads(args.curves.read_text(encoding="utf-8"))
    pcts = data["pcts"]
    syntax = data["syntax"]
    structure = data["structure"]
    report = {
        "syntax_emergence": emergence_times(pcts, syntax),
        "structure_emergence": emergence_times(pcts, structure),
        "area_syntax_minus_structure": area_between(syntax, structure, pcts),
    }
    if "id_structure" in data and "ood_structure" in data:
        report["id_ood_gaps"] = [
            float(i) - float(o) for i, o in zip(data["id_structure"], data["ood_structure"])
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(pcts, syntax, marker="o", label="syntax/validity")
        ax.plot(pcts, structure, marker="s", label="structure")
        ax.set_xlabel("SFT progress (%)")
        ax.set_ylabel("metric")
        ax.legend()
        ax.set_title("Checkpoint curves")
        fig_path = args.out.with_suffix(".png")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        print(f"wrote {fig_path}")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
