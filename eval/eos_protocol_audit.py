"""Audit the PT/base versus IT turn terminators used by the completed run."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/e4b_broad/e4b_broad/checkpoint_pct_100"),
    )
    ap.add_argument(
        "--budget-report",
        type=Path,
        default=Path("outputs/e4b_broad_v2/token_budget_report.json"),
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/analysis/eos_protocol_audit.json"))
    args = ap.parse_args()

    tok_cfg = json.loads((args.checkpoint / "tokenizer_config.json").read_text(encoding="utf-8"))
    template_path = args.checkpoint / "chat_template.jinja"
    template = template_path.read_text(encoding="utf-8")
    budget = json.loads(args.budget_report.read_text(encoding="utf-8"))
    tails = [str(row.get("tail", "")) for row in budget.get("dropped_rows", [])]
    # The report stores dropped examples only. Earlier audit previews can also be
    # recovered from the checked-in run metadata when present.
    if not tails:
        for candidate in (
            Path("outputs/e4b_broad_v2_meta/token_budget_report.json"),
            Path("outputs/e4b_broad_v2/token_budget_report.json"),
        ):
            if candidate.exists():
                data = json.loads(candidate.read_text(encoding="utf-8"))
                tails.extend(str(row.get("tail", "")) for row in data.get("dropped_rows", []))

    report = {
        "model_family": "google/gemma-4-E4B PT/base",
        "tokenizer_eos_token": tok_cfg.get("eos_token"),
        "tokenizer_eot_token": tok_cfg.get("eot_token"),
        "chat_template_source": "borrowed from -it sibling by train.model_load.ensure_chat_template",
        "template_closes_assistant_with_eot": bool(re.search(r"<turn\|>", template)),
        "observed_tail_count": len(tails),
        "observed_eot_tails": sum("<turn|>" in tail for tail in tails),
        "observed_eos_tails": sum("<eos>" in tail for tail in tails),
        "finding": (
            "The completed base-model SFT used the instruction/chat template and supervised <turn|>; "
            "the tokenizer declares <eos> as EOS. Treat termination results as protocol-confounded "
            "until a small EOS-vs-EOT ablation is run."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
