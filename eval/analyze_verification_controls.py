"""Analyze the HF merged-adapter verification and image controls."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analyze_cached_sweep import paired_ci, repeated_structure
from eval.run_bench_eval import _letterbox_image, _read_gold_image, _read_gold_svg
from outputs.analysis.ink_sanity import ink_f1
from structsvg_lib.svg_ops import extract_svg_blob, render_pil, validate_svg
from train.vllm_infer import recover_svg_prefix


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def target_image(row: dict):
    svg = _read_gold_svg(row)
    return render_pil(svg, size=512) if svg else _letterbox_image(_read_gold_image(row), size=512)


def score_rows(manifest: Path, preds_path: Path) -> list[dict]:
    gold = {str(r["id"]): r for r in load_jsonl(manifest)}
    rows = []
    for pred in load_jsonl(preds_path):
        eid = str(pred["id"])
        text = pred.get("pred_text") or ""
        blob = extract_svg_blob(text) or text
        val = validate_svg(blob, try_render=True)
        recovered = pred.get("recovered_svg")
        if not recovered:
            recovered, _ = recover_svg_prefix(text)
        rec_blob = extract_svg_blob(recovered or "") if recovered else None
        rec_val = validate_svg(rec_blob, try_render=True) if rec_blob else val
        target = target_image(gold[eid])
        raw_f1 = 0.0
        recovered_f1 = 0.0
        if val.ok:
            try:
                raw_f1 = ink_f1(render_pil(blob, size=512), target)
            except Exception:
                pass
        if rec_val.ok:
            try:
                recovered_f1 = ink_f1(render_pil(rec_blob or blob, size=512), target)
            except Exception:
                pass
        rows.append(
            {
                "id": eid,
                "raw_validity": float(val.ok),
                "recovered_validity": float(rec_val.ok),
                "foreground_f1_raw": raw_f1,
                "foreground_f1_recovered": recovered_f1,
                "svg_close": float("</svg>" in text.lower()),
                "repetition": float(repeated_structure(text)),
                "chars": len(text),
                "text": text,
            }
        )
    return rows


def aggregate(rows: list[dict]) -> dict:
    keys = [
        "raw_validity",
        "recovered_validity",
        "foreground_f1_raw",
        "foreground_f1_recovered",
        "svg_close",
        "repetition",
        "chars",
    ]
    return {"n": len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in keys}}


def delta(a: list[dict], b: list[dict], key: str) -> dict:
    av = {r["id"]: float(r[key]) for r in a}
    bv = {r["id"]: float(r[key]) for r in b}
    # paired_ci returns other-base; expose a-b for readability.
    ci = paired_ci(bv, av, seed=71)
    return ci


def plot_controls(controls: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pcts = [0, 20, 80, 100]
    conditions = ["correct", "shuffled", "blank"]
    colors = {"correct": "#2563eb", "shuffled": "#ea580c", "blank": "#6b7280"}
    panels = [
        ("foreground_f1_recovered", "Repaired foreground F1"),
        ("raw_validity", "Raw executable rate"),
        ("recovered_validity", "Repaired executable rate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7))
    width = 0.23
    x = np.arange(len(pcts))
    for ax, (key, title) in zip(axes, panels):
        for j, condition in enumerate(conditions):
            vals = [controls["checkpoints"][str(p)][condition][key] for p in pcts]
            ax.bar(x + (j - 1) * width, vals, width, color=colors[condition], label=condition)
        ax.set_xticks(x, [str(p) for p in pcts])
        ax.set_xlabel("Training progress (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Mean over 64 paired examples")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Correct images do not reliably outperform shuffled-image controls")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    hf_root = ROOT / "outputs/generations/hf_verify16_pulled/hf_verify16"
    hf_manifest = ROOT / "outputs/metrics/hf_verify16_pulled/hf_verify16/eval_subset_svg_diagrams_16_seed42.jsonl"
    hf_preds = hf_root / "svg_diagrams_pct_100_prompt.jsonl"
    vllm_preds = ROOT / "outputs/generations/eval128_svg_diagrams_ctx8192_files/eval128_svg_diagrams_ctx8192/svg_diagrams_pct_100_prompt.jsonl"
    ids = {str(r["id"]) for r in load_jsonl(hf_manifest)}
    vllm_subset_path = ROOT / "outputs/analysis/vllm_verify16_subset.jsonl"
    with vllm_subset_path.open("w", encoding="utf-8") as f:
        for row in load_jsonl(vllm_preds):
            if str(row["id"]) in ids:
                f.write(json.dumps(row) + "\n")

    hf = score_rows(hf_manifest, hf_preds)
    vllm = score_rows(hf_manifest, vllm_subset_path)
    hf_by_id, vl_by_id = {r["id"]: r for r in hf}, {r["id"]: r for r in vllm}
    exact = np.mean([hf_by_id[eid]["text"] == vl_by_id[eid]["text"] for eid in ids])
    verification = {
        "hf_merged": aggregate(hf),
        "vllm_hot_swap_same_16": aggregate(vllm),
        "exact_text_match_rate": float(exact),
        "paired_hf_minus_vllm": {
            key: delta(hf, vllm, key)
            for key in ("raw_validity", "recovered_validity", "foreground_f1_recovered", "repetition")
        },
        "interpretation": (
            "HF merged inference also exhibits severe non-termination/repetition. "
            "Large paired differences would still indicate backend sensitivity; exact token identity is not expected."
        ),
    }

    control_root = ROOT / "outputs/generations/controls_vfig_id_64_pulled/controls_vfig_id_64"
    control_manifest = control_root / "subset_vfig_id_64_seed42.jsonl"
    correct_root = ROOT / "outputs/generations/eval64_ctx8192_files/eval64_ctx8192"
    controls = {"checkpoints": {}}
    for pct in (0, 20, 80, 100):
        tag = "base_0pct" if pct == 0 else f"pct_{pct:03d}"
        correct_full = load_jsonl(correct_root / f"vfig_id_{tag}_prompt.jsonl")
        control_ids = {str(r["id"]) for r in load_jsonl(control_manifest)}
        correct_subset = ROOT / "outputs/analysis" / f"control_correct_{tag}.jsonl"
        with correct_subset.open("w", encoding="utf-8") as f:
            for row in correct_full:
                if str(row["id"]) in control_ids:
                    f.write(json.dumps(row) + "\n")
        correct = score_rows(control_manifest, correct_subset)
        shuffled = score_rows(control_manifest, control_root / f"vfig_id_{tag}_shuffled.jsonl")
        blank = score_rows(control_manifest, control_root / f"vfig_id_{tag}_blank.jsonl")
        c_by_id = {r["id"]: r for r in correct}
        s_by_id = {r["id"]: r for r in shuffled}
        b_by_id = {r["id"]: r for r in blank}
        controls["checkpoints"][str(pct)] = {
            "correct": aggregate(correct),
            "shuffled": aggregate(shuffled),
            "blank": aggregate(blank),
            "exact_text_match_to_correct": {
                "shuffled": float(np.mean([c_by_id[eid]["text"] == s_by_id[eid]["text"] for eid in control_ids])),
                "blank": float(np.mean([c_by_id[eid]["text"] == b_by_id[eid]["text"] for eid in control_ids])),
            },
            "paired_correct_minus_shuffled": {
                key: delta(correct, shuffled, key)
                for key in ("raw_validity", "recovered_validity", "foreground_f1_recovered", "repetition")
            },
            "paired_correct_minus_blank": {
                key: delta(correct, blank, key)
                for key in ("raw_validity", "recovered_validity", "foreground_f1_recovered", "repetition")
            },
        }
    controls["interpretation_rule"] = (
        "Evidence of visual conditioning requires correct-image fidelity to exceed both shuffled and blank under paired intervals; "
        "syntax rates alone are not evidence of image use."
    )

    out = ROOT / "outputs/analysis/verification_and_controls.json"
    out.write_text(json.dumps({"verification": verification, "controls": controls}, indent=2), encoding="utf-8")
    plot_controls(controls, ROOT / "paper/figures/image_controls.png")
    print(json.dumps({"verification": verification, "controls": controls}, indent=2))


if __name__ == "__main__":
    main()
