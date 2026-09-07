"""Paper-facing analysis of the cached 8-checkpoint × 3-bench sweep.

Computes batched DINOv2 similarity, foreground F1, paired bootstrap intervals,
gold-length buckets, a repetition/termination taxonomy, and two paper figures.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.run_bench_eval import _letterbox_image, _read_gold_image, _read_gold_svg
from outputs.analysis.ink_sanity import ink_f1
from structsvg_lib.svg_ops import extract_svg_blob, render_pil, validate_svg
from train.vllm_infer import _TAG_NAME_RE, _TAG_RE, recover_svg_prefix

PCTS = [0, 5, 10, 20, 40, 60, 80, 100]
SWEEPS = {
    "SVG-Diagrams": {
        "manifest": ROOT / "outputs/metrics/eval128_svg_diagrams_ctx8192_remote/eval128_svg_diagrams_ctx8192/eval_subset_svg_diagrams_128_seed42.jsonl",
        "pred_dir": ROOT / "outputs/generations/eval128_svg_diagrams_ctx8192_files/eval128_svg_diagrams_ctx8192",
        "prefix": "svg_diagrams",
    },
    "VFIG-ID": {
        "manifest": ROOT / "outputs/metrics/eval64_ctx8192_remote/eval64_ctx8192/eval_subset_vfig_id_128_seed42.jsonl",
        "pred_dir": ROOT / "outputs/generations/eval64_ctx8192_files/eval64_ctx8192",
        "prefix": "vfig_id",
    },
    "VFIG-OOD": {
        "manifest": ROOT / "outputs/metrics/eval128_vfig_ood_ctx8192_remote/eval128_vfig_ood_ctx8192/eval_subset_vfig_ood_128_seed42.jsonl",
        "pred_dir": ROOT / "outputs/generations/eval128_vfig_ood_ctx8192_files/eval128_vfig_ood_ctx8192",
        "prefix": "vfig_ood",
    },
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pred_path(spec: dict, pct: int) -> Path:
    tag = "base_0pct" if pct == 0 else f"pct_{pct:03d}"
    return spec["pred_dir"] / f"{spec['prefix']}_{tag}_prompt.jsonl"


def repeated_structure(text: str) -> bool:
    """Detect adjacent repeated XML structures while ignoring attribute values."""
    sigs: list[str] = []
    for match in _TAG_RE.finditer(text or ""):
        token = match.group(0)
        if token.startswith(("<!--", "<?", "<!")):
            continue
        named = _TAG_NAME_RE.match(token)
        if not named:
            continue
        closing, name = bool(named.group(1)), named.group(2).lower()
        if closing:
            sigs.append(f"/{name}")
            continue
        attrs = sorted(set(re.findall(r"\s([A-Za-z_:][\w:.-]*)\s*=", token)))
        sigs.append(f"{name}({','.join(attrs)})" + ("/" if token.rstrip().endswith("/>") else ""))
    # Three adjacent copies of a 1..12-tag structural block.
    for end in range(3, len(sigs) + 1):
        for width in range(1, min(12, end // 3) + 1):
            a = sigs[end - 3 * width : end - 2 * width]
            if a == sigs[end - 2 * width : end - width] == sigs[end - width : end]:
                if width >= 2 or (a and not a[0].startswith("/")):
                    return True
    return False


def taxonomy(pred: dict, raw_ok: bool) -> tuple[str, bool]:
    text = pred.get("pred_text") or ""
    opened = bool(re.search(r"<svg\b", text, flags=re.I))
    closed = "</svg>" in text.lower()
    repetition = repeated_structure(text) or str(pred.get("recovery", "")).startswith("trimmed_")
    if not opened:
        return "no_svg", repetition
    if raw_ok:
        return "valid_complete", repetition
    if closed:
        return "closed_invalid", repetition
    if pred.get("hit_length_limit") and repetition:
        return "length_repetition", repetition
    if pred.get("hit_length_limit"):
        return "length_unterminated", repetition
    if repetition:
        return "other_repetition", repetition
    return "other_unterminated", repetition


def target_image(row: dict, size: int) -> Image.Image:
    svg = _read_gold_svg(row)
    if svg:
        return render_pil(svg, size=size).convert("RGB")
    return _letterbox_image(_read_gold_image(row), size=size).convert("RGB")


def token_lengths(rows: list[dict]) -> dict[str, int | None]:
    from tokenizers import Tokenizer

    checkpoint = ROOT / "outputs/e4b_broad/e4b_broad/checkpoint_pct_100"
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    out = {}
    for row in rows:
        svg = _read_gold_svg(row)
        out[str(row["id"])] = len(tokenizer.encode(svg, add_special_tokens=False).ids) if svg else None
    return out


def dino_embeddings(images: list[Image.Image], batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    name = "facebook/dinov2-small"
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    feats = []
    for start in range(0, len(images), batch_size):
        batch = processor(images=images[start : start + batch_size], return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            feat = model(**batch).last_hidden_state[:, 0]
            feat = torch.nn.functional.normalize(feat, dim=-1)
        feats.append(feat.cpu().float().numpy())
    return np.concatenate(feats, axis=0)


def paired_ci(base: dict[str, float], other: dict[str, float], *, seed: int = 0, n_boot: int = 4000) -> dict:
    ids = sorted(set(base) & set(other))
    diffs = np.asarray([other[i] - base[i] for i in ids], dtype=np.float64)
    if not len(diffs):
        return {"n": 0, "mean_delta": math.nan, "lo": math.nan, "hi": math.nan}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, 500):
        count = min(500, n_boot - start)
        idx = rng.integers(0, len(diffs), size=(count, len(diffs)))
        means[start : start + count] = diffs[idx].mean(axis=1)
    return {
        "n": len(diffs),
        "mean_delta": float(diffs.mean()),
        "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
    }


def mean(records: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in records if r.get(key) is not None and not math.isnan(float(r[key]))]
    return float(np.mean(vals)) if vals else math.nan


def analyze_bench(name: str, spec: dict, *, render_size: int, dino_batch_size: int) -> tuple[dict, list[dict]]:
    gold_rows = load_jsonl(spec["manifest"])
    gold = {str(row["id"]): row for row in gold_rows}
    lengths = token_lengths(gold_rows)
    available_lengths = [x for x in lengths.values() if x is not None]
    quartiles = np.quantile(available_lengths, [0.25, 0.5, 0.75]) if available_lengths else None
    targets = {eid: target_image(row, render_size) for eid, row in gold.items()}
    target_ids = sorted(gold)
    target_features = dino_embeddings([targets[eid] for eid in target_ids], dino_batch_size)
    target_feature = {eid: feat for eid, feat in zip(target_ids, target_features)}

    all_records: list[dict] = []
    by_pct: dict[int, list[dict]] = {}
    for pct in PCTS:
        preds = {str(row["id"]): row for row in load_jsonl(pred_path(spec, pct))}
        records: list[dict] = []
        render_items: list[tuple[int, str, Image.Image]] = []
        for eid in target_ids:
            pred = preds[eid]
            text = pred.get("pred_text") or ""
            raw_blob = extract_svg_blob(text) or text
            raw_val = validate_svg(raw_blob, try_render=True)
            recovered = pred.get("recovered_svg")
            recovery_reason = pred.get("recovery")
            if not recovered:
                recovered, recovery_reason = recover_svg_prefix(text)
            rec_blob = extract_svg_blob(recovered or "") if recovered else None
            rec_val = validate_svg(rec_blob, try_render=True) if rec_blob else raw_val
            category, repetition = taxonomy(pred, raw_val.ok)
            length = lengths[eid]
            if length is None or quartiles is None:
                bucket = None
            else:
                q1, q2, q3 = quartiles
                bucket = "Q1 shortest" if length <= q1 else "Q2" if length <= q2 else "Q3" if length <= q3 else "Q4 longest"
            target_gray = np.asarray(targets[eid].convert("L"))
            record = {
                "bench": name,
                "pct": pct,
                "id": eid,
                "gold_tokens": length,
                "length_bucket": bucket,
                "target_ink_fraction": float((target_gray < 245).mean()),
                "raw_validity": float(raw_val.ok),
                "recovered_validity": float(rec_val.ok),
                "svg_close": float("</svg>" in text.lower()),
                "hit_length_limit": float(bool(pred.get("hit_length_limit"))),
                "repetition": float(repetition),
                "taxonomy": category,
                "recovery": recovery_reason,
                "output_tokens": pred.get("new_tokens"),
                "foreground_f1_raw": 0.0,
                "foreground_f1_recovered": 0.0,
                "dino_raw": 0.0,
                "dino_recovered": 0.0,
            }
            if raw_val.ok:
                try:
                    raw_img = render_pil(raw_blob, size=render_size).convert("RGB")
                    record["foreground_f1_raw"] = ink_f1(raw_img, targets[eid])
                    render_items.append((len(records), "raw", raw_img))
                except Exception:
                    record["raw_validity"] = 0.0
            if rec_val.ok:
                try:
                    rec_img = render_pil(rec_blob if rec_blob else raw_blob, size=render_size).convert("RGB")
                    record["foreground_f1_recovered"] = ink_f1(rec_img, targets[eid])
                    render_items.append((len(records), "recovered", rec_img))
                except Exception:
                    record["recovered_validity"] = 0.0
            records.append(record)

        if render_items:
            feats = dino_embeddings([item[2] for item in render_items], dino_batch_size)
            for (index, kind, _), feat in zip(render_items, feats):
                score = float(np.dot(feat, target_feature[records[index]["id"]]))
                records[index][f"dino_{kind}"] = score
        by_pct[pct] = records
        all_records.extend(records)
        print(f"analyzed {name} pct={pct} n={len(records)}", flush=True)

    aggregate = {}
    keys = [
        "raw_validity",
        "recovered_validity",
        "svg_close",
        "hit_length_limit",
        "repetition",
        "foreground_f1_raw",
        "foreground_f1_recovered",
        "dino_raw",
        "dino_recovered",
    ]
    for pct, records in by_pct.items():
        base = by_pct[0]
        base_by_id = {r["id"]: r for r in base}
        aggregate[str(pct)] = {
            "n": len(records),
            **{key: mean(records, key) for key in keys},
            "taxonomy": dict(Counter(r["taxonomy"] for r in records)),
            "length_buckets": {
                bucket: {
                    "n": len(bucket_rows),
                    **{key: mean(bucket_rows, key) for key in keys},
                }
                for bucket in ("Q1 shortest", "Q2", "Q3", "Q4 longest")
                if (bucket_rows := [r for r in records if r["length_bucket"] == bucket])
            },
            "paired_vs_base": {
                key: paired_ci(
                    {eid: row[key] for eid, row in base_by_id.items()},
                    {row["id"]: row[key] for row in records},
                    seed=1000 + pct,
                )
                for key in keys
            },
        }
    return {
        "gold_token_quartiles": [float(x) for x in quartiles] if quartiles is not None else None,
        "length_bucket_note": "unavailable: raster-only references" if quartiles is None else "gold SVG token quartiles",
        "checkpoints": aggregate,
    }, all_records


def plot_trajectories(report: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    colors = {"SVG-Diagrams": "#2563eb", "VFIG-ID": "#ea580c", "VFIG-OOD": "#7c3aed"}
    panels = [
        ("raw_validity", "recovered_validity", "Executable SVG rate"),
        ("foreground_f1_raw", "foreground_f1_recovered", "Foreground F1 (invalid = 0)"),
        ("hit_length_limit", "repetition", "Length-limit and repetition failures"),
    ]
    for ax, (raw_key, rec_key, title) in zip(axes, panels):
        for bench, bench_report in report["benches"].items():
            cps = bench_report["checkpoints"]
            raw = [cps[str(p)][raw_key] for p in PCTS]
            rec = [cps[str(p)][rec_key] for p in PCTS]
            raw_label = f"{bench} raw" if raw_key != "hit_length_limit" else f"{bench} length limit"
            rec_label = f"{bench} repaired" if rec_key != "repetition" else f"{bench} repetition"
            ax.plot(PCTS, raw, marker="o", color=colors[bench], label=raw_label)
            ax.plot(PCTS, rec, marker="o", linestyle="--", alpha=0.72, color=colors[bench], label=rec_label)
        ax.set_title(title)
        ax.set_xlabel("Training progress (%)")
        ax.grid(alpha=0.22)
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(lab)
    # First two panels share labels; retain one set plus the failure panel.
    keep = list(range(6)) + list(range(12, 18))
    fig.legend([handles[i] for i in keep], [labels[i] for i in keep], loc="lower center", ncol=6, frameon=False)
    fig.suptitle("Post-training improves execution more than visual reconstruction")
    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_failures(records: list[dict], out: Path, render_size: int) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_key = {(r["bench"], r["pct"], r["id"]): r for r in records}
    pools = [
        ("Autoregressive repetition", [r for r in records if r["pct"] == 100 and r["taxonomy"] == "length_repetition"]),
        ("Coarse silhouette after repair", [r for r in records if r["pct"] == 100 and not r["raw_validity"] and r["recovered_validity"]]),
        ("Valid syntax, wrong content", [r for r in records if r["pct"] == 100 and r["raw_validity"] and r["target_ink_fraction"] > 0.005]),
        ("OOD content failure", [r for r in records if r["bench"] == "VFIG-OOD" and r["pct"] == 100 and r["recovered_validity"] and r["target_ink_fraction"] > 0.005]),
    ]
    chosen = []
    used = set()
    for label, pool in pools:
        pool = [r for r in pool if (r["bench"], r["id"]) not in used]
        if label == "Coarse silhouette after repair":
            pool.sort(key=lambda r: r["foreground_f1_recovered"], reverse=True)
        else:
            pool.sort(key=lambda r: r["foreground_f1_recovered"])
        if pool:
            row = pool[0]
            chosen.append((label, row))
            used.add((row["bench"], row["id"]))

    fig, axes = plt.subplots(len(chosen), 2, figsize=(8.8, 3.0 * len(chosen)))
    if len(chosen) == 1:
        axes = np.asarray([axes])
    selection = []
    for i, (label, record) in enumerate(chosen):
        spec = SWEEPS[record["bench"]]
        gold_rows = {str(r["id"]): r for r in load_jsonl(spec["manifest"])}
        target = target_image(gold_rows[record["id"]], render_size)
        pred_rows = {str(r["id"]): r for r in load_jsonl(pred_path(spec, record["pct"]))}
        pred = pred_rows[record["id"]]
        text = pred.get("pred_text") or ""
        recovered = pred.get("recovered_svg")
        if not recovered:
            recovered, _ = recover_svg_prefix(text)
        blob = extract_svg_blob(recovered or text) or recovered or text
        try:
            prediction = render_pil(blob, size=render_size).convert("RGB")
        except Exception:
            prediction = Image.new("RGB", (render_size, render_size), "white")
        axes[i, 0].imshow(target)
        axes[i, 1].imshow(prediction)
        axes[i, 0].set_title(f"{label}: target")
        axes[i, 1].set_title(
            f"{record['bench']} @ {record['pct']}%\n"
            f"raw={int(record['raw_validity'])}, repaired={int(record['recovered_validity'])}, "
            f"F1={record['foreground_f1_recovered']:.3f}"
        )
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
        selection.append({"label": label, **record})
    fig.suptitle("Representative failure modes after post-training")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return selection


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/analysis/paper_v1")
    ap.add_argument("--figure-dir", type=Path, default=ROOT / "paper/figures")
    ap.add_argument("--render-size", type=int, default=512)
    ap.add_argument("--dino-batch-size", type=int, default=24)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.out_dir / "sweep_analysis.json"
    rows_path = args.out_dir / "per_example.jsonl"
    if args.plot_only:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        all_records = load_jsonl(rows_path)
        plot_trajectories(report, args.figure_dir / "checkpoint_trajectories.png")
        selected = plot_failures(all_records, args.figure_dir / "failure_cases.png", args.render_size)
        (args.out_dir / "failure_case_selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        print(f"wrote {args.figure_dir / 'checkpoint_trajectories.png'}")
        print(f"wrote {args.figure_dir / 'failure_cases.png'}")
        return

    report = {
        "protocol": {
            "n_per_bench": 128,
            "pcts": PCTS,
            "invalid_outputs_scored_as_zero": True,
            "repair": "deterministic well-formed prefix closure; secondary oracle upper bound",
            "dino": "facebook/dinov2-small CLS cosine, batched",
            "length_bucket": "within-bench quartiles of gold SVG tokens under saved Gemma tokenizer",
            "paired_ci": "paired example bootstrap, 4000 resamples, 95% percentile interval",
        },
        "benches": {},
    }
    all_records = []
    for name, spec in SWEEPS.items():
        bench_report, records = analyze_bench(
            name,
            spec,
            render_size=args.render_size,
            dino_batch_size=args.dino_batch_size,
        )
        report["benches"][name] = bench_report
        all_records.extend(records)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with rows_path.open("w", encoding="utf-8") as f:
        for row in all_records:
            f.write(json.dumps(row) + "\n")
    plot_trajectories(report, args.figure_dir / "checkpoint_trajectories.png")
    selected = plot_failures(all_records, args.figure_dir / "failure_cases.png", args.render_size)
    (args.out_dir / "failure_case_selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {rows_path}")
    print(f"wrote {args.figure_dir / 'checkpoint_trajectories.png'}")
    print(f"wrote {args.figure_dir / 'failure_cases.png'}")


if __name__ == "__main__":
    main()
