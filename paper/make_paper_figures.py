"""Generate the data-curation and experiment schematic used in the paper."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


BLUE = "#1765ad"
ORANGE = "#b64d13"
GREEN = "#21815c"
PURPLE = "#6d44a3"
INK = "#17202a"
PALE = "#f5f7fa"


def box(ax, xy, wh, title, body, color=BLUE, fontsize=9):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor="white", edgecolor=color, linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x + 0.025, y + h - 0.05, title, color=color,
            fontsize=fontsize + 0.5, fontweight="bold", va="top")
    ax.text(x + 0.025, y + h - 0.105, body, color=INK,
            fontsize=fontsize, va="top", linespacing=1.28)


def arrow(ax, a, b, color=INK):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.3, shrinkA=2, shrinkB=2))


def main():
    out = Path("paper/figures")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(12.2, 5.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.012, 0.025), 0.976, 0.93,
                               boxstyle="round,pad=0.012",
                               facecolor=PALE, edgecolor="#d8dee8", lw=0.9))

    ax.text(0.04, 0.91, "DATA CURATION", fontsize=10,
            fontweight="bold", color=INK)
    box(ax, (0.04, 0.72), (0.15, 0.14), "Source",
        "182,144\nSVG-Diagrams", BLUE, 8.4)
    box(ax, (0.235, 0.72), (0.17, 0.14), "Validate + filter",
        "44,807\nstatic native SVGs", BLUE, 8.4)
    box(ax, (0.45, 0.72), (0.15, 0.14), "Deduplicate",
        "30,011\nrendered candidates", BLUE, 8.4)
    box(ax, (0.645, 0.72), (0.15, 0.14), "Represent",
        "SigLIP image +\nSVG structure", BLUE, 8.4)
    box(ax, (0.84, 0.72), (0.12, 0.14), "Coreset",
        "2,000\nselected", ORANGE, 8.4)
    for a, b in [((0.19, 0.79), (0.235, 0.79)),
                 ((0.405, 0.79), (0.45, 0.79)),
                 ((0.60, 0.79), (0.645, 0.79)),
                 ((0.795, 0.79), (0.84, 0.79))]:
        arrow(ax, a, b)
    ax.text(0.32, 0.675, "native/render checks, geometry cleanliness, test-hash exclusion",
            fontsize=7.4, color=INK, ha="center")
    ax.text(0.72, 0.675, "perceptual-hash deduplication and diversity selection",
            fontsize=7.4, color=INK, ha="center")

    ax.plot([0.04, 0.96], [0.625, 0.625], color="#ccd3dd", lw=0.9)
    ax.text(0.04, 0.58, "POST-TRAINING TRAJECTORY AND EVALUATION", fontsize=10,
            fontweight="bold", color=INK)
    box(ax, (0.04, 0.34), (0.18, 0.16), "Pretrained base",
        "Gemma 4 E4B\n0% condition", BLUE, 8.6)
    box(ax, (0.29, 0.34), (0.22, 0.16), "Broad SVG SFT",
        "1,995 complete pairs\none two-epoch LoRA run", ORANGE, 8.6)
    arrow(ax, (0.22, 0.42), (0.29, 0.42))
    ax.text(0.74, 0.49, "checkpoints at 0, 5, 10, 20, 40, 60, 80, 100%",
            fontsize=8.3, color=INK, ha="center")
    ax.plot([0.56, 0.94], [0.42, 0.42], color=INK, lw=1.2)
    checkpoints = [0, 5, 10, 20, 40, 60, 80, 100]
    xs = [0.56 + 0.38 * p / 100 for p in checkpoints]
    ax.scatter(xs, [0.42] * len(xs), s=30, color=[BLUE] + [ORANGE] * 7,
               zorder=3, edgecolor="white", linewidth=0.5)
    for x, p in zip(xs, checkpoints):
        ax.text(x, 0.375, str(p), ha="center", fontsize=7.7)

    arrow(ax, (0.39, 0.34), (0.39, 0.255))
    box(ax, (0.04, 0.07), (0.57, 0.17), "Primary: free generation",
        "Same image + prompt at every checkpoint  →  generated SVG  →  parse and render\n"
        "SVG-Diagrams (within source)  |  VFIG-ID  |  VFIG-OOD\n"
        "Executable validity, output-limit failures, DINO and SSIM", GREEN, 8.4)
    box(ax, (0.66, 0.07), (0.30, 0.17), "Supporting diagnostic",
        "65 controlled image--SVG pairs\n"
        "Base, 20%, and final checkpoints\n"
        "Strict candidate matching", PURPLE, 8.4)
    arrow(ax, (0.74, 0.34), (0.81, 0.24), PURPLE)
    fig.tight_layout(pad=0.2)
    fig.savefig(out / "experiment_overview.pdf", bbox_inches="tight")
    fig.savefig(out / "experiment_overview.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
