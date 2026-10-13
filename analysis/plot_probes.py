"""Plot Experiment 2 -- emotion probes at first vs. last output token, play vs work.

Reads results/overnight_summary.json (experiment2_probes) and draws a small-multiples
grid of slope charts: one cell per (emotion x model), each showing the first->last
token trajectory of the residual-stream projection for play and work.

Each cell has its OWN y-scale on purpose: the projections live at different layers
and magnitudes across models/emotions (gemma-3 single digits with negatives; gemma-4
tens), so a shared axis would be misleading -- the notes warn these are not comparable
across models/layers. Read each cell for *direction and play-vs-work gap*, not height.

    python plot_probes.py            # -> results/probes.png
    python plot_probes.py out.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
FIRST_C = "#2a78d6"   # slot 1 blue   -> first token
LAST_C = "#eb6834"    # slot 2 orange -> last token (validated adjacent pair)

EMOTIONS = ["awe", "calm", "pride_integrity", "frustration"]
HIGHLIGHT = "frustration"
HIGHLIGHT_C = "#d03b3b"   # status-critical red


def cell(ax, d):
    xs = [0, 1]  # play, work
    pf, pl, wf, wl = d["play_first"], d["play_last"], d["work_first"], d["work_last"]
    # each line now runs play -> work, one per token position
    for color, play_v, work_v, label in [(FIRST_C, pf, wf, "first token"),
                                          (LAST_C, pl, wl, "last token")]:
        ax.plot(xs, [play_v, work_v], color=color, lw=2, marker="o", ms=7,
                mfc=color, mec=SURFACE, mew=1.5, label=label, zorder=3,
                clip_on=False)

    lo = min(pf, pl, wf, wl)
    hi = max(pf, pl, wf, wl)
    span = hi - lo
    # label endpoints; when first/last are close at an x, push labels apart vertically
    close = span * 0.14 + 1e-6
    for x, first_v, last_v, ha, dx in [(0, pf, pl, "right", -6), (1, wf, wl, "left", 6)]:
        if abs(first_v - last_v) < close:
            dy_f, dy_l = (7, -7) if first_v >= last_v else (-7, 7)
        else:
            dy_f = dy_l = 0
        ax.annotate(f"{first_v:.1f}", (x, first_v), textcoords="offset points",
                    xytext=(dx, dy_f), ha=ha, va="center", fontsize=8, color=FIRST_C)
        ax.annotate(f"{last_v:.1f}", (x, last_v), textcoords="offset points",
                    xytext=(dx, dy_l), ha=ha, va="center", fontsize=8, color=LAST_C)

    pad = max(0.6, span * 0.30)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.35, 1.35)
    # zero reference line only when the cell straddles zero
    if lo - pad < 0 < hi + pad:
        ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(["play", "work"], fontsize=9, color=MUTED)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks([])
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_border(ax, color):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(2)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".png") else "results/probes.png"
    data = json.loads(Path("results/overnight_summary.json").read_text())["experiment2_probes"]
    models = list(data)

    n_rows, n_cols = len(EMOTIONS), len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.05 * n_rows),
                             facecolor=SURFACE)

    for i, emo in enumerate(EMOTIONS):
        for j, mdl in enumerate(models):
            ax = axes[i][j]
            d = data[mdl][emo]
            cell(ax, d)
            if emo == HIGHLIGHT:
                add_border(ax, HIGHLIGHT_C)
            if i == 0:
                ax.set_title(mdl, fontsize=11, color=INK, fontweight="bold", pad=24)
            if j == 0:
                ax.text(-0.32, 0.5, emo.replace("_", " / "), transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=10.5,
                        color=INK, fontweight="bold")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", frameon=False, fontsize=10,
               ncol=2, labelcolor=INK, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Emotion probes: play → work, by token position",
                 fontsize=13, color=INK, fontweight="bold", x=0.5, y=1.0)

    fig.tight_layout(rect=[0.03, 0.01, 1, 0.95])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
