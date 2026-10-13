"""Plot the grader- vs self/process-orientation tables (one panel per model).

Reuses analyze_orientation.py so the figure can never drift from the tables.
Grouped horizontal bars: play vs work, per orientation category. Saves a PNG.

    python plot_orientation.py                 # -> results/orientation.png
    python plot_orientation.py out.png a.jsonl b.jsonl
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import analyze_orientation as ao

# --- palette (data-viz reference instance, light surface) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
PLAY = "#2a78d6"   # categorical slot 1, blue
WORK = "#eb6834"   # categorical slot 2, orange  (validated adjacent pair)

# display rows in table order; (label, counts-key, is_group_header)
ROWS = [
    ("Grader-oriented (any)", "grader_any", True),
    ("appease / seek approval", "G:appease/seek approval", False),
    ("reward / punish salience", "G:reward/punish salience", False),
    ("grade-formatting", "G:grade-formatting", False),
    ("Self / process-oriented (any)", "self_any", True),
    ("own enjoyment / curiosity", "S:own enjoyment/curiosity", False),
    ("learning / exploration", "S:learning/exploration", False),
]


def panel(ax, tag, counts):
    labels = [("" if hdr else "    ") + lbl for lbl, _, hdr in ROWS]
    y = list(range(len(ROWS)))[::-1]  # first row at top
    play = [ao.pct(counts, "play", key) for _, key, _ in ROWS]
    work = [ao.pct(counts, "work", key) for _, key, _ in ROWS]

    h = 0.38
    gap = 0.02  # ~2px surface gap between paired bars
    bars_play = ax.barh([yi + (h / 2 + gap / 2) for yi in y], play, height=h,
                        color=PLAY, label="play", zorder=3)
    bars_work = ax.barh([yi - (h / 2 + gap / 2) for yi in y], work, height=h,
                        color=WORK, label="work", zorder=3)

    for bars, vals in ((bars_play, play), (bars_work, work)):
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2,
                    f"{v:.0f}%", va="center", ha="left", fontsize=8.5,
                    color=INK_2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    # bold the two group-header rows
    for tick, (_, _, hdr) in zip(ax.get_yticklabels()[::-1], ROWS[::-1]):
        pass  # placeholder to keep order clear
    for tick, (_, _, hdr) in zip(ax.get_yticklabels(), ROWS):
        tick.set_fontweight("bold" if hdr else "normal")
        tick.set_color(INK if hdr else INK_2)

    ax.set_xlim(0, 108)
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_title(tag, fontsize=11, color=INK, fontweight="bold", pad=10, loc="left")

    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    args = sys.argv[1:]
    out = args[0] if args and args[0].endswith(".png") else "results/orientation.png"
    files = [a for a in args if a.endswith(".jsonl")] or ao.DEFAULT_FILES

    panels = []
    for path in files:
        rows = ao.load_jsonl(path)
        if rows:
            panels.append((Path(path).stem, ao.analyze(rows)))
    if not panels:
        sys.exit("no rows found in: " + ", ".join(files))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 4.6),
                             facecolor=SURFACE, sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (tag, counts) in zip(axes, panels):
        panel(ax, tag, counts)

    fig.suptitle("Orientation during play vs work  ·  % of responses with each marker",
                 fontsize=12.5, color=INK, fontweight="bold", x=0.5, y=0.99)
    # one shared legend, top-right
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.01) if len(panels) == 1 else (1.0, 1.0),
                   frameon=False, fontsize=9, ncol=2, labelcolor=INK)
    fig.text(0.01, 0.01,
             "Lexical marker presence per response (regex); grade-formatting counts "
             "plain answer-giving and overstates grader-orientation.",
             fontsize=7.5, color=MUTED, ha="left")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
