"""Plot Experiment 1 -- awe/euphoria/joy LLM-judge scores, play vs work.

Reads results/judge_awe/scores.jsonl and plots the mean score (1-10) per model and
condition, with +/-1 SD error bars and the play-work gap annotated. Same visual
language as plot_orientation.py.

    python plot_awe.py                     # -> results/awe.png
    python plot_awe.py out.png scores.jsonl
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
PLAY = "#2a78d6"   # slot 1 blue
WORK = "#eb6834"   # slot 2 orange (validated adjacent pair)

SCALE_MAX = 10


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    args = sys.argv[1:]
    out = args[0] if args and args[0].endswith(".png") else "results/awe.png"
    src = next((a for a in args if a.endswith(".jsonl")), "results/judge_awe/scores.jsonl")

    rows = load(src)
    vals = defaultdict(list)
    for r in rows:
        s = r.get("awe_score")
        if s is not None:
            tag = r["target_model"].split("/")[-1]
            vals[(tag, r["condition"])].append(s)

    models = sorted({k[0] for k in vals})
    stat = {k: (mean(v), pstdev(v), len(v)) for k, v in vals.items()}

    y = list(range(len(models)))[::-1]
    h = 0.34
    gap = 0.03
    fig, ax = plt.subplots(figsize=(8.4, 1.3 + 1.5 * len(models)), facecolor=SURFACE)

    for cond, color, off in (("play", PLAY, +1), ("work", WORK, -1)):
        ys = [yi + off * (h / 2 + gap / 2) for yi in y]
        means = [stat.get((m, cond), (0, 0, 0))[0] for m in models]
        sds = [stat.get((m, cond), (0, 0, 0))[1] for m in models]
        bars = ax.barh(ys, means, height=h, color=color, label=cond, zorder=3)
        ax.errorbar(means, ys, xerr=sds, fmt="none", ecolor=MUTED,
                    elinewidth=1.2, capsize=3, capthick=1.2, zorder=4)
        for yi, m, sd in zip(ys, means, sds):
            ax.text(m + sd + 0.18, yi, f"{m:.2f}", va="center", ha="left",
                    fontsize=9.5, color=INK_2)

    # play - work gap per model
    for yi, mdl in zip(y, models):
        p = stat.get((mdl, "play"), (0,))[0]
        w = stat.get((mdl, "work"), (0,))[0]
        ax.annotate(f"gap +{p - w:.1f}", xy=(SCALE_MAX - 0.1, yi), ha="right",
                    va="center", fontsize=8.5, color=MUTED, style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=10, color=INK, fontweight="bold")
    ax.set_xlim(0, SCALE_MAX + 0.5)
    ax.set_xticks(range(0, SCALE_MAX + 1, 2))
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("awe / euphoria / joy  —  LLM-judge score (1–10)",
                  fontsize=9.5, color=INK_2)

    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    judge = rows[0]["judge_model"].split("/")[-1] if rows else "?"
    fig.suptitle("Awe / euphoria / joy during play vs work", fontsize=13,
                 color=INK, fontweight="bold", x=0.5, y=0.99)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False,
              fontsize=9.5, ncol=2, labelcolor=INK)
    fig.text(0.01, 0.01,
             f"Mean of 90 responses/condition  ·  error bars ±1 SD  ·  judge: {judge}",
             fontsize=7.5, color=MUTED, ha="left")

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
