"""
fig_v2_fig1.py — Fig. 1

Claim: averaging raw AlphaGenome scores over each tissue's tracks produces a convincing but false
"eye hub".
  a  (hero) tissue rank under the raw pooled summary for five variant sets: eye, kidney, metabolic and
     immune-disease loci, and random common variants (negative control). The eye is first in every
     column, including the control.  Source: data/analysis/A1A2_orig_ranking_*.csv,
     data/immune_analysis/ranking_orig_all_abs_raw.csv
  b  why it looked confirmed: open chromatin in trabecular meshwork cells is enriched at loci with high
     raw scores for EVERY tissue, not only the eye.  Source: data/analysis/A7_hTMC_or_raw.csv

    python scripts/fig_v2_fig1.py
"""
import os
import sys

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs  # noqa: E402
import step35_analysis as s35  # noqa: E402

fs.setup()
import matplotlib.pyplot as plt  # noqa: E402

EXCL = {"other", "eye_misfiled"}
SETS = [("eye", "Eye\nloci", 1123), ("kidney", "Kidney\nloci", 1811), ("metabolic", "Metabolic\nloci", 5150),
        ("immune", "Immune\nloci", 1154), ("null", "Random\nvariants", 2000)]


def scores():
    cols = {}
    for key, _, _ in SETS:
        if key == "immune":
            t = pd.read_csv(os.path.join(s35.NEW, "data", "immune_analysis", "ranking_orig_all_abs_raw.csv"))
            cols[key] = t.set_index("tissue").immune_loci
        else:
            t = pd.read_csv(os.path.join(s35.AN, f"A1A2_orig_ranking_{key}.csv"), keep_default_na=False)
            cols[key] = t.set_index("tissue").abs_raw
    s = pd.DataFrame(cols)
    return s.drop([t for t in EXCL if t in s.index])


def main():
    s = scores()
    rk = s.rank(ascending=False, method="min").astype(int)
    order = rk.mean(axis=1).sort_values().index
    rk, s = rk.loc[order], s.loc[order]
    ratio = s.loc["eye"] / s.drop("eye").max()

    fig = plt.figure(figsize=(fs.DOUBLE, 74 * fs.MM))
    ax = fig.add_axes([0.125, 0.14, 0.40, 0.66])
    ax.imshow(rk.values, cmap=fs.SEQUENTIAL, vmin=1, vmax=12, aspect="auto")
    for i in range(rk.shape[0]):
        for j in range(rk.shape[1]):
            v = rk.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=6,
                    color="white" if v <= 4 else fs.INK2, fontweight="bold" if rk.index[i] == "eye" else "normal")
    ax.add_patch(mpatches.Rectangle((-0.5, -0.5), rk.shape[1], 1, fill=False, ec=fs.EYE, lw=1.3, zorder=5))
    ax.axvline(3.5, color="white", lw=3)
    ax.set_xticks(range(len(SETS)), [lab for _, lab, _ in SETS])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks(range(len(order)), [fs.tissue(t) for t in order])
    ax.tick_params(axis="y", length=0)
    lab = ax.get_yticklabels()[list(order).index("eye")]
    lab.set_color(fs.EYE); lab.set_fontweight("bold")
    for sp in ax.spines.values():
        sp.set_visible(False)
    # group header above the columns
    ax.annotate("", xy=(-0.4, -1.95), xytext=(3.4, -1.95), xycoords="data", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.6))
    ax.text(1.5, -2.15, "Disease loci", ha="center", va="bottom", fontsize=6, color=fs.INK2)
    ax.text(4, -2.15, "Control", ha="center", va="bottom", fontsize=6, color=fs.INK2)
    # eye / next tissue under each column
    for j, (key, _, n) in enumerate(SETS):
        ax.text(j, len(order) - 0.35, f"{ratio[key]:.1f}×", ha="center", va="top", fontsize=6,
                color=fs.EYE, fontweight="bold")
        ax.text(j, len(order) + 0.45, f"n = {n:,}", ha="center", va="top", fontsize=5.5, color=fs.MUTED)
    ax.text(-0.62, len(order) - 0.35, "Eye ÷ next", ha="right", va="top", fontsize=6, color=fs.EYE)
    ax.set_ylim(len(order) - 0.5, -0.5)
    fs.panel_label(ax, "a", x=-0.115, y=0.955)
    fig.text(0.125, 0.955, "Tissue rank, raw pooled summary (1 = highest mean |raw score|)", fontsize=7,
             ha="left", va="bottom")

    # b: every tissue's raw score 'predicts' ocular open chromatin
    o = pd.read_csv(os.path.join(s35.AN, "A7_hTMC_or_raw.csv")).set_index("tissue").iloc[:, 0]
    o = o.drop([t for t in EXCL if t in o.index]).sort_values()
    bx = fig.add_axes([0.70, 0.14, 0.27, 0.66])
    y = np.arange(len(o))
    colors = [fs.EYE if t == "eye" else fs.NEUTRAL for t in o.index]
    bx.hlines(y, 0, o.values, color=colors, lw=1.2)
    bx.scatter(o.values, y, s=14, color=colors, zorder=3, lw=0)
    bx.axvline(1, color=fs.MUTED, lw=0.6, ls=(0, (2, 2)))
    bx.text(1.4, len(o) - 0.1, "OR = 1", fontsize=5.5, color=fs.MUTED, va="bottom", ha="left")
    bx.set_yticks(y, [fs.tissue(t) for t in o.index])
    lab = bx.get_yticklabels()[list(o.index).index("eye")]
    lab.set_color(fs.EYE); lab.set_fontweight("bold")
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_xlim(0, 21)
    bx.set_xticks([0, 5, 10, 15, 20])
    bx.set_xlabel("Odds ratio, top vs bottom quartile of each\ntissue's raw score (trabecular meshwork ATAC)")
    bx.set_ylim(-0.6, len(o) + 0.4)
    fs.panel_label(bx, "b", x=-0.13, y=0.955)
    fig.text(0.585, 0.955, "Open chromatin in ocular cells tracks every tissue's score", fontsize=7,
             ha="left", va="bottom")
    return fs.save(fig, "Fig1_false_hub")


if __name__ == "__main__":
    main()
