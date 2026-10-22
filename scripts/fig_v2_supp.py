"""
fig_v2_supp.py — Supplementary Figs. 1-6 (style in figstyle.py)

  S1  Immune-mediated diseases: a raw pooled summary, b calibrated DNase, c immune and eye rank by method
  S2  Maximum per tissue vs number of tracks, all variant sets, pooled and DNase only
  S3  Calibrated DNase scores (mean |q|) per tissue for eye, kidney and metabolic loci   [function s6]
  S4  Multi-disease "gradient", raw vs calibrated                                     [function s5]
  S5  Specificity against the gene-distance-matched null: a absolute scores centred per trait set,
      b specificity                                                                   [function s4]
  S6  Enformer: a raw-effect scale by assay, b raw pooled rank vs ChIP share, c calibrated pooled
                                                                                      [function s3]
  (numbered by order of first citation in the text)

    python scripts/fig_v2_supp.py            # all
    python scripts/fig_v2_supp.py 3 6        # selected
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs  # noqa: E402
import step35_analysis as s35  # noqa: E402
from fig_v2_main import DOM3, DOM_LABEL, EXCL, header, eye_ticklabel  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NEW, AN = s35.NEW, s35.AN
EN = os.path.join(NEW, "data", "enformer")
IA = os.path.join(NEW, "data", "immune_analysis")
SET_STYLE = {"eye": (fs.INK, "o"), "kidney": (fs.INK2, "s"), "metabolic": (fs.MUTED, "^"), "null": (fs.NEUTRAL, "D")}


def ranked_bars(ax, s, highlight=("eye",), extra=None, fmt="{:.4f}"):
    """Horizontal ranked bars, highlighted tissues coloured, others neutral; zoomed x for quantile scores."""
    s = s.sort_values()
    colors = [extra.get(t) if extra and t in extra else (fs.EYE if t in highlight else fs.NEUTRAL) for t in s.index]
    ax.barh(range(len(s)), s.values, color=colors, height=0.7, lw=0)
    ax.set_yticks(range(len(s)), [fs.tissue(t) for t in s.index])
    eye_ticklabel(ax, list(s.index))
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    lo, hi = s.min(), s.max()
    if lo > 0.3:
        ax.set_xlim(lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.08)
    ax.set_ylim(-0.6, len(s) - 0.4)


# ───────────────────────────────────── S1 immune ─────────────────────────────────────
def s1():
    fig = plt.figure(figsize=(fs.DOUBLE, 118 * fs.MM))
    for k, (f, title, xl, pos, letter) in enumerate([
            ("ranking_orig_all_abs_raw.csv", "Raw pooled summary (all assays)", "Mean |raw score|",
             [0.13, 0.57, 0.30, 0.33], "a"),
            ("ranking_dnase_rec_abs_q.csv", "Calibrated, DNase only", "Mean |quantile score|",
             [0.635, 0.57, 0.30, 0.33], "b")]):
        t = pd.read_csv(os.path.join(IA, f)).set_index("tissue").immune_loci
        t = t.drop([x for x in EXCL if x in t.index])
        ax = fig.add_axes(pos)
        ranked_bars(ax, t, extra={"immune": fs.IMMUNE})
        ax.set_xlabel(xl)
        if k == 0:
            ax.ticklabel_format(axis="x", style="plain")
            ax.set_xticks(ax.get_xticks()[::2])
        header(fig, pos[0], 0.955, letter, title, dx=-0.10)
    r = pd.read_csv(os.path.join(IA, "rank_summary.csv")).set_index("method")
    order = [("a. raw pooled (original scorers, original pipeline)", "Raw, all assays pooled"),
             ("raw DNase only (original scorer, uncalibrated)", "Raw, DNase only"),
             ("b. calibrated pooled (recommended, null pct)", "Calibrated, all assays pooled"),
             ("b'. calibrated pooled (original scorers, null pct)", "Calibrated (mean-difference scorer), pooled"),
             ("c. DNase |q| (AlphaGenome quantile)", "Calibrated, DNase only (|q|)"),
             ("c. DNase null pct", "Calibrated, DNase only (null percentile)"),
             ("c'. ATAC null pct", "Calibrated, ATAC only"),
             ("c'. RNA null pct", "Calibrated, RNA-seq only")]
    order = [(m, lab) for m, lab in order if m in r.index]
    cx = fig.add_axes([0.30, 0.09, 0.50, 0.34])
    y = np.arange(len(order))[::-1]
    for yi, (m, _) in zip(y, order):
        cx.scatter(r.loc[m, "immune rank"], yi + 0.12, s=16, color=fs.IMMUNE, lw=0.4, ec="white", zorder=3)
        if not np.isnan(r.loc[m, "eye rank"]):
            cx.scatter(r.loc[m, "eye rank"], yi - 0.12, s=14, marker="s", color=fs.EYE, lw=0.4, ec="white", zorder=3)
        cx.text(13.0, yi, f"{r.loc[m, 'P_boot(immune #1)']:.0%}", va="center", fontsize=6, color=fs.INK2)
    cx.text(13.0, y.max() + 0.75, "P(immune\nfirst)", va="bottom", fontsize=5.5, color=fs.INK2)
    cx.set_yticks(y, [lab for _, lab in order])
    cx.tick_params(axis="y", length=0)
    cx.spines["left"].set_visible(False)
    cx.set_xlim(0.5, 12.5)
    cx.set_xticks([1, 3, 6, 9, 12])
    cx.set_xlabel("Rank among 12 tissues (eye has no ATAC or RNA-seq tracks)")
    for yy in y:
        cx.axhline(yy, color=fs.GRID, lw=0.5, zorder=0)
    handles = [Line2D([], [], ls="", marker="o", ms=4, color=fs.IMMUNE, label="Immune cells"),
               Line2D([], [], ls="", marker="s", ms=3.6, color=fs.EYE, label="Eye")]
    cx.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=5.5, borderaxespad=0.2)
    header(fig, 0.30, 0.48, "c", "Rank of immune cells and of the eye, by method", dx=-0.27)
    return fs.save(fig, "SuppFig1_immune")


# ───────────────────────────────────── S2 max per tissue ─────────────────────────────────────
def s2():
    t = pd.read_csv(os.path.join(AN, "step45_max_per_tissue_scores.csv"), keep_default_na=False)
    fig = plt.figure(figsize=(fs.DOUBLE, 100 * fs.MM))
    xs = [0.085, 0.325, 0.565, 0.805]
    rows = [("all recommended scorers pooled", 0.56, "All assays pooled"), ("DNase only", 0.10, "DNase only")]
    for i, (tr, y0, rl) in enumerate(rows):
        for j, d in enumerate(["eye", "kidney", "metabolic", "null"]):
            ax = fig.add_axes([xs[j], y0, 0.175, 0.31])
            x = t[(t.loci == d) & (t.tracks == tr)].set_index("tissue")
            x = x.drop([k for k in EXCL if k in x.index])
            oth = x.drop("eye")
            ax.scatter(oth.n_tracks, oth.max_q, s=10, color=fs.NEUTRAL, lw=0.4, ec="white", zorder=3)
            ax.scatter(x.n_tracks["eye"], x.max_q["eye"], s=16, color=fs.EYE, lw=0.4, ec="white", zorder=4)
            ax.set_xscale("log")
            ticks = [v for v in (4, 10, 25, 50, 100) if x.n_tracks.min() * 0.8 <= v <= x.n_tracks.max() * 1.3]
            ax.set_xticks(ticks, [str(v) for v in ticks])
            ax.minorticks_off()
            rho = spearmanr(x.n_tracks, x.max_q).correlation
            ax.text(0.97, 0.05, f"ρ = {rho:.2f}", transform=ax.transAxes, ha="right", fontsize=6)
            if i == 1:
                ax.set_xlabel("Tracks in tissue (log)")
            if j == 0:
                ax.set_ylabel(f"{rl}\nmean max |q|")
            if i == 0:
                fig.text(xs[j], 0.955, DOM_LABEL[d], fontsize=7, ha="left", va="bottom")
        fig.text(0.012, y0 + 0.33, "ab"[i], fontsize=8, fontweight="bold", va="bottom")
    return fs.save(fig, "SuppFig3_max_aggregation")


# ───────────────────────────────────── S3 Enformer ─────────────────────────────────────
def s3():
    fig = plt.figure(figsize=(fs.DOUBLE, 80 * fs.MM))
    tm_all = pd.read_csv(os.path.join(EN, "track_tissue_map.csv")).set_index("index")
    files = sorted(glob.glob(os.path.join(EN, "scores", "null_*.npz")))
    R = np.vstack([np.abs(np.load(f)["alt_c"] - np.load(f)["ref_c"]) for f in files])
    tmr = tm_all.loc[np.arange(R.shape[1])]
    m = R.mean(0)
    ax = fig.add_axes([0.085, 0.17, 0.22, 0.66])
    rows = [("CAGE", "CAGE"), ("DNASE", "DNase"), ("CHIP", "ChIP")]
    rng = np.random.default_rng(3)
    for i, (key, lab) in enumerate(rows):
        v = m[(tmr.assay == key).values & (tmr.tissue != "other").values]
        ax.scatter(v, i + rng.uniform(-0.28, 0.28, len(v)), s=1.5, color=fs.NEUTRAL, lw=0, rasterized=True)
        ax.plot([np.median(v)] * 2, [i - 0.36, i + 0.36], color=fs.INK, lw=1.0)
    ax.set_xscale("log")
    lo, hi = np.floor(np.log10(m[m > 0].min())), np.ceil(np.log10(m.max()))
    ticks = [10.0 ** e for e in range(int(lo), int(hi) + 1)]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(range(3), [lab for _, lab in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Mean |raw effect| per track,\nrandom variants (log)")
    header(fig, 0.085, 0.93, "a", "Effect scale by assay", dx=0.0)

    tm = tm_all[tm_all.tissue != "other"]
    chip = tm.groupby("tissue").assay.apply(lambda a: (a == "CHIP").mean())
    for k, (kind, sets, pos, letter, title) in enumerate([
            ("raw", ["eye", "kidney", "metabolic", "null"], [0.42, 0.17, 0.25, 0.66], "b", "Raw effects, pooled"),
            ("cal", DOM3, [0.735, 0.17, 0.25, 0.66], "c", "Calibrated, then pooled")]):
        bx = fig.add_axes(pos)
        for d in sets:
            s = pd.read_csv(os.path.join(EN, f"step48_{kind}_pooled_{d}.csv"), index_col=0)["score"]
            rk = s.rank(ascending=False)
            c, mk = SET_STYLE[d]
            bx.scatter(chip[rk.index], rk, s=10, marker=mk, color=c, lw=0.3, ec="white", zorder=3)
            bx.scatter(chip["eye"], rk["eye"], s=24, marker=mk, facecolor="none", ec=fs.EYE, lw=0.9, zorder=4)
        bx.set_ylim(12.8, 0.2)
        bx.set_yticks([1, 3, 6, 9, 12])
        bx.set_xlim(0, 1)
        bx.set_xlabel("Share of ChIP-seq tracks")
        bx.set_ylabel("Tissue rank")
        header(fig, pos[0], 0.93, letter, title, dx=-0.05)
    handles = [Line2D([], [], ls="", marker=SET_STYLE[d][1], ms=3.6, color=SET_STYLE[d][0], label=DOM_LABEL[d])
               for d in ["eye", "kidney", "metabolic", "null"]]
    handles.append(Line2D([], [], ls="", marker="o", ms=4.5, mfc="none", mec=fs.EYE, label="Eye (tissue)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.70, 0.005), ncol=5, fontsize=5.5,
               handletextpad=0.3, columnspacing=1.0)
    return fs.save(fig, "SuppFig6_enformer")


# ───────────────────────────────────── S4 specificity, matched null ─────────────────────────────────────
def s4():
    c = pd.read_csv(os.path.join(NEW, "data", "null_matched_analysis", "c_rankings.csv"))
    c = c[c.method == "calibrated DNase only, matched null"].pivot(index="tissue", columns="domain", values="score")[DOM3]
    absolute = (c - c.mean()) * 100
    sp = pd.read_csv(os.path.join(AN, "step49_specificity_all.csv"))
    spec = sp[(sp.calibration == "null pct, matched null") & (sp.metric == "mean")].pivot(
        index="tissue", columns="loci", values="specificity")[DOM3] * 100
    order = ["eye"] + sorted([t for t in spec.index if t != "eye"], key=lambda t: -spec.loc[t, "eye"])
    absolute, spec = absolute.loc[order], spec.loc[order]
    lim = float(np.ceil(max(absolute.abs().max().max(), spec.abs().max().max())))
    fig = plt.figure(figsize=(fs.DOUBLE, 92 * fs.MM))
    for k, (M, title, x0, letter) in enumerate([
            (absolute, "Calibrated score, centred within each trait set", 0.13, "a"),
            (spec, "Specificity: each tissue vs itself across trait sets", 0.60, "b")]):
        ax = fig.add_axes([x0, 0.16, 0.30, 0.66])
        im = ax.imshow(M.values, cmap=fs.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(v) > 0.65 * lim else fs.INK)
        ax.set_xticks(range(3), ["Eye\nloci", "Kidney\nloci", "Metabolic\nloci"])
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        ax.set_yticks(range(len(order)), [fs.tissue(t) for t in order])
        eye_ticklabel(ax, order)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        header(fig, x0, 0.945, letter, title, dx=-0.10)
    cb_ax = fig.add_axes([0.60, 0.075, 0.30, 0.018])
    cb = fig.colorbar(im, cax=cb_ax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=5.5)
    cb.set_label("Percentile points (DNase-seq, percentile against the matched null)", fontsize=5.5)
    return fs.save(fig, "SuppFig5_specificity_matched_null")


# ───────────────────────────────────── S5 gradient ─────────────────────────────────────
def s5():
    g = pd.read_csv(os.path.join(AN, "A6_gradient.csv"))
    g = g[g.tissue != "other"]
    piv = g.pivot(index="tissue", columns="score", values="median ratio 3/1")
    raw, cal = "raw (original pipeline)", "calibrated DNase |q|"
    piv = piv.dropna(subset=[raw]).sort_values(raw)
    fig = plt.figure(figsize=(fs.SINGLE, 78 * fs.MM))
    ax = fig.add_axes([0.36, 0.17, 0.58, 0.72])
    y = np.arange(len(piv))
    ax.hlines(y, piv[cal], piv[raw], color=fs.GRID, lw=2)
    ax.scatter(piv[raw], y, s=14, color=fs.RAW, zorder=3, lw=0.3, ec="white", label="Raw pooled summary")
    ax.scatter(piv[cal], y, s=14, color=fs.STRONG, zorder=3, lw=0.3, ec="white", label="Calibrated DNase, mean |q|")
    ax.axvline(1, color=fs.MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_yticks(y, [fs.tissue(t) for t in piv.index])
    eye_ticklabel(ax, list(piv.index))
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Median score, three-domain / single-domain loci")
    ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=1, fontsize=5.5, handletextpad=0.2)
    return fs.save(fig, "SuppFig2_gradient", )


# ───────────────────────────────────── S6 calibrated DNase rankings ─────────────────────────────────────
def s6():
    lt, _ = s35.aggregate()
    dom = s35.domain_map(s35.read_variants())
    fig = plt.figure(figsize=(fs.DOUBLE, 70 * fs.MM))
    for k, d in enumerate(DOM3):
        ids = set(dom.loc[dom.domain == d, "id"])
        x = lt[(lt.scorer == "dnase_rec") & lt.id.isin(ids) & (lt.tissue != "other")]
        s = x.groupby("tissue").abs_q.mean()
        x0 = 0.145 + k * 0.32
        ax = fig.add_axes([x0, 0.17, 0.19, 0.70])
        ranked_bars(ax, s, extra={"eye_misfiled": fs.NEUTRAL_LIGHT})
        labels = [lab.get_text() for lab in ax.get_yticklabels()]
        ax.set_yticklabels([("Choroid plexus*" if lab.startswith("Choroid") else lab) for lab in labels])
        eye_ticklabel(ax, [("eye" if lab == "Eye" else lab) for lab in labels])
        ax.set_xlabel("Mean |q|, DNase-seq")
        header(fig, x0, 0.93, "abc"[k], DOM_LABEL[d], dx=-0.10)
    fig.text(0.145, 0.02, "*Choroid plexus epithelial cell, placed in the eye category by the ontology-based grouping",
             fontsize=5.5, color=fs.MUTED)
    return fs.save(fig, "SuppFig4_dnase_rankings")


FIGS = {"1": s1, "2": s2, "3": s3, "4": s4, "5": s5, "6": s6}

if __name__ == "__main__":
    for k in sys.argv[1:] or list(FIGS):
        FIGS[k]()
