"""
fig_v2_main.py — Figs. 2-5 (one claim per figure)

  Fig. 1  A convincing but false eye hub                       (fig_v2_fig1.py)
  Fig. 2  The hub is set by assay composition
          a raw-score scale by assay (eye tracks marked) | b assay mix per tissue |
          c tissue score vs chromatin share | d random five-DNase-track "tissues"
  Fig. 3  Each obvious repair fails in its own way
          a the eye's rank for the same loci under four comparisons | b maximum per tissue vs track count |
          c, d the same composition effect in Enformer, raw and calibrated
  Fig. 4  Comparing each tissue with itself across trait sets
          a specificity heatmap (hero) | b eye rank, absolute vs specificity, over six calibrations/summaries |
          c the eye's specificity with 95% intervals
  Fig. 5  Recovery of independently expected tissues
          a nine traits | b eye traits | c immune-mediated diseases, rank of immune cells per disease

All sizes are final (183 mm wide); style from figstyle.py; each figure is checked by figure_qa.py.

    python scripts/fig_v2_main.py            # all main figures
    python scripts/fig_v2_main.py 3 4        # selected figures
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs  # noqa: E402
import step35_analysis as s35  # noqa: E402

fs.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NEW, AN = s35.NEW, s35.AN
EN = os.path.join(NEW, "data", "enformer")
IA = os.path.join(NEW, "data", "immune_analysis")
DOM3 = ["eye", "kidney", "metabolic"]
DOM_LABEL = {"eye": "Eye loci", "kidney": "Kidney loci", "metabolic": "Metabolic loci", "null": "Random variants"}
EXCL = {"other", "eye_misfiled"}


def header(fig, x, y, letter, title, dx=0.0):
    """Panel letter + short panel title on one baseline (figure coordinates)."""
    fig.text(x - 0.012 + dx, y, letter, fontsize=8, fontweight="bold", ha="right", va="bottom")
    if title:
        fig.text(x + dx, y, title, fontsize=7, ha="left", va="bottom")


def eye_ticklabel(ax, keys, axis="y"):
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    if "eye" in keys:
        lab = labels[list(keys).index("eye")]
        lab.set_color(fs.EYE)
        lab.set_fontweight("bold")


def csv(*p, **kw):
    return pd.read_csv(os.path.join(NEW, *p), keep_default_na=False, **kw)


# ───────────────────────────────────────────── Fig. 2 ─────────────────────────────────────────────
def assay_composition(lt):
    """Tracks per tissue and assay as scored (both strands of stranded RNA-seq separately; 12 named tissues)."""
    one = lt[(lt.id == lt.id.iloc[0]) & lt.scorer.isin(s35.ORIG)].copy()
    one["tissue"] = one.tissue.replace({"eye_misfiled": "eye"})
    n = one.pivot_table(index="tissue", columns="scorer", values="n_tracks", aggfunc="sum", fill_value=0)
    n = n.rename(columns={"dnase_orig": "dnase", "atac_orig": "atac", "rna_orig": "rna_seq"})
    return n.drop("other", errors="ignore").astype(int)


def fig2():
    lt, td = s35.aggregate()
    fig = plt.figure(figsize=(fs.DOUBLE, 118 * fs.MM))

    # a: per-track mean |raw| over GWAS loci, by assay
    g = td[td.domain.isin(DOM3) & td.scorer.isin(s35.ORIG)]
    pt = g.groupby(["scorer", "track_name", "tissue"])[["sum_raw", "n"]].sum()
    pt = (pt.sum_raw / pt.n).rename("y").reset_index()
    ax = fig.add_axes([0.105, 0.625, 0.36, 0.28])
    rows = [("rna_orig", "RNA-seq"), ("atac_orig", "ATAC"), ("dnase_orig", "DNase")]
    rng = np.random.default_rng(7)
    for i, (sc, lab) in enumerate(rows):
        z = pt[pt.scorer == sc]
        j = rng.uniform(-0.28, 0.28, len(z))
        other = ~z.tissue.isin(["eye", "eye_misfiled"])
        ax.scatter(z.y[other], i + j[other], s=2.5, color=fs.NEUTRAL, lw=0, alpha=0.9, rasterized=True)
        ax.plot([z.y.median()] * 2, [i - 0.36, i + 0.36], color=fs.INK, lw=1.0, zorder=4)
        e = z[z.tissue == "eye"]
        ax.scatter(e.y, np.full(len(e), i) + j[z.tissue.values == "eye"], s=12, color=fs.EYE, lw=0.5,
                   ec="white", zorder=5)
        m = z[z.tissue == "eye_misfiled"]
        ax.scatter(m.y, np.full(len(m), i) + j[z.tissue.values == "eye_misfiled"], s=12, facecolor="white",
                   ec=fs.EYE, lw=0.8, zorder=5)
    ax.set_xscale("log")
    ax.set_xticks([0.0005, 0.001, 0.002, 0.005, 0.01], ["0.0005", "0.001", "0.002", "0.005", "0.01"])
    ax.minorticks_off()
    ax.set_yticks(range(3), [lab for _, lab in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Mean |raw score| per track, GWAS loci (log scale)")
    ax.set_ylim(-0.6, 2.6)
    handles = [Line2D([], [], ls="", marker="o", ms=3.5, color=fs.EYE, mec="white", label="Eye tracks"),
               Line2D([], [], ls="", marker="o", ms=3.5, mfc="white", mec=fs.EYE, label="Choroid plexus (filed as eye)"),
               Line2D([], [], color=fs.INK, lw=1.0, label="Median")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.02, 1.0), ncol=3, fontsize=5.5,
              handletextpad=0.3, columnspacing=0.9, borderaxespad=0.2)
    header(fig, 0.105, 0.955, "a", "Raw effects are set by the assay")

    # b: assay mix per tissue, as scored by the original pipeline (both strands of a stranded RNA-seq
    # library are separate tracks; the misfiled choroid plexus track counts as eye, as it did)
    comp = assay_composition(lt)
    share = comp.div(comp.sum(axis=1), axis=0)
    chrom = share["dnase"] + share["atac"]
    order = chrom.sort_values().index
    bx = fig.add_axes([0.665, 0.625, 0.27, 0.28])
    left = np.zeros(len(order))
    for key, lab in [("dnase", "DNase"), ("atac", "ATAC"), ("rna_seq", "RNA-seq")]:
        v = share.loc[order, key].values if key in share else np.zeros(len(order))
        bx.barh(range(len(order)), v, left=left, color=fs.ASSAY[lab], height=0.72, lw=0, label=lab)
        left += v
    for i, t in enumerate(order):
        bx.text(1.02, i, f"{int(comp.loc[t].sum())}", va="center", fontsize=5.5, color=fs.MUTED)
    bx.text(1.02, len(order) - 0.2, "Tracks", va="bottom", fontsize=5.5, color=fs.MUTED)
    bx.set_yticks(range(len(order)), [fs.tissue(t) for t in order])
    eye_ticklabel(bx, order)
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_xlim(0, 1)
    bx.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    bx.set_xlabel("Share of the tissue's tracks")
    bx.set_ylim(-0.6, len(order) - 0.4)
    bx.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, fontsize=5.5, handlelength=0.9,
              handleheight=0.9, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
    header(fig, 0.555, 0.955, "b", "Tissues differ in assay mix")

    # c: tissue score vs chromatin share
    pub = s35.as_published(lt)
    ids = s35.domain_map(s35.read_variants()).query("domain != 'null'").id
    x = pub[(pub.scorer == "orig_all") & pub.id.isin(ids)]
    score = x.groupby("tissue").abs_raw.mean().drop([t for t in EXCL if t in x.tissue.unique()], errors="ignore")
    both = pd.concat([chrom.rename("frac"), score.rename("score")], axis=1).dropna()
    rho = spearmanr(both.frac, both.score).correlation
    pd.concat([comp, chrom.rename("chromatin_share"), score.rename("tissue_score_gwas")], axis=1).dropna().to_csv(
        os.path.join(AN, "fig2_assay_composition.csv"))
    cx = fig.add_axes([0.105, 0.105, 0.36, 0.34])
    oth = both.drop("eye")
    cx.scatter(oth.frac, oth.score * 1e3, s=14, color=fs.NEUTRAL, lw=0.5, ec="white", zorder=3)
    cx.scatter(both.frac["eye"], both.score["eye"] * 1e3, s=22, color=fs.EYE, lw=0.5, ec="white", zorder=4)
    cx.annotate("Eye (5 DNase tracks)", (both.frac["eye"], both.score["eye"] * 1e3), xytext=(-6, -2),
                textcoords="offset points", ha="right", va="top", fontsize=6, color=fs.EYE)
    for t, off in [("skin", (-4, 3)), ("liver", (4, 1)), ("reproductive", (5, -1))]:
        cx.annotate(fs.tissue(t), (both.frac[t], both.score[t] * 1e3), xytext=off, textcoords="offset points",
                    fontsize=5.5, color=fs.INK2, ha="right" if off[0] < 0 else ("center" if off[0] == 0 else "left"))
    cx.text(0.03, 0.97, f"Spearman ρ = {rho:.2f}", transform=cx.transAxes, va="top", fontsize=6)
    cx.set_xlabel("Share of chromatin tracks (DNase + ATAC)")
    cx.set_ylabel("Tissue score, GWAS loci\n(mean |raw score| × 10³)")
    cx.set_xlim(0.15, 1.05)
    header(fig, 0.105, 0.475, "c", "Tissue score follows the assay mix")

    # d: random five-DNase-track 'tissues', random variants
    draws = np.load(os.path.join(AN, "A3_fake_eye_draws_null.npy")) * 1e3
    # same basis as the draws (step35 A3): track-level tissue means, eye = 5 DNase tracks, 'other' included
    tdp = td.copy()
    tdp["tissue"] = tdp.tissue.replace({"eye_misfiled": "eye"})
    real = tdp[(tdp.domain == "null") & tdp.scorer.isin(s35.ORIG)].groupby("tissue").abs_raw.mean() * 1e3
    best = real.drop("eye").max()
    frac = (draws > best).mean()
    dx = fig.add_axes([0.665, 0.105, 0.30, 0.34])
    dx.hist(draws, bins=40, color=fs.NEUTRAL, lw=0)
    top = dx.get_ylim()[1]
    dx.set_ylim(0, top * 1.35)
    dx.axvline(real["eye"], color=fs.EYE, lw=1.2)
    dx.axvline(best, color=fs.INK2, lw=0.8, ls=(0, (3, 2)))
    dx.text(real["eye"], top * 1.3, "Eye ", color=fs.EYE, ha="right", va="top", fontsize=6)
    dx.text(best, top * 1.3, f" Best other tissue\n ({fs.tissue(real.drop('eye').idxmax()).lower()})",
            color=fs.INK2, ha="left", va="top", fontsize=5.5)
    dx.text(0.47, 0.76, f"{frac:.0%} of random sets\nexceed every real\nnon-eye tissue",
            transform=dx.transAxes, ha="center", va="center", fontsize=6)
    dx.set_xlabel("Mean |raw score| of the set, random variants (× 10³)")
    dx.set_ylabel("Random five-DNase-track sets")
    header(fig, 0.555, 0.475, "d", "Any five DNase tracks form a 'hub'")
    return fs.save(fig, "Fig2_mechanism")


# ───────────────────────────────────────────── Fig. 3 ─────────────────────────────────────────────
def fig3():
    fig = plt.figure(figsize=(fs.DOUBLE, 112 * fs.MM))
    st = pd.read_csv(os.path.join(AN, "step49_supp_table_eye_ranks.csv")).set_index("method")
    cols = [("Raw, all assays pooled", "Raw scores,\nall assays pooled"),
            ("Calibrated, all assays pooled, random null", "Calibrated (random\nnull), pooled"),
            ("Calibrated, all assays pooled, matched null", "Calibrated (matched\nnull), pooled"),
            ("Calibrated, DNase only, AlphaGenome |q| (common-variant background), mean", "Calibrated,\nDNase only")]
    ax = fig.add_axes([0.085, 0.60, 0.44, 0.30])
    styles = {"eye": (fs.INK, "o", -0.07), "kidney": (fs.INK2, "s", 0.0), "metabolic": (fs.MUTED, "^", 0.07)}
    xs = np.arange(len(cols))
    for d in DOM3:
        c, m, off = styles[d]
        y = [st.loc[k, f"eye rank, {d} loci"] for k, _ in cols]
        ax.plot(xs + off, y, color=c, lw=0.9, marker=m, ms=4, mec="white", mew=0.5, zorder=3)
        ax.text(xs[-1] + 0.18, y[-1], DOM_LABEL[d], va="center", fontsize=6, color=c)
    ax.set_xticks(xs, [lab for _, lab in cols])
    ax.set_ylim(12.8, 0.2)
    ax.set_yticks([1, 3, 6, 9, 12])
    ax.set_xlim(-0.3, len(cols) - 0.3)
    ax.set_ylabel("Eye rank among 12 tissues")
    ax.axhspan(0.5, 1.5, color=fs.EYE, alpha=0.08, lw=0)
    header(fig, 0.085, 0.955, "a", "Same GWAS loci, four comparisons")

    # b: maximum over a tissue's tracks
    mx = pd.read_csv(os.path.join(AN, "step45_max_per_tissue_scores.csv"), keep_default_na=False)
    mx = mx[(mx.loci == "eye") & (mx.tracks == "all recommended scorers pooled")].set_index("tissue")
    mx = mx.drop([t for t in EXCL if t in mx.index])
    rho = spearmanr(mx.n_tracks, mx.max_q).correlation
    bx = fig.add_axes([0.70, 0.60, 0.27, 0.30])
    oth = mx.drop("eye")
    bx.scatter(oth.n_tracks, oth.max_q, s=14, color=fs.NEUTRAL, lw=0.5, ec="white", zorder=3)
    bx.scatter(mx.n_tracks["eye"], mx.max_q["eye"], s=22, color=fs.EYE, lw=0.5, ec="white", zorder=4)
    bx.annotate("Eye (4 tracks)", (mx.n_tracks["eye"], mx.max_q["eye"]), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=6, color=fs.EYE)
    bx.set_xscale("log")
    bx.set_xticks([4, 10, 25, 50, 100], ["4", "10", "25", "50", "100"])
    bx.minorticks_off()
    bx.text(0.97, 0.05, f"Spearman ρ = {rho:.2f}", transform=bx.transAxes, ha="right", fontsize=6)
    bx.set_xlabel("Tracks in the tissue (log scale)")
    bx.set_ylabel("Mean over loci of the largest\n|q| among the tissue's tracks")
    header(fig, 0.70, 0.955, "b", "Maximum per tissue, eye loci", dx=-0.07)

    # c, d: Enformer
    tm = pd.read_csv(os.path.join(EN, "track_tissue_map.csv"))
    tm = tm[tm.tissue != "other"]
    chip = tm.groupby("tissue").assay.apply(lambda a: (a == "CHIP").mean())
    for k, (fname, title, letter, xpos) in enumerate([
            ("step48_raw_pooled_eye.csv", "Enformer: raw effects, pooled", "c", 0.085),
            ("step48_cal_pooled_eye.csv", "Enformer: calibrated, then pooled", "d", 0.585)]):
        s = pd.read_csv(os.path.join(EN, fname), index_col=0)["score"]
        rk = s.rank(ascending=False)
        r = spearmanr(s, chip[s.index]).correlation
        cx = fig.add_axes([xpos, 0.10, 0.36, 0.30])
        oth = rk.drop("eye")
        cx.scatter(chip[oth.index], oth, s=14, color=fs.NEUTRAL, lw=0.5, ec="white", zorder=3)
        cx.scatter(chip["eye"], rk["eye"], s=22, color=fs.EYE, lw=0.5, ec="white", zorder=4)
        cx.annotate("Eye", (chip["eye"], rk["eye"]), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=6, color=fs.EYE)
        top = rk.idxmin()
        if top != "eye":
            cx.annotate(fs.tissue(top), (chip[top], rk[top]), xytext=(-5, 0), textcoords="offset points",
                        va="center", ha="right", fontsize=5.5, color=fs.INK2)
        cx.set_ylim(12.8, 0.2)
        cx.set_yticks([1, 3, 6, 9, 12])
        cx.set_xlim(0, 1)
        cx.text(0.97, 0.05 if k == 0 else 0.95, f"ρ (score, ChIP share) = {r:.2f}", transform=cx.transAxes,
                ha="right", va="bottom" if k == 0 else "top", fontsize=6)
        cx.set_xlabel("Share of the tissue's tracks that are ChIP-seq")
        cx.set_ylabel("Tissue rank, eye loci")
        header(fig, xpos, 0.435, letter, title)
    return fs.save(fig, "Fig3_repairs")


# ───────────────────────────────────────────── Fig. 4 ─────────────────────────────────────────────
CALS = [("AlphaGenome |q| (common-variant background)", "AlphaGenome |q|"),
        ("null pct, random null", "Random null"), ("null pct, matched null", "Matched null")]


def fig4():
    fig = plt.figure(figsize=(fs.DOUBLE, 104 * fs.MM))
    sp = pd.read_csv(os.path.join(AN, "step49_specificity_all.csv"))
    main = sp[(sp.calibration == CALS[0][0]) & (sp.metric == "strong fraction")]
    M = main.pivot(index="tissue", columns="loci", values="specificity")[DOM3] * 100
    lo = main.pivot(index="tissue", columns="loci", values="CI lo")[DOM3]
    hi = main.pivot(index="tissue", columns="loci", values="CI hi")[DOM3]
    order = ["eye"] + sorted([t for t in M.index if t != "eye"], key=lambda t: -M.loc[t, "eye"])
    M, lo, hi = M.loc[order], lo.loc[order], hi.loc[order]
    lim = float(np.ceil(M.abs().max().max()))
    ax = fig.add_axes([0.135, 0.14, 0.25, 0.70])
    im = ax.imshow(M.values, cmap=fs.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            sig = lo.values[i, j] > 0 or hi.values[i, j] < 0
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(v) > 0.65 * lim else fs.INK, fontweight="bold" if sig else "normal")
    ax.set_xticks(range(3), ["Eye\nloci", "Kidney\nloci", "Metabolic\nloci"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_yticks(range(len(order)), [fs.tissue(t) for t in order])
    eye_ticklabel(ax, order)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cb_ax = fig.add_axes([0.135, 0.075, 0.25, 0.018])
    cb = fig.colorbar(im, cax=cb_ax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=5.5)
    cb.set_label("Specificity (percentage points of strong effects); bold: 95% interval excludes 0", fontsize=5.5)
    header(fig, 0.135, 0.955, "a", "Each tissue vs itself across trait sets")

    # b: eye rank, absolute vs specificity, six calibration x summary combinations
    ab = pd.read_csv(os.path.join(AN, "step49_eye_rank_recommended.csv"), keep_default_na=False)
    spe = sp[sp.tissue == "eye"]
    bx = fig.add_axes([0.575, 0.62, 0.395, 0.26])
    ylabels, y = [], 0
    offs = np.linspace(-0.27, 0.27, 6)
    for d in DOM3[::-1]:
        for kind in ["Specificity", "Absolute score"]:
            k = 0
            for cal, _ in CALS:
                for met in ["mean", "strong fraction"]:
                    if kind == "Absolute score":
                        r = ab[(ab.calibration == cal) & (ab.metric == met) & (ab.loci == d)]["eye rank"].item()
                        bx.scatter(r, y + offs[k], s=9, facecolor="white", ec=fs.INK2, lw=0.7, zorder=3)
                    else:
                        r = spe[(spe.calibration == cal) & (spe.metric == met) & (spe.loci == d)]["rank"].item()
                        bx.scatter(r, y + offs[k], s=9, color=fs.EYE, lw=0, zorder=3)
                    k += 1
            ylabels.append(f"{DOM_LABEL[d]}, {kind.lower()}")
            y += 1
        y += 0.5
    ypos = [i + 0.5 * (i // 2) for i in range(len(ylabels))]
    bx.set_yticks(ypos, ylabels)
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_xlim(0.5, 12.5)
    bx.set_xticks([1, 3, 6, 9, 12])
    bx.set_xlabel("Eye rank among 12 tissues (six calibrations × summaries per row)")
    for yy in ypos:
        bx.axhline(yy, color=fs.GRID, lw=0.5, zorder=0)
    header(fig, 0.575, 0.955, "b", "Only specificity ranks the eye consistently", dx=-0.185)

    # c: eye specificity with 95% intervals
    cx = fig.add_axes([0.575, 0.14, 0.395, 0.30])
    e = sp[sp.tissue == "eye"]
    xo = np.linspace(-0.3, 0.3, 6)
    for g_, d in enumerate(DOM3):
        k = 0
        for cal, _ in CALS:
            for met, mk in [("mean", "o"), ("strong fraction", "s")]:
                r = e[(e.calibration == cal) & (e.metric == met) & (e.loci == d)].iloc[0]
                x = g_ + xo[k]
                cx.plot([x, x], [100 * r["CI lo"], 100 * r["CI hi"]], color=fs.MUTED, lw=0.8)
                cx.scatter(x, 100 * r.specificity, marker=mk, s=12, color=fs.EYE, lw=0.4, ec="white", zorder=3)
                k += 1
    cx.axhline(0, color=fs.INK2, lw=0.6)
    cx.set_xticks(range(3), [DOM_LABEL[d] for d in DOM3])
    cx.tick_params(axis="x", length=0)
    cx.set_ylabel("Eye specificity\n(percentage points)")
    handles = [Line2D([], [], ls="", marker="o", ms=3.5, color=fs.EYE, label="Mean"),
               Line2D([], [], ls="", marker="s", ms=3.5, color=fs.EYE, label="Strong-effect fraction")]
    cx.legend(handles=handles, loc="upper right", fontsize=5.5, handletextpad=0.3, borderaxespad=0.2)
    cx.set_xlabel("Within each group, left to right: AlphaGenome |q|, random null, matched null",
                  fontsize=5.5, color=fs.MUTED)
    header(fig, 0.575, 0.475, "c", "The eye is specific to eye loci only", dx=-0.185)
    return fs.save(fig, "Fig4_specificity")


# ───────────────────────────────────────────── Fig. 5 ─────────────────────────────────────────────
METHODS5 = [("raw, original pipeline", fs.RAW, "o", "Raw pooled summary"),
            ("calibrated DNase |q|", fs.CAL, "s", "Calibrated DNase, mean |q|"),
            ("calibrated DNase top-5%", fs.STRONG, "D", "Calibrated DNase, strong-effect fraction")]
IMM = [("a. raw pooled (original scorers, original pipeline)", "All\nassays"),
       ("raw DNase only (original scorer, uncalibrated)", "DNase"),
       ("c. DNase |q| (AlphaGenome quantile)", "DNase"),
       ("c'. ATAC null pct", "ATAC"),
       ("c'. RNA null pct", "RNA-\nseq")]
TRAIT_LABEL = {"UACR": "UACR", "eGFR": "eGFR", "CKD": "CKD", "LDL": "LDL", "HDL": "HDL",
               "triglycerides": "Triglycerides", "hypertension": "Hypertension", "BMI": "BMI", "T2D": "T2D",
               "glaucoma": "Glaucoma", "IOP": "IOP", "myopia": "Myopia"}


def rank_dots(ax, piv):
    y = np.arange(len(piv))
    cols = [m for m in METHODS5 if m[0] in piv.columns]
    lo, hi = piv[[c[0] for c in cols]].min(axis=1), piv[[c[0] for c in cols]].max(axis=1)
    ax.hlines(y, lo, hi, color=fs.GRID, lw=2.2, zorder=1)
    for (col, color, marker, _), off in zip(cols, np.linspace(-0.2, 0.2, len(cols))):
        ax.scatter(piv[col], y + off, s=13, color=color, marker=marker, lw=0.4, ec="white", zorder=3)
    ax.set_yticks(y, [f"{TRAIT_LABEL.get(t, t)} → {fs.tissue(e).lower() if e != 'pancreas_islet' else 'islet'}"
                      for t, e in piv.index])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks([1, 3, 6, 9, 12])
    ax.set_ylim(-0.6, len(piv) - 0.4)
    for c, *_ in cols:
        pass


def fig5():
    fig = plt.figure(figsize=(fs.DOUBLE, 100 * fs.MM))
    r = pd.read_csv(os.path.join(AN, "B1_expected_tissue_ranks.csv"), keep_default_na=False)
    for st, pos, letter, title in [("primary", [0.20, 0.40, 0.36, 0.47], "a",
                                    "Traits with an independently established tissue"),
                                   ("secondary", [0.20, 0.10, 0.36, 0.155], "b", "Eye traits")]:
        piv = r[r.set == st].pivot_table(index=["trait", "expected"], columns="method", values="rank")
        piv = piv.sort_values("calibrated DNase top-5%", ascending=False)
        ax = fig.add_axes(pos)
        rank_dots(ax, piv)
        means = ", ".join(f"{lab.split(',')[0] if 'Raw' in lab else lab.split(', ')[1]} {piv[c].mean():.1f}"
                          for c, _, _, lab in METHODS5)
        ax.text(1.0, 1.02, f"Mean rank: {means}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.5, color=fs.MUTED)
        if st == "secondary":
            ax.set_xlabel("Rank of the expected tissue among 12 tissues (1 = first)")
        header(fig, 0.20, 0.955 if st == "primary" else pos[1] + pos[3] + 0.05, letter, title, dx=-0.13)
    handles = [Line2D([], [], ls="", marker=m, ms=3.8, color=c, mec="white", mew=0.4, label=lab)
               for _, c, m, lab in METHODS5]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.945), ncol=3, fontsize=5.5,
               handletextpad=0.3, columnspacing=1.0)

    # c: immune diseases, rank of immune cells per disease and overall
    p = pd.read_csv(os.path.join(IA, "per_trait_ranks.csv"))
    rs = pd.read_csv(os.path.join(IA, "rank_summary.csv")).set_index("method")
    traits = ["CD", "UC", "RA", "MS", "SLE", "T1D"]
    M = p.pivot_table(index="trait", columns="method", values="immune rank").loc[traits, [m for m, _ in IMM]]
    allrow = pd.DataFrame([[rs.loc[m, "immune rank"] if m in rs.index else np.nan for m, _ in IMM]],
                          index=["All loci"], columns=M.columns)
    M = pd.concat([M, allrow])
    cx = fig.add_axes([0.735, 0.30, 0.235, 0.52])
    cx.imshow(M.values, cmap=fs.SEQUENTIAL, vmin=1, vmax=12, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isnan(v):
                continue
            cx.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=6,
                    color="white" if v <= 4 else fs.INK2, fontweight="bold" if i == len(M) - 1 else "normal")
    cx.axhline(len(M) - 1.5, color="white", lw=2.5)
    cx.set_xticks(range(len(IMM)), [lab for _, lab in IMM], fontsize=5.5)
    cx.xaxis.tick_top()
    cx.set_yticks(range(len(M)), list(M.index))
    cx.tick_params(length=0)
    for s_ in cx.spines.values():
        s_.set_visible(False)
    prob = [rs.loc[m, "P_boot(immune #1)"] if m in rs.index else np.nan for m, _ in IMM]
    for j, pv in enumerate(prob):
        if not np.isnan(pv):
            cx.text(j, len(M) - 0.35, f"{pv:.0%}", ha="center", va="top", fontsize=5.5, color=fs.INK2)
    cx.text(-0.62, len(M) - 0.35, "P(first)", ha="right", va="top", fontsize=5.5, color=fs.INK2)
    cx.set_ylim(len(M) - 0.5 + 0.9, -0.5)
    for x0, x1, lab in [(-0.4, 1.4, "Raw"), (1.6, 4.4, "Calibrated")]:
        cx.annotate("", xy=(x0, -1.55), xytext=(x1, -1.55), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.6))
        cx.text((x0 + x1) / 2, -1.7, lab, ha="center", va="bottom", fontsize=5.5, color=fs.INK2)
    header(fig, 0.735, 0.955, "c", "Immune diseases: rank of immune cells", dx=-0.07)
    fig.text(0.735, 0.20, "Rank of immune cells among 12 tissues for 1,154 loci\n"
             "of six immune-mediated diseases. P(first): share of\n"
             "bootstrap resamples of loci in which immune cells rank first.",
             fontsize=5.5, color=fs.MUTED, ha="left", va="top")
    return fs.save(fig, "Fig5_benchmarks")


FIGS = {"2": fig2, "3": fig3, "4": fig4, "5": fig5}

if __name__ == "__main__":
    import fig_v2_fig1
    wanted = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for k in wanted:
        fig_v2_fig1.main() if k == "1" else FIGS[k]()
