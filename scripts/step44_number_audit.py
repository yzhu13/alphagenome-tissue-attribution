"""
step44_number_audit.py

Checks every key number quoted in the paper against the analysis tables. Each check is a phrase from
the text and a function that rebuilds the same phrase from the tables; a check passes when
  1. the exact phrase appears in the manuscript text, and
  2. recomputing it from the tables gives that same phrase.
A number that is edited in the text, or a table that changes after a rerun, makes the audit fail.

The text is read from drafts/ (intro_results, abstract, discussion, figure legends), or from the file
named by AUDIT_TEXT. Without any text (as in the public repository before publication) the script
only recomputes the phrases from the tables and prints them.

    python scripts/step44_number_audit.py                                  # exit code 1 if anything fails
    AUDIT_TEXT=manuscript/manuscript.md python scripts/step44_number_audit.py
Writes reports/number_audit.md.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step35_analysis as s35  # noqa: E402

NEW = s35.NEW
AN = s35.AN
DRAFT = os.path.join(NEW, "drafts", "intro_results_draft.md")
DRAFTS = [DRAFT] + [os.path.join(NEW, "drafts", f) for f in ("abstract_draft.md", "discussion_draft.md",
                                                             "figure_legends_draft.md", "methods_draft.md",
                                                             "limitations_draft.md")]
if os.environ.get("AUDIT_TEXT"):  # e.g. AUDIT_TEXT=manuscript/manuscript.md: check the assembled manuscript instead
    DRAFTS = [os.path.join(NEW, os.environ["AUDIT_TEXT"])]
REPORT = os.path.join(NEW, "reports", "number_audit.md")


def csv(*p):
    # keep_default_na=False: otherwise pandas reads the domain label "null" as NaN
    return pd.read_csv(os.path.join(NEW, *p), keep_default_na=False, na_values=[""])


def pct(x):
    return f"{100 * x:.0f}%"


OCULAR3 = ["iris pigment epithelial cell", "retinal pigment epithelial cell", "non-pigmented ciliary epithelial cell"]


def nth(n):
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def a5c():
    return csv("data", "analysis", "A5c_eye_track_percentiles.csv")


def mb():
    return csv("data", "analysis", "step50_metabolic_batch.csv")


def mb_row(analysis, version):
    t = mb()
    return t[(t.analysis == analysis) & (t["metabolic loci"] == version)].iloc[0]


def mb_pair(analysis, col):
    return [mb_row(analysis, v)[col] for v in ("published", "with added loci")]


def mb_added():
    t = csv("data", "metabolic_batch", "loci_resolved.csv")
    return t[t.excluded.fillna("") == ""]


def a1(domain):
    return pd.read_csv(os.path.join(AN, f"A1A2_orig_ranking_{domain}.csv"), index_col=0)["abs_raw"]


def a3():
    return csv("data", "analysis", "A3_permutation.csv").set_index("domain")


def a5b():
    t = csv("data", "analysis", "A5b_gwas_vs_null_by_assay.csv")
    return t[t.loci == "all GWAS"].set_index("scorer")["mean null-percentile"]


def a6():
    g = csv("data", "analysis", "A6_gradient.csv")
    return g[~g.tissue.isin(["other", "eye_misfiled"])]


def bench():
    r = csv("data", "analysis", "B1_expected_tissue_ranks.csv")
    return r[r.set == "primary"]


def unc():
    return csv("data", "uncertainty", "fig4a_rank_uncertainty.csv").set_index(["domain", "method"])


def paired():
    return csv("data", "uncertainty", "benchmark_paired_vs_raw.csv").set_index("method")


def immune():
    return csv("data", "immune_analysis", "rank_summary.csv").set_index("method")


def gap():
    t = csv("data", "null_matched_analysis", "b_rna_dnase_gap.csv")
    t = t[t.scorers == "rec"].set_index("background")["RNA - DNase"]
    return f"from {t['random null']:.3f} to {t['matched null']:.3f}"


def a7_raw_or():
    txt = open(os.path.join(NEW, "reports", "analysis_A1_A7.md"), encoding="utf-8").read()
    return [float(x) for x in re.findall(r"hTMC Q4-vs-Q1 OR by tissue score: eye ([0-9.]+)", txt)]


def null_margin():
    """Eye score / next tissue for random variants under the original pipeline."""
    return a1("null")["eye"] / a1("null").drop("eye").max()


def a7_raw_top():
    """The two tissues whose raw scores predicted hTMC open chromatin best (skin, muscle/adipose)."""
    txt = open(os.path.join(NEW, "reports", "analysis_A1_A7.md"), encoding="utf-8").read()
    top = re.findall(r"hTMC Q4-vs-Q1 OR by tissue score: eye [0-9.]+ \(rank \d+ of \d+\); top: (.*)", txt)[0]
    return re.findall(r"[a-z_]+ ([0-9.]+)", top)[:2]


def fig2_comp():
    from scipy.stats import spearmanr as _sp
    t = csv("data", "analysis", "fig2_assay_composition.csv").set_index("tissue")
    return t, _sp(t.chromatin_share, t.tissue_score_gwas).correlation


CHECKS = [  # (what, exact phrase in draft, function returning the phrase recomputed from data)
    ("Fig. 2c: chromatin share vs tissue score", "(Spearman ρ = 0.83, all GWAS",
     lambda: f"(Spearman ρ = {fig2_comp()[1]:.2f}, all GWAS"),
    ("Fig. 2b: chromatin share of non-eye tissues", "(23–45% chromatin tracks; Fig. 2b)",
     lambda: (lambda t: f"({100 * t.min():.0f}–{100 * t.max():.0f}% chromatin tracks; Fig. 2b)")(
         fig2_comp()[0].drop("eye").chromatin_share)),
    ("A1 eye loci: eye vs skin", "0.00609 vs 0.00327 for skin",
     lambda: f"{a1('eye')['eye']:.5f} vs {a1('eye')['skin']:.5f} for skin"),
    ("A1 kidney loci: eye vs liver", "0.00618 vs 0.00596 for liver",
     lambda: f"{a1('kidney')['eye']:.5f} vs {a1('kidney')['liver']:.5f} for liver"),
    ("A1 metabolic loci: eye vs skin", "0.00592 vs 0.00287 for skin",
     lambda: f"{a1('metabolic')['eye']:.5f} vs {a1('metabolic')['skin']:.5f} for skin"),
    ("A2 null: eye / next", "at 2.5 times the next tissue",
     lambda: f"at {a1('null')['eye'] / a1('null').drop('eye').max():.1f} times the next tissue"),
    ("A2 null: values", "0.0015 vs 0.00059 for skin",
     lambda: f"{a1('null')['eye']:.4f} vs {a1('null')['skin']:.5f} for skin"),
    ("A4 assay vs tissue share", "assay type\nexplained 63% of the variance in mean raw score and tissue only 1.7%",
     lambda: (lambda r: f"assay type\nexplained {100 * r['assay share']:.0f}% of the variance in mean raw score and "
                        f"tissue only {100 * r['tissue share']:.1f}%")(
         csv("data", "analysis", "A4_variance.csv").set_index("score").loc["original raw |score|"])),
    ("A3 permutation rho", "ρ = 0.58, 0.73 and 0.81 for\neye, kidney and metabolic loci; 0.80 for random variants",
     lambda: (lambda t: f"ρ = {t.loc['eye'].iloc[0]:.2f}, {t.loc['kidney'].iloc[0]:.2f} and {t.loc['metabolic'].iloc[0]:.2f} for\n"
                        f"eye, kidney and metabolic loci; {t.loc['null'].iloc[0]:.2f} for random variants")(a3())),
    ("A3 random 5-DNase beats all", "in 95% of draws for random variants, and in 38%, 16% and 70% of draws",
     lambda: (lambda c: f"in {pct(c.loc['null'])} of draws for random variants, and in {pct(c.loc['eye'])}, "
                        f"{pct(c.loc['kidney'])} and {pct(c.loc['metabolic'])} of draws")(a3().iloc[:, 1])),
    ("A5b excess by assay", "(mean null-percentile 0.633) than in DNase (0.529) or ATAC (0.536)",
     lambda: (lambda s: f"(mean null-percentile {s['rna_rec']:.3f}) than in DNase ({s['dnase_rec']:.3f}) or ATAC "
                        f"({s['atac_rec']:.3f})")(a5b())),
    ("Matched null gap", "fell from 0.104 to 0.025", lambda: "fell " + gap()),
    ("A6 eye raw median", "the eye had the smallest rise of all (1.46-fold)",
     lambda: (lambda g: f"the eye had the smallest rise of all ({g.loc[g.tissue == 'eye', 'median ratio 3/1'].item():.2f}-fold)")(
         a6()[a6().score == "raw (original pipeline)"])),
    ("A6 calibrated range", "the ratio was 0.99–1.08 for every tissue",
     lambda: (lambda g: f"the ratio was {g['median ratio 3/1'].min():.2f}–{g['median ratio 3/1'].max():.2f} for every tissue")(
         a6()[a6().score == "calibrated DNase |q|"])),
    ("A7 raw hTMC OR, all tissues", "odds ratio 8.9–19.4 across the 12 tissues: skin 19.4, muscle/adipose 16.4, eye 15.9",
     lambda: (lambda o: f"odds ratio {o.min():.1f}–{o.max():.1f} across the {len(o)} tissues: skin {o['skin']:.1f}, "
                        f"muscle/adipose {o['muscle_adipose']:.1f}, eye {o['eye']:.1f}")(
         csv("data", "analysis", "A7_hTMC_or_raw.csv").set_index("tissue").iloc[:, 0].drop("other"))),
    ("A7 raw hTMC OR, discussion", "(top-quartile OR 19.4 and 16.4 for skin and muscle/adipose vs 15.9 for the eye)",
     lambda: (lambda top, o: f"(top-quartile OR {top[0]} and {top[1]} for skin and muscle/adipose vs {o[0]:.1f} for the eye)")(
         a7_raw_top(), a7_raw_or())),
    ("A7 calibrated hTMC OR", "OR for trabecular meshwork open chromatin fell to 2.5",
     lambda: f"OR for trabecular meshwork open chromatin fell to {a7_raw_or()[1]:.1f}"),
    ("B1 mean ranks", "mean rank 3.4, first for\n4 of 9 traits and in the top three for 6 of 9, against 4.1, 1 of 9 and 4 of 9",
     lambda: (lambda r: (lambda s5, raw: f"mean rank {s5['rank'].mean():.1f}, first for\n{(s5['rank'] == 1).sum()} of 9 traits and "
                                         f"in the top three for {(s5['rank'] <= 3).sum()} of 9, against {raw['rank'].mean():.1f}, "
                                         f"{(raw['rank'] == 1).sum()} of 9 and {(raw['rank'] <= 3).sum()} of 9")(
         r[r.method == "calibrated DNase top-5%"], r[r.method == "raw, original pipeline"]))(bench())),
    ("B1 mean |q| rank", "(mean rank 5.4)",
     lambda: f"(mean rank {bench()[bench().method == 'calibrated DNase |q|']['rank'].mean():.1f})"),
    ("B1 paired test", "mean rank difference −0.67, 95% bootstrap CI −2.1 to 1.3; Wilcoxon P = 0.42",
     lambda: (lambda r: f"mean rank difference {r['mean difference']:.2f}, 95% bootstrap CI "
                        f"{float(r['95% CI'].split(' to ')[0]):.1f} to {float(r['95% CI'].split(' to ')[1]):.1f}; "
                        f"Wilcoxon P = {r['Wilcoxon p']:.2f}".replace("-", "−"))(paired().loc["strong fraction |q|>=0.95"])),
    ("Uncertainty: raw eye #1", "first in 99.8% and 99.6% of\n500 resamples",
     lambda: (lambda u: f"first in {100 * u.loc[('eye', 'raw, pooled (original pipeline)'), 'P(eye #1)']:.1f}% and "
                        f"{100 * u.loc[('metabolic', 'raw, pooled (original pipeline)'), 'P(eye #1)']:.1f}% of\n500 resamples")(unc())),
    ("Uncertainty: kidney raw", "(first in 53%)",
     lambda: f"(first in {pct(unc().loc[('kidney', 'raw, pooled (original pipeline)'), 'P(eye #1)'])})"),
    ("Uncertainty: strong fraction eye / kidney", "(first in 76% of\nresamples; interval 1–3) and kidney first for kidney loci (68%)",
     lambda: (lambda u: f"(first in {pct(u.loc[('eye', 'calibrated DNase, strong fraction (|q|>=0.95)'), 'P(eye #1)'])} of\n"
                        f"resamples; interval {u.loc[('eye', 'calibrated DNase, strong fraction (|q|>=0.95)'), 'eye 95% interval']}) "
                        f"and kidney first for kidney loci ({pct(u.loc[('kidney', 'calibrated DNase, strong fraction (|q|>=0.95)'), 'P(top #1)'])})")(unc())),
    ("Uncertainty: metabolic strong fraction", "the eye ranked first in 36% of resamples and its 95% rank interval\nwas 1–8",
     lambda: (lambda r: f"the eye ranked first in {pct(r['P(eye #1)'])} of resamples and its 95% rank interval\nwas "
                        f"{r['eye 95% interval']}")(unc().loc[("metabolic", "calibrated DNase, strong fraction (|q|>=0.95)")])),
    ("Uncertainty: DNase mean |q| eye intervals", "the eye's 95% rank interval was 1–6, 1–11 and 1–11",
     lambda: "the eye's 95% rank interval was " + ", ".join(
         unc().loc[(d, "calibrated DNase, mean |q|"), "eye 95% interval"] for d in ["eye", "kidney"]) +
         " and " + unc().loc[("metabolic", "calibrated DNase, mean |q|"), "eye 95% interval"]),
    ("Recommended raw pooled: eye / next", "(1.93, 1.82, 1.88 and 1.94 times the next tissue)",
     lambda: (lambda t: f"({t.loc['eye']:.2f}, {t.loc['kidney']:.2f}, {t.loc['metabolic']:.2f} and {t.loc['null']:.2f} "
                        f"times the next tissue)")(csv("data", "analysis", "step45_raw_recommended_pooled.csv")
                                                  .set_index("loci")["top / next"])),
    ("Max per tissue vs track count", "(Spearman ρ = 0.84–0.86 pooling all assays, 0.92–0.93 within DNase)",
     lambda: (lambda t: (lambda a, b: f"(Spearman ρ = {a.min():.2f}–{a.max():.2f} pooling all assays, "
                                      f"{b.min():.2f}–{b.max():.2f} within DNase)")(
         t[t.tracks == "all recommended scorers pooled"]["Spearman(score, n tracks)"],
         t[t.tracks == "DNase only"]["Spearman(score, n tracks)"]))(
         csv("data", "analysis", "step45_max_per_tissue_summary.csv"))),
    ("Max per tissue: top-tissue tracks and eye rank",
     "the top tissue always had many tracks (23–92), and the eye, with four tracks, ranked 11th or 12th of 12",
     lambda: (lambda t: f"the top tissue always had many tracks ({t['top tissue tracks'].min()}–{t['top tissue tracks'].max()}), "
                        f"and the eye, with four tracks, ranked {t['eye rank'].min()}th or {t['eye rank'].max()}th of 12")(
         csv("data", "analysis", "step45_max_per_tissue_summary.csv"))),
    ("Immune raw eye #1", "(first in 95% of 1,000 bootstrap resamples of loci)",
     lambda: f"(first in {pct(immune().loc['a. raw pooled (original scorers, original pipeline)', 'P_boot(eye #1)'])} of 1,000 bootstrap resamples of loci)"),
    ("Immune DNase", "(DNase: first in 83% of bootstrap resamples",
     lambda: f"(DNase: first in {pct(immune().loc['c. DNase |q| (AlphaGenome quantile)', 'P_boot(immune #1)'])} of bootstrap resamples"),
    ("Immune raw DNase only", "compared (97% of resamples)",
     lambda: f"compared ({pct(immune().loc['raw DNase only (original scorer, uncalibrated)', 'P_boot(immune #1)'])} of resamples)"),
    ("Organ systems: DNase eye ranks", "calibrated DNase scores ranked it first, fourth and third for eye, kidney and metabolic loci",
     lambda: (lambda t: "calibrated DNase scores ranked it " + ", ".join(
         {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}[int(t.loc[(t.method == "calibrated DNase, mean |q|") & (t.loci == d), "eye rank"].item())]
         for d in ["eye", "kidney"]) + " and " + {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}[
         int(t.loc[(t.method == "calibrated DNase, mean |q|") & (t.loci == "metabolic"), "eye rank"].item())] +
         " for eye, kidney and metabolic loci")(csv("data", "analysis", "step46_organ_systems.csv"))),
    ("Biosample level: raw vs chromatin share", "(Spearman ρ = 0.56–0.86 for disease loci and 0.92 for random variants)",
     lambda: (lambda t: (lambda g, n: f"(Spearman ρ = {g.min():.2f}–{g.max():.2f} for disease loci and {n:.2f} for random variants)")(
         t[t.loci != "null"]["Spearman(score, chromatin share)"],
         t[t.loci == "null"]["Spearman(score, chromatin share)"].item()))(csv("data", "analysis", "step46_biosample_raw.csv"))),
    ("Biosample level: calibrated DNase RPE", "random variants (52%) and to about a third of the way down for kidney and metabolic loci (32–34%)",
     lambda: (lambda t: f"random variants ({t.loc['null']:.0f}%) and to about a third of the way down for kidney and metabolic loci "
                        f"({min(t.loc['kidney'], t.loc['metabolic']):.0f}–{max(t.loc['kidney'], t.loc['metabolic']):.0f}%)")(
         csv("data", "analysis", "step46_biosample_dnase_calibrated.csv").set_index("loci")["retinal pigment epithelium (top %)"])),
    ("Enformer scale", "were 13 times larger than on DNase tracks and 24 times larger than on CAGE tracks",
     lambda: (lambda t: f"were {t['CHIP/DNASE'].item():.0f} times larger than on DNase tracks and "
                        f"{t['CHIP/CAGE'].item():.0f} times larger than on CAGE tracks")(
         csv("data", "enformer", "step48_scale.csv"))),
    ("Enformer variance explained", "83% of the variance in per-track mean effect against 9% for tissue",
     lambda: (lambda t: f"{t.var_explained_assay.item():.0%} of the variance in per-track mean effect against "
                        f"{t.var_explained_tissue.item():.0%} for tissue")(csv("data", "enformer", "step48_scale.csv"))),
    ("Enformer raw pooled vs ChIP share",
     "ρ = 0.87, 0.90 and 0.95 for eye, kidney and metabolic loci, and 0.97 for random variants",
     lambda: (lambda t: "ρ = " + ", ".join(f"{t.loc[d]:.2f}" for d in ["eye", "kidney"]) +
                        f" and {t.loc['metabolic']:.2f} for eye, kidney and metabolic loci, and "
                        f"{t.loc['null']:.2f} for random variants")(
         csv("data", "enformer", "step48_raw_pooled_summary.csv").set_index("loci")["ρ with CHIP share"])),
    ("Enformer calibrated assay means", "DNase 0.617, CAGE 0.608, ChIP 0.529",
     lambda: (lambda t: f"DNase {t['mean pct DNASE']:.3f}, CAGE {t['mean pct CAGE']:.3f}, "
                        f"ChIP {t['mean pct CHIP']:.3f}")(
         csv("data", "enformer", "step48_calibrated_pooled.csv").set_index("loci").loc["eye"])),
    ("Enformer max per tissue vs track count",
     "(Spearman ρ = 0.84–0.92 pooling assays, 0.73–0.82 within DNase)",
     lambda: (lambda t: (lambda a, b: f"(Spearman ρ = {a.min():.2f}–{a.max():.2f} pooling assays, "
                                      f"{b.min():.2f}–{b.max():.2f} within DNase)")(
         t[t.tracks == "all"]["Spearman(max, n tracks)"], t[t.tracks == "DNASE"]["Spearman(max, n tracks)"]))(
         csv("data", "enformer", "step48_max_per_tissue.csv"))),
    ("Pooled calibrated eye rank, both nulls",
     "matched null the eye ranked first, twelfth and third for eye, kidney and metabolic loci, against the random null twelfth in all three",
     lambda: (lambda c: (lambda m, r: "matched null the eye ranked " + ", ".join(ORD[int(m[d])] for d in ["eye", "kidney"]) +
                         f" and {ORD[int(m['metabolic'])]} for eye, kidney and metabolic loci, against the random null " +
                         (f"{ORD[int(r['eye'])]} in all three" if len(set(r)) == 1 else "MIXED"))(
         c[c.method == "calibrated pooled, matched null (rec scorers)"].set_index("domain")["rank"],
         c[c.method == "calibrated pooled, random null (rec scorers)"].set_index("domain")["rank"]))(
         (lambda c: c[c.tissue == "eye"])(csv("data", "null_matched_analysis", "c_rankings.csv")))),
    ("Matched null DNase mean: eye P(#1)", "(first in 65% and 67% of resamples of loci and tracks)",
     lambda: (lambda t: f"(first in {t.loc['eye']:.0%} and {t.loc['metabolic']:.0%} of resamples of loci and tracks)")(
         (lambda t: t[(t.calibration == "null pct, matched null") & (t.metric == "mean")].set_index("loci")["eye P(#1)"])(
             csv("data", "analysis", "step49_eye_rank_recommended.csv")))),
    ("Fake-eye percentile (matched null, metabolic)", "the eye scored higher than 98% of 5,000 random sets",
     lambda: (lambda t: f"the eye scored higher than {t.item():.0%} of 5,000 random sets")(
         (lambda t: t[(t.calibration == "null pct, matched null") & (t.metric == "mean") & (t.loci == "metabolic")][
             "eye percentile among fakes"])(csv("data", "analysis", "step49_fake_eye.csv")))),
    ("Specificity: eye P(#1) for eye loci", "(first in 57–76% of resamples;",
     lambda: (lambda t: f"(first in {100 * t.min():.0f}–{t.max():.0%} of resamples;")(
         (lambda t: t[(t.loci == "eye") & (t.tissue == "eye")]["P(#1)"])(csv("data", "analysis", "step49_specificity.csv")))),
    ("Specificity: eye rank for kidney/metabolic", "ranked 10th–12th for kidney and metabolic loci, where its specificity was negative",
     lambda: (lambda t: f"ranked {ORD2[t['rank'].min()]}–{ORD2[t['rank'].max()]} for kidney and metabolic loci, where its "
                        f"specificity was {'negative' if (t.specificity < 0).all() else 'MIXED'}")(
         (lambda t: t[(t.loci != "eye") & (t.tissue == "eye")].drop_duplicates(["calibration", "metric", "loci"]))(
             csv("data", "analysis", "step49_specificity_all.csv")))),
    ("Specificity: metabolic top = islet in all six", "the most specific tissue was the pancreatic islet in all six combinations",
     lambda: (lambda t: "the most specific tissue was the pancreatic islet in all six combinations"
              if (t.tissue == "pancreas_islet").all() and len(t) == 6 else "NOT ALL ISLET")(
         (lambda t: t[(t.loci == "metabolic") & (t["rank"] == 1)])(csv("data", "analysis", "step49_specificity_all.csv")))),
    ("Specificity: kidney rank for kidney loci", "kidney ranked third or fourth, behind liver or lung",
     lambda: (lambda t, top: f"kidney ranked {ORD[t['rank'].min()]} or {ORD[t['rank'].max()]}, behind "
                             + " or ".join(sorted(set(top.tissue))))(
         (lambda a: a[(a.loci == "kidney") & (a.tissue == "kidney")])(csv("data", "analysis", "step49_specificity_all.csv")),
         (lambda a: a[(a.loci == "kidney") & (a["rank"] == 1)])(csv("data", "analysis", "step49_specificity_all.csv")))),
    ("Discussion: null margin exceeds every domain", "for which the eye ranked first by a wider margin than for any disease",
     lambda: "for which the eye ranked first by a wider margin than for any disease"
     if null_margin() > max(a1(d)["eye"] / a1(d).drop("eye").max() for d in ["eye", "kidney", "metabolic"]) else "NOT WIDER"),
    ("Discussion: eye specificity, |q| strong fraction", "about two percentage points in the fraction of strong effects",
     lambda: (lambda v: "about two percentage points in the fraction of strong effects" if 1.5 <= v < 2.5 else f"{v:.1f} POINTS")(
         100 * (lambda t: t[(t.calibration.str.startswith("AlphaGenome")) & (t.metric == "strong fraction") &
                            (t.loci == "eye") & (t.tissue == "eye")].specificity.item())(
             csv("data", "analysis", "step49_specificity_all.csv")))),
    ("Discussion: metabolic top P(#1) with |q|", "no tissue was first in more than 34% of resamples",
     lambda: (lambda t: f"no tissue was first in more than {t.max():.0%} of resamples")(
         (lambda t: t[(t.calibration.str.startswith("AlphaGenome")) & (t.loci == "metabolic")]["top P(#1)"])(
             csv("data", "analysis", "step49_eye_rank_recommended.csv")))),
    ("Abstract: variance by assay and tissue", "Assay type explained 63% of the variance in per-track scores, tissue 1.7%",
     lambda: (lambda r: f"Assay type explained {100 * r['assay share']:.0f}% of the variance in per-track scores, "
                        f"tissue {100 * r['tissue share']:.1f}%")(
         csv("data", "analysis", "A4_variance.csv").set_index("score").loc["original raw |score|"])),
    ("A6 raw range, other tissues", "every other tissue rose 1.77- to 2.30-fold",
     lambda: (lambda g: f"every other tissue rose {g.min():.2f}- to {g.max():.2f}-fold")(
         (lambda g: g[(g.score == "raw (original pipeline)") & (g.tissue != "eye")]["median ratio 3/1"])(a6()))),
    ("Legend Fig. 2b: random sets", "95% of random sets exceed every real non-eye tissue",
     lambda: f"{pct(a3().loc['null'].iloc[1])} of random sets exceed every real non-eye tissue"),
    ("Legend Fig. 1: domain sizes", "(n = 1,811) and metabolic (n = 5,150)",
     lambda: (lambda c: f"(n = {c['kidney']:,}) and metabolic (n = {c['metabolic']:,})")(
         s35.domain_map(s35.read_variants()).domain.value_counts())),
    ("A5c: choroid plexus rank among DNase tracks",
     "this track ranked 2nd to 9th of all 305 DNase tracks in every disease domain",
     lambda: (lambda t: f"this track ranked {nth(t['rank'].min())} to {nth(t['rank'].max())} of all "
                        f"{t.n_tracks.iloc[0]} DNase tracks in every disease domain")(
         a5c()[lambda z: (z.biosample == "choroid plexus epithelial cell") & (z.domain != "null")])),
    ("A5c: ocular track percentiles",
     "sat between the 63rd and 91st percentiles of DNase tracks in every domain, against the 49th to 64th "
     "for random variants; the fourth, retina, sat between the 16th and 41st",
     lambda: (lambda t: (lambda three, gw, nul, ret:
                         f"sat between the {nth(gw.min())} and {nth(gw.max())} percentiles of DNase tracks in every "
                         f"domain, against the {nth(nul.min())} to {nth(nul.max())} for random variants; the fourth, "
                         f"retina, sat between the {nth(ret.min())} and {nth(ret.max())}")(
         None,
         t[t.biosample.isin(OCULAR3) & (t.domain != "null")].percentile.round(),
         t[t.biosample.isin(OCULAR3) & (t.domain == "null")].percentile.round(),
         t[(t.biosample == "retina") & (t.domain != "null")].percentile.round()))(a5c())),
    ("Missing metabolic loci: resolved", "Re-queried on 21 October 2026, 195 resolved and passed the\nfilters",
     lambda: f"Re-queried on 21 October 2026, {len(mb_added())} resolved and passed the\nfilters"),
    ("Missing metabolic loci: already scored / scored now",
     "64 had already been scored as eye or kidney loci and 131 were scored for this check",
     lambda: (lambda a, v: f"{a.rsid.isin(v).sum()} had already been scored as eye or kidney loci and "
                           f"{(~a.rsid.isin(v)).sum()} were scored for this check")(
         mb_added(), set(s35.read_variants().id))),
    ("Missing metabolic loci: metabolic n", "enlarged from\n5,150 to 5,345",
     lambda: (lambda t: f"enlarged from\n{t.n.min():,} to {t.n.max():,}")(
         mb()[lambda z: z.analysis == "raw pooled summary"])),
    ("Missing metabolic loci: raw pooled eye first",
     "still ranked the eye first (2.04 times the next tissue)",
     lambda: (lambda r: f"still ranked the eye {ORD[int(r.eye_rank)]} ({r.eye_over_next:.2f} times the next tissue)")(
         mb_row("raw pooled summary", "with added loci"))),
    ("Missing metabolic loci: strong-fraction eye ranks",
     "the eye fell from first to third for metabolic loci with\nAlphaGenome quantile scores and from first to second "
     "against the random null",
     lambda: (lambda a, b: f"the eye fell from {ORD[int(a[0])]} to {ORD[int(a[1])]} for metabolic loci with\n"
                           f"AlphaGenome quantile scores and from {ORD[int(b[0])]} to {ORD[int(b[1])]} against the random null")(
         mb_pair("DNase strong fraction, AlphaGenome |q|", "eye_rank"),
         mb_pair("DNase strong fraction, null pct, random null", "eye_rank"))),
    ("Missing metabolic loci: hypertension expected rank", "the expected tissue\nfor hypertension rose from fourth to third",
     lambda: (lambda a: f"the expected tissue\nfor hypertension rose from {ORD[int(a[0])]} to {ORD[int(a[1])]}")(
         mb_pair("expected tissue, hypertension -> heart_vascular", "expected_rank"))),
    ("Missing metabolic loci: islet most specific in all six",
     "the pancreatic islet remained the most specific tissue for metabolic loci\nunder all six combinations",
     lambda: (lambda t: "the pancreatic islet remained the most specific tissue for metabolic loci\nunder all "
                        f"{['no', 'one', 'two', 'three', 'four', 'five', 'six'][int((t.top == 'pancreas_islet').sum())]} combinations")(
         mb()[lambda z: z.analysis.str.startswith("specificity") & (z["metabolic loci"] == "with added loci")])),
]
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
       9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth"}
ORD2 = {i: f"{i}th" for i in range(4, 13)} | {1: "1st", 2: "2nd", 3: "3rd"}


def data_only():
    """No manuscript text available: recompute each phrase from the tables and print it."""
    for what, _, fn in CHECKS:
        try:
            got = re.sub(r"\s+", " ", fn())
        except Exception as e:  # checks that need the text itself
            got = f"(not computed: {type(e).__name__})"
        print(f"{what}: {got}")
    print(f"{len(CHECKS)} phrases recomputed from the tables (no manuscript text found, nothing compared)")


def main():
    if not any(os.path.exists(f) for f in DRAFTS):
        data_only()
        return
    text = chr(10).join(open(f, encoding="utf-8").read() for f in DRAFTS if os.path.exists(f))
    flat = re.sub(r"\s+", " ", text)
    rows = []
    for what, phrase, fn in CHECKS:
        in_text = re.sub(r"\s+", " ", phrase) in flat
        try:
            got = fn()
            ok_val = re.sub(r"\s+", " ", got) == re.sub(r"\s+", " ", phrase)
        except Exception as e:  # a missing table is a failure, not a crash
            got, ok_val = f"ERROR: {type(e).__name__}: {e}", False
        rows.append({"check": what, "in draft": in_text, "matches data": ok_val,
                     "status": "PASS" if in_text and ok_val else "FAIL",
                     "draft says": re.sub(r"\s+", " ", phrase), "data gives": re.sub(r"\s+", " ", str(got))})
    r = pd.DataFrame(rows)
    n_fail = (r.status == "FAIL").sum()
    L = ["# Number audit", "", f"Generated by `scripts/step44_number_audit.py`. {len(r)} checks, "
         f"**{len(r) - n_fail} PASS, {n_fail} FAIL**.", "", r.to_markdown(index=False), ""]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(r[["check", "status"]].to_string(index=False))
    print(f"{len(r) - n_fail} PASS, {n_fail} FAIL -> {REPORT}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
