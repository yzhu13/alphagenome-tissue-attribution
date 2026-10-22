"""
step39b_matched_null_analysis.py

Does the RNA > DNase gap (GWAS loci exceed the null more in RNA-seq than in DNase) just reflect
that GWAS loci sit closer to genes? Recalibrates every track against two backgrounds:

  random null   data/full_rescoring/            2,000 random common SNVs (step33)
  matched null  data/null_matched_scoring/      2,000 common SNVs matched to the GWAS loci on
                                                distance to the nearest protein-coding TSS (step39)

and answers:
  a. distance to TSS: GWAS vs random null vs matched null
  b. per assay and scorer: mean null-percentile of GWAS loci (and fraction >= 0.95) under each null
  c. eye rank among tissues per GWAS domain: raw pooled (original scorers, original pipeline), calibrated
     pooled (recommended scorers) and assay-matched calibrated DNase, against the matched null
     (random-null version alongside for comparison). `other` and `eye_misfiled` are not ranked
     (raw pooled folds eye_misfiled into eye, as in the original pipeline).

Reuses step35_analysis (null_grids, empirical, track_map, domain_map, ORIG, REC, paths) but never
calls its aggregate() and never writes into data/analysis/ or data/full_rescoring/. Main-run parts
are read read-only; a part that cannot be read (possibly mid-write) is skipped. Rerunnable: the
per-locus aggregate is cached in data/null_matched_analysis/ and rebuilt when the set of parts changes.

Output: reports/matched_null.md, data/null_matched_analysis/*.csv (+ locus_tissue.parquet cache)

    python scripts/step39b_matched_null_analysis.py              # (re)aggregate if parts changed, report
    python scripts/step39b_matched_null_analysis.py --reaggregate
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step35_analysis as s35  # noqa: E402
import step39_build_matched_null as s39  # noqa: E402

NEW = s35.NEW
MAIN_RUN = s35.RUN
MATCH_RUN = os.path.join(NEW, "data", "null_matched_scoring")
OUT = os.path.join(NEW, "data", "null_matched_analysis")
REPORT = os.path.join(NEW, "reports", "matched_null.md")
ORIG, REC = s35.ORIG, s35.REC
DOMAINS = ["eye", "kidney", "metabolic"]
BG = {"rand": "random null", "match": "matched null"}
UNRANKED = {"other", "eye_misfiled"}
N_BOOT = 500


def readable_parts(folder):
    ok, bad = [], []
    for p in sorted(glob.glob(os.path.join(folder, "parts", "*.parquet"))):
        try:
            pq.ParquetFile(p).metadata
            ok.append(p)
        except Exception:
            bad.append(p)
    return ok, bad


def read_sets():
    v = s35.read_variants()[["id", "set", "CHR", "pos"]]
    m = pd.read_csv(os.path.join(MATCH_RUN, "variants.csv"), keep_default_na=False)[["id", "set", "CHR", "pos"]]
    return pd.concat([v, m], ignore_index=True)


# ── aggregation ───────────────────────────────────────────────────────────────
def aggregate(force=False):
    main_parts, bad_main = readable_parts(MAIN_RUN)
    match_parts, bad_match = readable_parts(MATCH_RUN)
    parts = main_parts + match_parts
    lt_f, stamp_f = os.path.join(OUT, "locus_tissue.parquet"), os.path.join(OUT, "aggregated_parts.txt")
    stamp = [os.path.relpath(p, NEW) for p in parts]
    if bad_main or bad_match:
        print(f"skipped unreadable parts: {[os.path.basename(p) for p in bad_main + bad_match]}")
    if not force and os.path.exists(stamp_f) and os.path.exists(lt_f) and open(stamp_f).read().split() == stamp:
        return pd.read_parquet(lt_f)
    sets = read_sets()
    tissue = s35.track_map()
    grids = {"rand": s35.null_grids(main_parts, set(sets.id[sets.set == "null"])),
             "match": s35.null_grids(match_parts, set(sets.id[sets.set == "null_matched"]))}
    lt = []
    for i, p in enumerate(parts, 1):
        try:
            d = pd.read_parquet(p)
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}")
            continue
        d["tissue"] = d.track_name.map(tissue).fillna("other")
        d["abs_raw"] = d.raw_score.abs()
        agg = {"abs_raw": ("abs_raw", "mean"), "n_tracks": ("abs_raw", "size")}
        for k, (index, G) in grids.items():
            d[f"emp_{k}"] = s35.empirical(d, index, G)
            d[f"top5_{k}"] = (d[f"emp_{k}"] >= 0.95).where(d[f"emp_{k}"].notna())
            agg[f"emp_{k}"] = (f"emp_{k}", "mean")
            agg[f"top5_{k}"] = (f"top5_{k}", "mean")
        pooled = d.assign(scorer=np.where(d.scorer.isin(ORIG), "orig_all", "rec_all"))
        for x in (d, pooled):
            lt.append(x.groupby(["id", "scorer", "tissue"]).agg(**agg).reset_index())
        if i % 20 == 0:
            print(f"  aggregated {i}/{len(parts)} parts", flush=True)
    lt = pd.concat(lt, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    lt.to_parquet(lt_f, index=False)
    with open(stamp_f, "w") as f:
        f.write("\n".join(stamp))
    return lt


def per_variant(lt, cols):
    """Track-count-weighted mean of the per-tissue means = mean over all tracks of the variant."""
    x = lt[lt.scorer.isin(ORIG + REC)]
    out = {}
    for c in cols:
        ok = x[c].notna()
        num = (x[c].where(ok, 0) * x.n_tracks).groupby([x.id, x.scorer]).sum()
        den = x.n_tracks.where(ok, 0).groupby([x.id, x.scorer]).sum()
        out[c] = num / den.replace(0, np.nan)
    return pd.DataFrame(out).reset_index()


def fold_eye(lt):
    x = lt.copy()
    x["tissue"] = x.tissue.replace({"eye_misfiled": "eye"})
    cols = ["abs_raw", "emp_rand", "emp_match", "top5_rand", "top5_match"]
    for c in cols:
        x[c] = x[c] * x.n_tracks
    g = x.groupby(["id", "scorer", "tissue"])[cols + ["n_tracks"]].sum()
    for c in cols:
        g[c] = g[c] / g.n_tracks
    return g.reset_index()


def rank_table(lt, ids, scorer, col):
    x = lt[(lt.scorer == scorer) & lt.id.isin(ids) & ~lt.tissue.isin(UNRANKED)]
    t = x.groupby("tissue")[col].mean().sort_values(ascending=False).to_frame(col)
    t["rank"] = range(1, len(t) + 1)
    return t, x.id.nunique()


def boot_rank(lt, ids, scorer, col, n_boot=N_BOOT):
    """Eye rank when the loci are resampled with replacement: (2.5%, median, 97.5%)."""
    x = lt[(lt.scorer == scorer) & lt.id.isin(ids) & ~lt.tissue.isin(UNRANKED)]
    w = x.pivot_table(index="id", columns="tissue", values=col)
    if "eye" not in w.columns or len(w) < 20:
        return None, None, None
    W = w.values
    rng = np.random.default_rng(s35.SEED)
    e = list(w.columns).index("eye")
    r = []
    for _ in range(n_boot):
        m = np.nanmean(W[rng.integers(0, len(W), len(W))], axis=0)
        r.append(1 + (m > m[e]).sum())
    return int(np.percentile(r, 2.5)), int(np.median(r)), int(np.percentile(r, 97.5))


# ── report ────────────────────────────────────────────────────────────────────
def analyze(lt):
    sets = read_sets()
    scored = set(lt.id)
    sets = sets[sets.id.isin(scored) | (sets.set == "gwas")]
    dom = s35.domain_map(s35.read_variants())
    gwas_ids = set(sets.id[(sets.set == "gwas")]) & scored
    n = {s: len(set(sets.id[sets.set == s]) & scored) for s in ["gwas", "null", "null_matched"]}
    md = s35.md
    L = ["# Matched null: is the RNA > DNase gap a gene-proximity effect?", "",
         "Generated by `scripts/step39b_matched_null_analysis.py` (rerun on the final data). Null sets: "
         "**random null** = 2,000 random common SNVs (`step33`); **matched null** = 2,000 common SNVs matched "
         "to the GWAS loci on distance to the nearest protein-coding TSS (`step39`, Ensembl 116 = GENCODE 50).", "",
         f"Scored so far: **{n['gwas']:,} GWAS loci** (of 7,782), **{n['null']:,} random-null** and "
         f"**{n['null_matched']:,} matched-null** variants.", ""]

    # a. distance
    tss = s39.load_tss()
    sets["dist_tss"] = s39.dist_to_tss(sets.CHR, sets.pos, tss)
    all_gwas = s35.read_variants().query("set == 'gwas'")
    d_sets = {"GWAS (all 7,782)": s39.dist_to_tss(all_gwas.CHR, all_gwas.pos, tss),
              "GWAS (scored)": sets.dist_tss[sets.id.isin(gwas_ids)].values,
              "random null": sets.dist_tss[sets.set == "null"].values,
              "matched null": sets.dist_tss[sets.set == "null_matched"].values}
    qt = s39.quantile_table(d_sets)
    qt.loc["mean"] = [np.mean(v) for v in d_sets.values()]
    qt.loc["frac < 10 kb"] = [np.mean(v < 10_000) for v in d_sets.values()]
    qt.to_csv(os.path.join(OUT, "a_distance_quantiles.csv"))
    ks = [{"comparison": f"GWAS (all) vs {k}", "KS D": ks_2samp(d_sets["GWAS (all 7,782)"], d_sets[k]).statistic,
           "P": ks_2samp(d_sets["GWAS (all 7,782)"], d_sets[k]).pvalue} for k in ["random null", "matched null"]]
    fmt = qt.copy().astype(object)
    for r in fmt.index:
        fmt.loc[r] = [f"{x:.3f}" if r.startswith("frac") else f"{x:,.0f}" for x in qt.loc[r]]
    L += ["## a. Distance to the nearest protein-coding TSS (bp)", "", md(fmt, disable_numparse=True), "",
          md(pd.DataFrame(ks).round({"KS D": 3}), index=False, floatfmt=".2g"), ""]

    # b. per assay, GWAS loci vs each null
    per_var = per_variant(lt, ["emp_rand", "emp_match", "top5_rand", "top5_match"])
    per_var["set"] = per_var.id.map(sets.set_index("id").set)
    rows = []
    for s in ["gwas", "null", "null_matched"]:
        for sc in ["dnase_rec", "atac_rec", "rna_rec", "dnase_orig", "atac_orig", "rna_orig"]:
            x = per_var[(per_var.set == s) & (per_var.scorer == sc)]
            if x.empty:
                continue
            rows.append({"variants": s, "scorer": sc, "n": len(x),
                         "mean pct vs random null": x.emp_rand.mean(), "mean pct vs matched null": x.emp_match.mean(),
                         "frac>=0.95 vs random": x.top5_rand.mean(), "frac>=0.95 vs matched": x.top5_match.mean()})
    b = pd.DataFrame(rows)
    b.to_csv(os.path.join(OUT, "b_assay_percentiles.csv"), index=False)
    g = b[b.variants == "gwas"].set_index("scorer")
    gap = []
    for kind in ["rec", "orig"]:
        for bg, col in [("random null", "mean pct vs random null"), ("matched null", "mean pct vs matched null")]:
            gap.append({"scorers": kind, "background": bg, "DNase": g.loc[f"dnase_{kind}", col],
                        "ATAC": g.loc[f"atac_{kind}", col], "RNA": g.loc[f"rna_{kind}", col],
                        "RNA - DNase": g.loc[f"rna_{kind}", col] - g.loc[f"dnase_{kind}", col],
                        "(RNA - 0.5) / (DNase - 0.5)": (g.loc[f"rna_{kind}", col] - 0.5) / (g.loc[f"dnase_{kind}", col] - 0.5)})
    gap = pd.DataFrame(gap)
    gap.to_csv(os.path.join(OUT, "b_rna_dnase_gap.csv"), index=False)
    # by domain
    dr = []
    pv = per_var[per_var.set == "gwas"].merge(dom, on="id")
    for (d, sc), x in pv[pv.scorer.isin(REC)].groupby(["domain", "scorer"]):
        dr.append({"domain": d, "scorer": sc, "n": len(x), "vs random": x.emp_rand.mean(), "vs matched": x.emp_match.mean()})
    dr = pd.DataFrame(dr).pivot(index="domain", columns="scorer", values=["vs random", "vs matched"])
    dr.columns = [f"{sc} {bg}" for bg, sc in dr.columns]
    dr.to_csv(os.path.join(OUT, "b_by_domain_rec.csv"))
    L += ["## b. How far GWAS loci sit above each null, per assay", "",
          "Each track's score → percentile among the null variants on the same scorer and track (0.5 = typical "
          "null variant); averaged over all tracks of a variant, then over variants. `frac>=0.95` = share of "
          "(variant, track) pairs above the null's 95th percentile (0.05 expected under the null).", "",
          md(b.round(3), index=False), "",
          "**RNA vs DNase gap for GWAS loci:**", "", md(gap.round(3), index=False), "",
          "By GWAS domain (recommended scorers, mean percentile):", "", md(dr.round(3)), ""]

    # c. eye rank per domain
    fl = fold_eye(lt)
    methods = [("raw pooled (orig scorers, |raw|, original pipeline)", fl, "orig_all", "abs_raw"),
               ("calibrated pooled, matched null (rec scorers)", lt, "rec_all", "emp_match"),
               ("calibrated DNase only, matched null", lt, "dnase_rec", "emp_match"),
               ("calibrated pooled, random null (rec scorers)", lt, "rec_all", "emp_rand"),
               ("calibrated DNase only, random null", lt, "dnase_rec", "emp_rand")]
    summ, full = [], []
    for d in DOMAINS:
        ids = set(dom.id[dom.domain == d]) & gwas_ids
        for lab, frame, sc, col in methods:
            t, nn = rank_table(frame, ids, sc, col)
            if t.empty:
                continue
            full.append(t.rename(columns={col: "score"}).reset_index().assign(domain=d, method=lab))
            lo, med, hi = boot_rank(frame, ids, sc, col)
            summ.append({"domain": d, "n loci": nn, "method": lab,
                         "eye rank": int(t.loc["eye", "rank"]) if "eye" in t.index else None, "of": len(t),
                         "bootstrap 95% range": f"{lo}-{hi}", "top 3": ", ".join(t.index[:3])})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, "c_eye_rank_summary.csv"), index=False)
    full = pd.concat(full)[["domain", "method", "tissue", "score", "rank"]]
    full.to_csv(os.path.join(OUT, "c_rankings.csv"), index=False)
    L += ["## c. Eye rank among tissues, GWAS loci by domain", "",
          "Mean over loci of the per-tissue mean score. `other` and `eye_misfiled` (choroid plexus) are not "
          "ranked; the raw pooled row folds choroid plexus into eye as in the original pipeline.", "",
          md(summ, index=False), "", "Full rankings: `data/null_matched_analysis/c_rankings.csv`.", ""]
    piv = full[full.method.str.contains("matched")].pivot_table(index="tissue", columns=["domain", "method"],
                                                                 values="score")
    piv.columns = [f"{d} loci, {'pooled' if 'pooled' in m else 'DNase'}" for d, m in piv.columns]
    L += ["Matched-null calibrated scores by tissue:", "", md(piv.round(3)), ""]
    # proximity sensitivity per tissue: where do matched-null variants sit against the random null?
    ps = lt[(lt.scorer == "dnase_rec") & lt.id.isin(set(sets.id[sets.set == "null_matched"]))]
    ps = ps.groupby("tissue").agg(pct_vs_random_null=("emp_rand", "mean"), n_tracks=("n_tracks", "first"))
    gw = lt[(lt.scorer == "dnase_rec") & lt.id.isin(gwas_ids)].groupby("tissue")[["emp_rand", "emp_match"]].mean()
    ps = ps.join(gw.rename(columns={"emp_rand": "GWAS vs random", "emp_match": "GWAS vs matched"}))
    ps = ps.sort_values("pct_vs_random_null", ascending=False)
    ps.to_csv(os.path.join(OUT, "c_dnase_proximity_sensitivity.csv"))
    L += ["**Why DNase ranks move with the background.** Mean percentile of the matched-null variants against "
          "the random null, per tissue (DNase, recommended scorer): > 0.5 means a tissue's DNase tracks score "
          "gene-proximal variants higher, so switching to the matched null lowers that tissue's GWAS percentile "
          "more. All GWAS loci pooled.", "", md(ps.round(3)), ""]

    # d. interpretation (numbers filled in; wording chosen by the direction of the result)
    rr = gap[(gap.scorers == "rec") & (gap.background == "random null")].iloc[0]
    rm = gap[(gap.scorers == "rec") & (gap.background == "matched null")].iloc[0]
    shrink = 1 - rm["RNA - DNase"] / rr["RNA - DNase"] if rr["RNA - DNase"] else np.nan
    nm = b[(b.variants == "null") & (b.scorer == "rna_rec")]
    rn_vs_m = nm["mean pct vs matched null"].iloc[0] if len(nm) else np.nan
    persists = rm["RNA - DNase"] > 0.03
    mostly = shrink >= 0.5
    def eye_ranks(prefix):
        x = summ[summ.method.str.startswith(prefix)]
        return ", ".join(f"{d} #{r} (95% {b})" for d, r, b in zip(x.domain, x["eye rank"], x["bootstrap 95% range"]))
    eye_pooled, eye_dnase = eye_ranks("calibrated pooled, matched"), eye_ranks("calibrated DNase only, matched")
    eye_raw, eye_dnase_r = eye_ranks("raw pooled"), eye_ranks("calibrated DNase only, random")
    L += ["## d. Interpretation", "",
          f"- Matching works: the matched null has the GWAS loci's distance-to-TSS distribution "
          f"(KS D = {ks[1]['KS D']:.3f}, vs {ks[0]['KS D']:.3f} for the random null; median "
          f"{np.median(d_sets['matched null']):,.0f} vs {np.median(d_sets['GWAS (all 7,782)']):,.0f} bp for GWAS and "
          f"{np.median(d_sets['random null']):,.0f} bp for the random null).",
          f"- Gene proximity explains part of the RNA excess: random-null variants sit at mean RNA percentile "
          f"{rn_vs_m:.2f} against the matched null (<0.5 = gene-proximal variants score higher in RNA). For GWAS "
          f"loci (recommended scorers) the RNA − DNase gap is {rr['RNA - DNase']:.3f} against the random null "
          f"and {rm['RNA - DNase']:.3f} against the matched null ({shrink:.0%} smaller).",
          (("- The gap is **mostly explained by gene proximity, but a residual persists**: " if mostly else
            "- The gap **persists** with the matched null: ") +
           "GWAS loci still exceed a proximity-matched background more in RNA-seq than in DNase, so pooling "
           "assays on a calibrated scale still up-weights tissues with many RNA tracks. The assay-matched "
           "comparison remains necessary."
           if persists else
           "- With the matched null the gap is **largely gone**: the RNA > DNase excess mostly reflects that GWAS "
           "loci are closer to genes. Pooled calibrated rankings against a proximity-matched null are less "
           "confounded; the assay-matched comparison is still the conservative choice."),
          f"- Eye rank of 12 tissues. Raw pooled (original pipeline): {eye_raw}. Calibrated pooled, matched null: "
          f"{eye_pooled}. Calibrated DNase only, matched null: {eye_dnase} (random null: {eye_dnase_r}). "
          "DNase tissue means differ by only a few percentile points, so DNase ranks are unstable; read them "
          "with the bootstrap ranges.",
          f"- Provisional: based on {n['gwas']:,} of 7,782 GWAS loci; rerun this script when the main run finishes.", ""]

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reaggregate", action="store_true")
    args = ap.parse_args()
    analyze(aggregate(force=args.reaggregate))
