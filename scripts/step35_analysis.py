"""
step35_analysis.py

Analyses A1-A7 on the output of step34_full_rescoring.py.

Stage 1, aggregate: reads data/full_rescoring/parts/*.parquet (~3,500 track rows per variant) and
writes two compact tables to data/analysis/:
  locus_tissue.parquet  per variant x scorer x tissue: mean |raw|, mean |q|, fraction |q| >= 0.95, n tracks
  track_domain.parquet  per domain x scorer x track: mean |raw|, mean |q| over the variants in that domain
Stage 2, analyze: writes reports/analysis_A1_A7.md and CSV tables in data/analysis/.

Scorer codes: dnase_rec / atac_rec / rna_rec = AlphaGenome recommended scorers (calibrated
quantile_score available); dnase_orig / atac_orig / rna_orig = the original pipeline's 501 bp
DIFF_MEAN scorer (raw only). orig_all and rec_all pool the three assays, as the original Fig. 2 did.

Tissue labels: "eye" is the four genuine ocular DNase tracks; the choroid plexus track, which the
original catalog filed under eye, is kept apart as "eye_misfiled". Analyses that reproduce the
original pipeline fold it back into eye.

    python scripts/step35_analysis.py              # aggregate (if parts changed) + analyze
    python scripts/step35_analysis.py --reaggregate
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

NEW = paths.NEW
CATALOG = paths.TRACK_CATALOG   # step01
LOCI = paths.LOCUS_TABLE        # step05: GWAS loci per domain, with ocular ATAC overlap flags
RUN = os.path.join(NEW, "data", "full_rescoring")
AN = os.path.join(NEW, "data", "analysis")
REPORT = os.path.join(NEW, "reports", "analysis_A1_A7.md")

MISFILED = "choroid plexus epithelial cell"
ORIG = ["dnase_orig", "atac_orig", "rna_orig"]
REC = ["dnase_rec", "atac_rec", "rna_rec"]
DOMAINS = ["eye", "kidney", "metabolic", "null"]
N_PERM = 2000
SEED = 20260921


def read_variants():
    # keep_default_na=False: otherwise pandas reads the set label "null" as NaN
    return pd.read_csv(os.path.join(RUN, "variants.csv"), keep_default_na=False)


def track_map():
    cat = pd.read_csv(CATALOG).drop_duplicates("name").set_index("name")
    tissue = cat.tissue_category.copy()
    tissue[(tissue == "eye") & (cat.biosample_name == MISFILED)] = "eye_misfiled"
    return tissue


def domain_map(variants):
    """Rows (id, domain): a GWAS locus can sit in several disease domains; nulls get 'null'."""
    rows = pd.read_csv(LOCI)[["rsid", "disease_class"]].drop_duplicates()
    g = rows.rename(columns={"rsid": "id", "disease_class": "domain"})
    n = variants.loc[variants.set == "null", ["id"]].assign(domain="null")
    return pd.concat([g[g.id.isin(variants.id)], n], ignore_index=True)


# ── Stage 1 ───────────────────────────────────────────────────────────────────
GRID = np.linspace(0, 1, 201)  # null quantile grid, 0.5% resolution


def null_grids(parts, null_ids):
    """Per (scorer, track): quantiles of the score among the null variants.

    Score = |raw| for the original scorers, |quantile score| for the recommended ones. The null
    variants are the empirical background: a variant's percentile against them ("emp") puts every
    track, assay and scorer on one 0-1 scale, even scorers AlphaGenome does not calibrate
    (DIFF_MEAN) and derived statistics AlphaGenome's quantiles do not cover (our max-over-genes RNA).
    """
    vals = []
    for p in parts:
        d = pd.read_parquet(p)
        d = d[d.id.isin(null_ids)]
        if len(d):
            d["s"] = np.where(d.scorer.isin(ORIG), d.raw_score.abs(), d.quantile_score.abs())
            vals.append(d[["scorer", "track_name", "s"]])
    if not vals:
        return None, None
    v = pd.concat(vals)
    q = v.groupby(["scorer", "track_name"]).s.quantile(GRID).unstack()
    index = {f"{s}|{t}": i for i, (s, t) in enumerate(q.index)}
    return index, q.values


def empirical(d, index, G, chunk=100_000):
    """Percentile of each row's score among the null variants on the same scorer and track."""
    s = np.where(d.scorer.isin(ORIG), d.raw_score.abs(), d.quantile_score.abs())
    idx = (d.scorer + "|" + d.track_name).map(index).values
    out = np.full(len(d), np.nan)
    ok = ~pd.isna(idx)
    rows, sv = idx[ok].astype(int), s[ok]
    res = np.empty(len(rows))
    for i in range(0, len(rows), chunk):
        g = G[rows[i:i + chunk]]
        res[i:i + chunk] = (g < sv[i:i + chunk, None]).mean(axis=1)
    out[ok] = res
    return out


def aggregate(force=False):
    parts = sorted(glob.glob(os.path.join(RUN, "parts", "*.parquet")))
    lt_f, td_f = os.path.join(AN, "locus_tissue.parquet"), os.path.join(AN, "track_domain.parquet")
    stamp_f = os.path.join(AN, "aggregated_parts.txt")
    cached = all(os.path.exists(f) for f in (stamp_f, lt_f, td_f))
    if not force and cached and open(stamp_f).read().split() == [os.path.basename(p) for p in parts]:
        return pd.read_parquet(lt_f), pd.read_parquet(td_f)
    tissue = track_map()
    variants = read_variants()
    dom = domain_map(variants)
    index, G = null_grids(parts, set(variants.id[variants.set == "null"]))
    lt, td = [], []
    for p in parts:
        d = pd.read_parquet(p)
        d["tissue"] = d.track_name.map(tissue).fillna("other")
        d["abs_raw"] = d.raw_score.abs()
        d["abs_q"] = d.quantile_score.abs()
        d["top5"] = (d.abs_q >= 0.95).where(d.quantile_score.notna())
        d["emp"] = empirical(d, index, G) if index else np.nan
        d["emp_top5"] = (d.emp >= 0.95).where(d.emp.notna())
        pooled = [d.assign(scorer=np.where(d.scorer.isin(ORIG), "orig_all", "rec_all"))]
        for x in [d] + pooled:
            lt.append(x.groupby(["id", "scorer", "tissue"]).agg(
                abs_raw=("abs_raw", "mean"), abs_q=("abs_q", "mean"), top5=("top5", "mean"),
                emp=("emp", "mean"), emp_top5=("emp_top5", "mean"),
                n_tracks=("abs_raw", "size")).reset_index())
        t = d.merge(dom, on="id")
        td.append(t.groupby(["domain", "scorer", "track_name", "tissue"]).agg(
            sum_raw=("abs_raw", "sum"), sum_q=("abs_q", "sum"), sum_emp=("emp", "sum"),
            n=("abs_raw", "size")).reset_index())
    os.makedirs(AN, exist_ok=True)
    lt = pd.concat(lt, ignore_index=True)
    td = (pd.concat(td).groupby(["domain", "scorer", "track_name", "tissue"])[["sum_raw", "sum_q", "sum_emp", "n"]]
          .sum().reset_index())
    td["abs_raw"], td["abs_q"], td["emp"] = td.sum_raw / td.n, td.sum_q / td.n, td.sum_emp / td.n
    lt.to_parquet(lt_f, index=False)
    td.to_parquet(td_f, index=False)
    with open(stamp_f, "w") as f:
        f.write("\n".join(os.path.basename(p) for p in parts))
    return lt, td


# ── helpers ───────────────────────────────────────────────────────────────────
def as_published(lt):
    """Fold eye_misfiled back into eye (track-count weighted), as the original pipeline did."""
    x = lt.copy()
    x["tissue"] = x.tissue.replace({"eye_misfiled": "eye"})
    w = x.assign(wr=x.abs_raw * x.n_tracks, wq=x.abs_q * x.n_tracks, we=x.emp * x.n_tracks)
    g = w.groupby(["id", "scorer", "tissue"]).agg(wr=("wr", "sum"), wq=("wq", "sum"), we=("we", "sum"),
                                                  n_tracks=("n_tracks", "sum"))
    g["abs_raw"], g["abs_q"], g["emp"] = g.wr / g.n_tracks, g.wq / g.n_tracks, g.we / g.n_tracks
    return g.reset_index()


def ranking(lt, dom, scorer, col, domain):
    ids = dom.loc[dom.domain == domain, "id"]
    x = lt[(lt.scorer == scorer) & lt.id.isin(ids)]
    t = x.groupby("tissue")[col].mean().sort_values(ascending=False).to_frame(col)
    t["rank"] = range(1, len(t) + 1)
    return t, x.id.nunique()


def md(df, **kw):
    return df.to_markdown(**kw)


# ── Stage 2 ───────────────────────────────────────────────────────────────────
def analyze(lt, td):
    from scipy.stats import fisher_exact, spearmanr
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    variants = read_variants()
    dom = domain_map(variants)
    done = set(lt.id)
    n_null, n_gwas = len(done & set(variants.id[variants.set == "null"])), len(done & set(variants.id[variants.set == "gwas"]))
    pub = as_published(lt)
    L = ["# Analyses A1–A7", "", "Generated by `scripts/step35_analysis.py`.",
         f"Variants scored so far: **{n_gwas:,} GWAS loci**, **{n_null:,} null variants**.", ""]

    # A1 + A2: the original pipeline on GWAS loci and on null variants
    L += ["## A1 / A2. Original pipeline (raw DIFF_MEAN, DNase+ATAC+RNA pooled, eye as in the original pipeline)", "",
          "Mean over variants of the per-tissue mean |raw score| — the Fig. 2 metric.", ""]
    ranks = {}
    for d in DOMAINS:
        t, n = ranking(pub, dom, "orig_all", "abs_raw", d)
        if n == 0:
            continue
        ranks[d] = t
        t.to_csv(os.path.join(AN, f"A1A2_orig_ranking_{d}.csv"))
        L += [f"### {d} {'(NULL: random common variants)' if d == 'null' else 'loci'} (n={n:,})", "",
              md(t.round(5)), ""]
    L += ["**Eye rank under the original pipeline:** " +
          ", ".join(f"{d} #{int(t.loc['eye', 'rank'])}" for d, t in ranks.items()), ""]
    # A3: permutation — does the assay mix alone reproduce the ranking?
    L += ["## A3. Permutation: fake tissues with each real tissue's assay mix", "",
          f"For each tissue, {N_PERM:,} fake tissues are built by drawing the same number of DNase, ATAC and "
          "RNA-seq tracks at random from all tissues. The fake tissue's score is the mean of its tracks' mean "
          "|raw| (equal to the Fig. 2 metric, since every variant has every track). Eye original pipeline (5 DNase).", ""]
    tdp = td.copy()
    tdp["tissue"] = tdp.tissue.replace({"eye_misfiled": "eye"})
    rng = np.random.default_rng(SEED)
    perm_rows = []
    for d in [x for x in DOMAINS if x in ranks]:
        sub = tdp[(tdp.domain == d) & tdp.scorer.isin(ORIG)]
        pools = {s: sub[sub.scorer == s].abs_raw.values for s in ORIG}
        mix = sub.groupby(["tissue", "scorer"]).size().unstack(fill_value=0)
        real = sub.groupby("tissue").abs_raw.mean()
        fake_mean, eye_top = {}, 0
        fake_eye = []
        for t in mix.index:
            draws = np.zeros(N_PERM)
            n_tot = mix.loc[t].sum()
            for s in ORIG:
                k = mix.loc[t, s]
                if k:
                    draws += np.array([rng.choice(pools[s], k, replace=False).sum() for _ in range(N_PERM)])
            draws /= n_tot
            fake_mean[t] = draws.mean()
            if t == "eye":
                fake_eye = draws
        others = real.drop("eye")
        np.save(os.path.join(AN, f"A3_fake_eye_draws_{d}.npy"), fake_eye)  # Fig. 2b plots these same draws
        rho = spearmanr(real, pd.Series(fake_mean).reindex(real.index)).correlation
        perm_rows.append({"domain": d, "Spearman(real, assay-mix expectation)": rho,
                          "P(random 5-DNase 'eye' beats every real non-eye tissue)": (fake_eye > others.max()).mean(),
                          "real eye": real["eye"], "median random 5-DNase set": np.median(fake_eye),
                          "best real non-eye": others.max()})
    L += [md(pd.DataFrame(perm_rows).round(4), index=False), ""]
    pd.DataFrame(perm_rows).to_csv(os.path.join(AN, "A3_permutation.csv"), index=False)

    # A4: variance decomposition across tracks
    L += ["## A4. What explains a track's average score: assay type or tissue?", "",
          "Per-track mean score over all GWAS loci, two-way ANOVA (type II), share of total sum of squares.", ""]
    rows = []
    g_all = td[td.domain.isin(["eye", "kidney", "metabolic"])]
    for label, scorers, col, s_col in [("original raw |score|", ORIG, "abs_raw", "sum_raw"),
                                       ("AlphaGenome quantile |q|", REC, "abs_q", "sum_q"),
                                       ("null-calibrated, original scorers", ORIG, "emp", "sum_emp"),
                                       ("null-calibrated, recommended scorers", REC, "emp", "sum_emp")]:
        x = g_all[g_all.scorer.isin(scorers)].groupby(["scorer", "track_name", "tissue"])[[s_col, "n"]].sum()
        if x.empty:
            continue
        x = (x[s_col] / x.n).rename("y").reset_index()
        x["y"] = np.log(x.y) if col == "abs_raw" else x.y
        fit = smf.ols("y ~ C(scorer) + C(tissue)", data=x).fit()
        a = sm.stats.anova_lm(fit, typ=2)
        share = a.sum_sq / a.sum_sq.sum()
        rows.append({"score": label, "assay share": share["C(scorer)"], "tissue share": share["C(tissue)"],
                     "residual (track) share": share["Residual"], "n tracks": len(x)})
    L += [md(pd.DataFrame(rows).round(3), index=False), "", "(raw scores log-transformed before ANOVA)", ""]
    pd.DataFrame(rows).to_csv(os.path.join(AN, "A4_variance.csv"), index=False)

    # A5: calibrated, assay-matched rankings
    L += ["## A5. Calibrated, assay-matched rankings (recommended scorers, |quantile score|)", "",
          "`eye` = 4 genuine ocular DNase tracks; `eye_misfiled` = choroid plexus, shown separately. "
          "`top5` = fraction of tracks with |q| ≥ 0.95. RNA uses the gene with the largest |q| per track, "
          "which inflates |q| for all tissues alike.", ""]
    summary = []
    for s in REC + ["rec_all"]:
        L += [f"### {s}", ""]
        blocks = []
        for d in DOMAINS:
            ids = dom.loc[dom.domain == d, "id"]
            x = lt[(lt.scorer == s) & lt.id.isin(ids)]
            if x.empty:
                continue
            t = x.groupby("tissue")[["abs_q", "top5"]].mean()
            t[f"{d}_rank"] = t.abs_q.rank(ascending=False, method="min").astype(int)
            blocks.append(t.rename(columns={"abs_q": f"{d}_|q|", "top5": f"{d}_top5"}))
            if "eye" in t.index:
                summary.append({"scorer": s, "domain": d, "eye rank": int(t.loc["eye", f"{d}_rank"]),
                                "of": len(t)})
        if blocks:
            tab = pd.concat(blocks, axis=1).sort_values(blocks[0].columns[0], ascending=False)
            L += [md(tab.round(4)), ""]
            tab.to_csv(os.path.join(AN, f"A5_calibrated_{s}.csv"))
    L += ["**Eye rank after calibration (AlphaGenome quantiles):**", "", md(pd.DataFrame(summary), index=False), "",
          "⚠️ `rec_all` pools assays on the AlphaGenome quantile scale, but our RNA statistic (max |q| over "
          "genes) is not on that scale (null mean ≈ 0.87 vs 0.50 for DNase/ATAC), so `rec_all` |q| is NOT a "
          "valid cross-assay comparison. Use the null-calibrated version below.", ""]
    L += ["### Null-calibrated scores (percentile among the 2,000 null variants, per track)", "",
          "Every track, assay and scorer on one 0–1 scale; 0.5 = typical random common variant. "
          "The null domain is ~0.5 by construction and is omitted.", ""]
    summ_e = []
    for s, lab in [("dnase_rec", "recommended DNase"), ("rec_all", "recommended, all assays"),
                   ("orig_all", "ORIGINAL scorer, all assays")]:
        blocks = []
        for d in ["eye", "kidney", "metabolic"]:
            ids = dom.loc[dom.domain == d, "id"]
            x = lt[(lt.scorer == s) & lt.id.isin(ids)]
            if x.empty:
                continue
            t = x.groupby("tissue")[["emp", "emp_top5"]].mean()
            t[f"{d}_rank"] = t.emp.rank(ascending=False, method="min").astype(int)
            blocks.append(t.rename(columns={"emp": f"{d}_emp", "emp_top5": f"{d}_top5"}))
            if "eye" in t.index:
                summ_e.append({"score": lab, "domain": d, "eye rank": int(t.loc["eye", f"{d}_rank"]), "of": len(t)})
        if blocks:
            tab = pd.concat(blocks, axis=1).sort_values(blocks[0].columns[0], ascending=False)
            L += [f"#### {lab} (`{s}`)", "", md(tab.round(4)), ""]
            tab.to_csv(os.path.join(AN, f"A5_nullcal_{s}.csv"))
    L += ["**Eye rank after null calibration:**", "", md(pd.DataFrame(summ_e), index=False), ""]

    # A5c: the five eye-category DNase tracks among all DNase tracks (mean |q| over each domain's loci)
    bio = pd.read_csv(CATALOG).drop_duplicates("name").set_index("name").biosample_name
    dn = td[td.scorer == "dnase_rec"].copy()
    dn["percentile"] = dn.groupby("domain").abs_q.rank(pct=True) * 100
    dn["rank"] = dn.groupby("domain").abs_q.rank(ascending=False, method="min").astype(int)
    dn["n_tracks"] = dn.groupby("domain").abs_q.transform("size")
    eye_tracks = dn[dn.tissue.isin(["eye", "eye_misfiled"])].assign(biosample=lambda z: z.track_name.map(bio))
    a5c = eye_tracks[["domain", "biosample", "track_name", "abs_q", "rank", "n_tracks", "percentile"]] \
        .sort_values(["biosample", "domain"])
    a5c.to_csv(os.path.join(AN, "A5c_eye_track_percentiles.csv"), index=False)
    L += ["### A5c. Eye-category DNase tracks among all DNase tracks (recommended scorer, mean |q|)", "",
          "Rank 1 = the track with the largest mean |q| over the domain's loci; percentile among all DNase tracks.", "",
          md(a5c.pivot(index="biosample", columns="domain", values="percentile").round(1)), "",
          md(a5c.pivot(index="biosample", columns="domain", values="rank")), ""]

    # A5b: how far GWAS loci sit above the null, per assay — why pooled calibrated scores re-confound
    L += ["### A5b. GWAS loci vs null, per assay (mean null-percentile; null = 0.50)", "",
          "If GWAS loci exceed the null by different amounts in different assays, pooling assays on any "
          "calibrated scale still rewards tissues rich in the more-enriched assay.", ""]
    rows = []
    for d in ["eye", "kidney", "metabolic", "all GWAS"]:
        ids = dom.loc[dom.domain != "null", "id"] if d == "all GWAS" else dom.loc[dom.domain == d, "id"]
        x = lt[lt.scorer.isin(ORIG + REC) & lt.id.isin(ids)]
        if x.empty:
            continue
        per = x.groupby("scorer").apply(lambda z: pd.Series({
            "mean": np.average(z.emp, weights=z.n_tracks),
            "top5": np.average(z.emp_top5, weights=z.n_tracks)}), include_groups=False)
        for sc, r in per.iterrows():
            rows.append({"loci": d, "scorer": sc, "mean null-percentile": r["mean"], "fraction >= 0.95": r["top5"]})
    if rows:
        a5b = pd.DataFrame(rows)
        a5b.to_csv(os.path.join(AN, "A5b_gwas_vs_null_by_assay.csv"), index=False)
        L += [md(a5b.pivot(index="scorer", columns="loci", values="mean null-percentile").round(3)), ""]

    # A6: multi-disease gradient
    L += ["## A6. Multi-disease gradient: raw vs calibrated", "",
          "Ratio of mean score at loci shared by 3 disease categories vs 1, per tissue.", ""]
    nc = pd.read_csv(LOCI).drop_duplicates("rsid").set_index("rsid").n_disease_cats
    rows = []
    for label, frame, s, col in [("raw (original pipeline)", pub, "orig_all", "abs_raw"),
                                 ("calibrated DNase |q|", lt, "dnase_rec", "abs_q"),
                                 ("null-calibrated, original scorers", lt, "orig_all", "emp"),
                                 ("null-calibrated, recommended all assays", lt, "rec_all", "emp")]:
        x = frame[(frame.scorer == s) & frame.id.isin(nc.index)].copy()
        x["n"] = x.id.map(nc)
        for t, z in x.groupby("tissue"):
            m = z.groupby("n")[col].mean()
            if {1, 3} <= set(m.index):
                med = z.groupby("n")[col].median()
                rows.append({"score": label, "tissue": t, "ratio 3/1": m[3] / m[1],
                             "median ratio 3/1": med[3] / med[1],
                             "rho": spearmanr(z.n, z[col]).correlation})
    g = pd.DataFrame(rows)
    if not g.empty:
        piv = g.pivot(index="tissue", columns="score", values="ratio 3/1")
        L += ["Mean-based ratio (sensitive to a few extreme loci; raw per-locus scores are heavy-tailed, "
              "mean/median 6–30x):", "", md(piv.sort_values(piv.columns[0], ascending=False).round(3)), "",
              "Median-based ratio (robust):", ""]
        pm = g.pivot(index="tissue", columns="score", values="median ratio 3/1")
        L += [md(pm.sort_values(pm.columns[0], ascending=False).round(3)), ""]
        g.to_csv(os.path.join(AN, "A6_gradient.csv"), index=False)

    # A7: TM / RGC open chromatin, raw vs calibrated
    L += ["## A7. hTMC / iPSC-RGC open chromatin: raw vs calibrated", ""]
    fl = pd.read_csv(LOCI).drop_duplicates("rsid").set_index("rsid")[["in_hTMC_atac", "in_iPSC_RGC_atac"]]
    for label, frame, s, col in [("raw (original pipeline)", pub, "orig_all", "abs_raw"),
                                 ("calibrated DNase |q|", lt, "dnase_rec", "abs_q")]:
        w = frame[frame.scorer == s].pivot_table(index="id", columns="tissue", values=col).join(fl, how="inner")
        tis = [c for c in w.columns if c not in fl.columns and c != "eye_misfiled"]
        if len(w) < 100:
            continue
        ors = {}
        for t in tis:
            q = pd.qcut(w[t].rank(method="first"), 4, labels=[1, 2, 3, 4])
            x = w.in_hTMC_atac.astype(bool)
            ors[t] = fisher_exact([[x[q == 4].sum(), (~x[q == 4]).sum()], [x[q == 1].sum(), (~x[q == 1]).sum()]])[0]
        o = pd.Series(ors).sort_values(ascending=False)
        tag = "raw" if s == "orig_all" else "calibrated_dnase"
        o.rename("hTMC_OR_Q4_vs_Q1").rename_axis("tissue").to_csv(os.path.join(AN, f"A7_hTMC_or_{tag}.csv"))
        Z = w[tis].rank(pct=True)
        Z["non_eye_mean"] = Z.drop(columns="eye").mean(axis=1)
        res = []
        for c in fl.columns:
            fit = sm.Logit(w[c].astype(int), sm.add_constant(Z[["eye", "non_eye_mean"]])).fit(disp=0)
            res.append(f"{c}: eye coef {fit.params['eye']:.2f} (p={fit.pvalues['eye']:.1e}), "
                       f"non-eye mean {fit.params['non_eye_mean']:.2f} (p={fit.pvalues['non_eye_mean']:.1e})")
        L += [f"### {label} (n={len(w):,})", "",
              f"hTMC Q4-vs-Q1 OR by tissue score: eye {o['eye']:.1f} (rank {list(o.index).index('eye') + 1} of {len(o)}); "
              f"top: {', '.join(f'{k} {v:.1f}' for k, v in o.head(3).items())}", "",
              "Joint model (does eye add information beyond the other tissues?):", ""] + [f"- {r}" for r in res] + [""]

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reaggregate", action="store_true")
    args = ap.parse_args()
    lt, td = aggregate(force=args.reaggregate)
    analyze(lt, td)
