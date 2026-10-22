"""
step40b_immune_analysis.py

Benchmark B2 (generalisation). Does the false "tissue hub" from pooling raw
scores across assays appear for an unrelated trait set, and does assay-matched calibration recover
the known answer? Immune-mediated diseases are the test case: independent S-LDSC puts their
heritability in immune cells (Finucane 2018 Nat Genet, Supp. Table 7).

Inputs (all read-only)
  data/immune_loci/immune_variants.csv       lead SNVs for CD, UC, RA, MS, SLE, T1D (step40)
  data/immune_scoring/parts/*.parquet        step34 scores for those variants (six scorers)
  data/full_rescoring/parts/*.parquet        main run; ONLY its 2,000 random null variants
                                             (set == "null" in its variants.csv) are used, as the
                                             empirical calibration background. Unreadable parts
                                             (e.g. being written) are skipped.
  data/external/Finucane2018_MOESM3_ESM.xlsx  S-LDSC reference via step38.reference()

Helpers come from step35 (null_grids, empirical, track_map, as_published, ORIG, REC) and step38
(reference, category); step35's aggregate() is not called and nothing is written to data/analysis/.

Methods compared (tissue score of a locus = mean over the tissue's tracks; tissue score of a set
= mean over loci; `other` and the misfiled choroid plexus track are never ranked):
  a. raw pooled        original DIFF_MEAN scorers, DNase+ATAC+RNA pooled, mean |raw|; choroid
                       plexus folded into eye, as in the original pipeline
  b. calibrated pooled recommended scorers, all assays pooled, null percentile per track
  c. assay-matched     recommended DNase: AlphaGenome |quantile| and null percentile
                       (ATAC / RNA null percentile and raw DNase-only shown as extra rows)
  d. per trait: Spearman with S-LDSC mean tau* over the 11 shared categories (as in step38),
     raw pooled vs assay-matched calibrated, paired over the six traits.

Output: reports/immune_generalization.md, data/immune_analysis/*.csv

    python scripts/step40b_immune_analysis.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step35_analysis as s35  # noqa: E402
from step35_analysis import ORIG, REC, empirical, null_grids, track_map  # noqa: E402
from step38_benchmark_sldsc import category, reference  # noqa: E402,F401  (category used by reference())

NEW = os.path.dirname(HERE)
VAR_F = os.path.join(NEW, "data", "immune_loci", "immune_variants.csv")
COUNTS_F = os.path.join(NEW, "data", "immune_loci", "immune_trait_counts.csv")
SCORE = os.path.join(NEW, "data", "immune_scoring")
MAIN = os.path.join(NEW, "data", "full_rescoring")
AN = os.path.join(NEW, "data", "immune_analysis")
REPORT = os.path.join(NEW, "reports", "immune_generalization.md")

TRAITS = {"CD": "Crohn's disease", "UC": "Ulcerative colitis", "RA": "Rheumatoid Arthritis",
          "MS": "Multiple sclerosis", "SLE": "Lupus", "T1D": "Type 1 Diabetes"}
NOT_TISSUES = {"other", "eye_misfiled"}
RAW_LAB = "a. raw pooled (original scorers, original pipeline)"
METHODS = [  # label, frame, scorer, column
    (RAW_LAB, "pub", "orig_all", "abs_raw"),
    ("b. calibrated pooled (recommended, null pct)", "lt", "rec_all", "emp"),
    ("b'. calibrated pooled (original scorers, null pct)", "lt", "orig_all", "emp"),
    ("c. DNase |q| (AlphaGenome quantile)", "lt", "dnase_rec", "abs_q"),
    ("c. DNase null pct", "lt", "dnase_rec", "emp"),
    ("c'. ATAC null pct", "lt", "atac_rec", "emp"),
    ("c'. RNA null pct", "lt", "rna_rec", "emp"),
    ("raw DNase only (original scorer, uncalibrated)", "lt", "dnase_orig", "abs_raw"),
]
MATCHED = ["c. DNase |q| (AlphaGenome quantile)", "c. DNase null pct"]
N_BOOT, SEED = 1000, 20260921


def readable(parts):
    ok, bad = [], []
    for p in parts:
        try:
            pq.read_metadata(p)
            ok.append(p)
        except Exception:
            bad.append(os.path.basename(p))
    return ok, bad


def per_locus(parts, keep_ids, index, G, tissue):
    """Per variant x scorer x tissue means, for the six scorers and the pooled orig_all / rec_all."""
    lt = []
    for p in parts:
        d = pd.read_parquet(p)
        d = d[d.id.isin(keep_ids)]
        if d.empty:
            continue
        d["tissue"] = d.track_name.map(tissue).fillna("other")
        d["abs_raw"] = d.raw_score.abs()
        d["abs_q"] = d.quantile_score.abs()
        d["emp"] = empirical(d, index, G)
        pooled = d.assign(scorer=np.where(d.scorer.isin(ORIG), "orig_all", "rec_all"))
        for x in (d, pooled):
            lt.append(x.groupby(["id", "scorer", "tissue"]).agg(
                abs_raw=("abs_raw", "mean"), abs_q=("abs_q", "mean"), emp=("emp", "mean"),
                n_tracks=("abs_raw", "size")).reset_index())
    return pd.concat(lt, ignore_index=True) if lt else pd.DataFrame()


def wide(frames, fk, scorer, col, ids=None):
    x = frames[fk]
    x = x[(x.scorer == scorer) & ~x.tissue.isin(NOT_TISSUES)]
    if ids is not None:
        x = x[x.id.isin(ids)]
    return x.pivot_table(index="id", columns="tissue", values=col)


def rank_of(m, t):
    return int(m.rank(ascending=False, method="min")[t]) if t in m.index else None


def main():
    os.makedirs(AN, exist_ok=True)
    v = pd.read_csv(VAR_F, keep_default_na=False)
    tissue = track_map()

    # calibration background: the main run's null variants only
    mv = pd.read_csv(os.path.join(MAIN, "variants.csv"), keep_default_na=False)
    null_ids = set(mv.id[mv.set == "null"])
    main_parts, bad_main = readable(sorted(glob.glob(os.path.join(MAIN, "parts", "*.parquet"))))
    null_parts = [p for p in main_parts
                  if pq.read_table(p, columns=["id"]).column("id").to_pandas().isin(null_ids).any()]
    index, G = null_grids(null_parts, null_ids)
    n_null = len(set().union(*[set(pq.read_table(p, columns=["id"]).column("id").to_pylist()) for p in null_parts])
                 & null_ids)

    parts, bad_imm = readable(sorted(glob.glob(os.path.join(SCORE, "parts", "*.parquet"))))
    lt = per_locus(parts, set(v.id), index, G, tissue)
    lt_null = per_locus(null_parts, null_ids, index, G, tissue)
    lt.to_parquet(os.path.join(AN, "locus_tissue.parquet"), index=False)
    frames = {"lt": lt, "pub": s35.as_published(lt)}
    nframes = {"lt": lt_null, "pub": s35.as_published(lt_null)}
    scored = set(lt.id)
    v = v[v.id.isin(scored)]
    trait_ids = {t: set(v.id[v.trait.str.split(";").apply(lambda s, t=t: t in s)]) for t in TRAITS}

    # assay composition per tissue (for "DNase-heavy")
    one = pd.read_parquet(parts[0])
    one = one[(one.id == one.id.iloc[0]) & one.scorer.isin(ORIG)]
    one["tissue"] = one.track_name.map(tissue).fillna("other").replace({"eye_misfiled": "eye"})
    comp = one.groupby(["tissue", "scorer"]).size().unstack(fill_value=0)
    comp["n_tracks"] = comp.sum(axis=1)
    comp["dnase_share"] = comp.get("dnase_orig", 0) / comp.n_tracks

    # ── a-c: rankings over all immune loci (and the null variants for reference) ──
    rng = np.random.default_rng(SEED)
    rank_rows, tabs = [], {}
    for label, fk, sc, col in METHODS:
        w = wide(frames, fk, sc, col)
        m = w.mean().sort_values(ascending=False)
        wn = wide(nframes, fk, sc, col)
        mn = wn.mean() if len(wn) else pd.Series(dtype=float)
        tab = pd.DataFrame({"immune_loci": m, "rank": m.rank(ascending=False, method="min").astype(int),
                            "null_variants": mn.reindex(m.index),
                            "null_rank": mn.rank(ascending=False, method="min").reindex(m.index)})
        tabs[label] = tab
        # locus bootstrap of the top tissue
        W = w.values
        top = []
        for _ in range(N_BOOT):
            top.append(w.columns[np.nanmean(W[rng.integers(0, len(W), len(W))], axis=0).argmax()])
        top = pd.Series(top).value_counts(normalize=True)
        rank_rows.append({"method": label, "n_loci": len(w), "n_tissues": len(m), "top tissue": m.index[0],
                          "immune rank": rank_of(m, "immune"), "eye rank": rank_of(m, "eye"),
                          "P_boot(immune #1)": top.get("immune", 0.0), "P_boot(eye #1)": top.get("eye", 0.0),
                          "null: top tissue": mn.idxmax() if len(mn) else None,
                          "null: eye rank": rank_of(mn, "eye") if len(mn) else None})
        tab.to_csv(os.path.join(AN, f"ranking_{sc}_{col}.csv"))
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(os.path.join(AN, "rank_summary.csv"), index=False)

    # per-trait ranks
    tr_rows = []
    for t, ids in trait_ids.items():
        for label, fk, sc, col in METHODS:
            m = wide(frames, fk, sc, col, ids).mean()
            tr_rows.append({"trait": t, "n_loci": len(ids), "method": label, "top": m.idxmax(),
                            "immune rank": rank_of(m, "immune"), "eye rank": rank_of(m, "eye")})
    tr = pd.DataFrame(tr_rows)
    tr.to_csv(os.path.join(AN, "per_trait_ranks.csv"), index=False)

    # ── d: S-LDSC benchmark (exactly as step38: mean tau*, 11 shared categories) ──
    ref, mapped = reference()
    rows = []
    for t, ids in trait_ids.items():
        r = ref[ref.Phenotype == TRAITS[t]].set_index("category")
        for label, fk, sc, col in METHODS:
            x = frames[fk]
            x = x[(x.scorer == sc) & x.id.isin(ids) & ~x.tissue.isin(NOT_TISSUES | {"eye"})]
            m = x.groupby("tissue")[col].mean()
            both = pd.concat([m.rename("ag"), r.tau], axis=1, join="inner")
            rows.append({"trait": t, "method": label, "n_loci": len(ids), "n_categories": len(both),
                         "rho_tau": spearmanr(both.ag, both.tau).correlation,
                         "sldsc_top": both.tau.idxmax(), "method_top": both.ag.idxmax(),
                         "rank_of_sldsc_top": int(both.ag.rank(ascending=False)[both.tau.idxmax()])})
    bench = pd.DataFrame(rows)
    bench.to_csv(os.path.join(AN, "sldsc_benchmark.csv"), index=False)
    piv = bench.pivot_table(index=["trait", "n_loci"], columns="method", values="rho_tau")
    piv = piv[[m[0] for m in METHODS if m[0] in piv.columns]]
    paired = []
    a = piv[RAW_LAB]
    for label in [m[0] for m in METHODS[1:]]:
        b = piv[label]
        try:
            p = wilcoxon(b, a).pvalue
        except ValueError:
            p = np.nan
        paired.append({"method vs raw pooled": label, "mean rho (method)": b.mean(), "mean rho (raw)": a.mean(),
                       "mean difference": (b - a).mean(), "traits higher": f"{int((b > a).sum())}/{len(a)}",
                       "Wilcoxon p (paired, 2-sided)": p})
    paired = pd.DataFrame(paired)
    paired.to_csv(os.path.join(AN, "sldsc_paired.csv"), index=False)
    ref_top = ref[ref.Phenotype.isin(TRAITS.values())].sort_values("tau", ascending=False).groupby("Phenotype").head(3)

    # ── report ──
    R = ranks.set_index("method")
    raw, pooled = R.loc[RAW_LAB], R.loc["b. calibrated pooled (recommended, null pct)"]
    dq, de = R.loc[MATCHED[0]], R.loc[MATCHED[1]]
    pr = paired.set_index("method vs raw pooled")
    nt = raw["n_tissues"]
    S = ["## Plain-English summary", ""]
    hub = raw["top tissue"] != "immune" and comp.dnase_share.get(raw["top tissue"], 0) >= comp.dnase_share.drop("other", errors="ignore").max()
    S.append(f"- **{'The artifact generalizes' if hub else 'Raw pooled ranking'}.** With the original pipeline (raw scores "
             f"pooled over DNase, ATAC and RNA), {len(scored):,} lead variants for six immune-mediated diseases put "
             f"**{raw['top tissue']}** first (DNase share of its tracks {comp.dnase_share.get(raw['top tissue'], np.nan):.0%}; "
             f"eye #{raw['eye rank']:.0f}, immune #{raw['immune rank']} of {nt}). Random common variants give "
             f"**{raw['null: top tissue']}** first under the same pipeline (eye #{raw['null: eye rank']:.0f}), so the "
             f"raw ranking {'reflects assay composition, not immune biology' if hub else 'should be read with that in mind'}.")
    bt = tabs["b. calibrated pooled (recommended, null pct)"].drop(index="immune", errors="ignore")
    rna_share = comp.get("rna_orig", 0) / comp.n_tracks
    rho_rna = spearmanr(bt.immune_loci, rna_share.reindex(bt.index)).correlation
    S.append(f"- **Calibrated but pooled across assays:** null-percentile scores pooled over all assays put "
             f"**{pooled['top tissue']}** first (immune #{pooled['immune rank']}, eye #{pooled['eye rank']:.0f}). "
             f"Immune loci exceed the null more in RNA than in DNase, so pooling still favours RNA-rich tissues: among "
             f"the non-immune tissues, Spearman(score, RNA share of tracks) = {rho_rna:.2f}"
             f"{'; here the immune signal is strong enough to stay on top, but the eye (DNase-only) drops to last' if pooled['top tissue'] == 'immune' else ''}.")
    best = min(dq["immune rank"], de["immune rank"])
    verdict = ("recovers the known answer" if dq["immune rank"] == 1 and de["immune rank"] == 1 else
               "partly recovers the known answer" if best <= 3 else "does not recover the known answer")
    S.append(f"- **Assay-matched calibration {verdict}.** Comparing DNase tracks only, "
             f"immune ranks #{dq['immune rank']} (AlphaGenome |quantile|) and #{de['immune rank']} (null percentile) of "
             f"{dq['n_tissues']}; bootstrap over loci: immune #1 in {dq['P_boot(immune #1)']:.0%} / "
             f"{de['P_boot(immune #1)']:.0%} of resamples.")
    S.append(f"- **Agreement with independent S-LDSC (6 traits, 11 categories):** mean Spearman "
             f"{pr.loc[MATCHED[1], 'mean rho (raw)']:.2f} for raw pooled vs "
             f"{pr.loc[MATCHED[0], 'mean rho (method)']:.2f} (DNase |q|) and {pr.loc[MATCHED[1], 'mean rho (method)']:.2f} "
             f"(DNase null pct); higher in {pr.loc[MATCHED[0], 'traits higher']} and {pr.loc[MATCHED[1], 'traits higher']} "
             f"traits (paired Wilcoxon p = {pr.loc[MATCHED[0], 'Wilcoxon p (paired, 2-sided)']:.2g} and "
             f"{pr.loc[MATCHED[1], 'Wilcoxon p (paired, 2-sided)']:.2g}; with 6 traits the smallest possible p is 0.031).")
    S.append("  Caveat: the reference has no eye category, so this benchmark (like step38) drops eye and cannot "
             "penalise the raw pipeline's eye artifact; with eye removed, raw pooled ranks the S-LDSC top category "
             f"#{bench[(bench.method == RAW_LAB)].rank_of_sldsc_top.median():.1f} (median over traits). Mean-tau* reference tops are not always immune (see table in d), and 11-category "
             "Spearman values are noisy.")
    S.append("")

    counts = pd.read_csv(COUNTS_F) if os.path.exists(COUNTS_F) else None
    L = ["# B2: generalization to immune-mediated diseases", "",
         "Generated by `scripts/step40b_immune_analysis.py` (loci: `step40_build_immune_loci.py`; scores: "
         "`step34_full_rescoring.py --variants data/immune_loci/immune_variants.csv --out data/immune_scoring`).", "",
         f"Immune lead variants scored: **{len(scored):,}** of {len(pd.read_csv(VAR_F))} "
         f"(parts read {len(parts)}, unreadable {len(bad_imm)}). Calibration background: **{n_null:,}** random "
         f"common null variants from the main run (parts read {len(null_parts)}; unreadable main-run parts skipped: "
         f"{len(bad_main)}).", ""] + S
    if counts is not None:
        L += ["## Loci per trait (GWAS Catalog, P < 5e-8, 500 kb distance clumping)", "",
              counts.to_markdown(index=False), "",
              "Variants scored per trait (a variant that leads for several traits counts for each): " +
              ", ".join(f"{t} {len(i)}" for t, i in trait_ids.items()), ""]
    L += ["## Summary of rankings (all immune loci)", "",
          "`P_boot` = fraction of 1,000 locus bootstrap resamples in which the tissue ranks first. `null:` columns = "
          "same method on the 2,000 random null variants (for null-percentile methods every tissue is ~0.5 there "
          "by construction, so the null ranking is noise).", "",
          ranks.round(3).to_markdown(index=False), ""]
    L += ["## a. Raw pooled (the original pipeline)", "",
          "Mean |raw DIFF_MEAN| over each tissue's DNase+ATAC+RNA tracks. `dnase_share` = fraction of the "
          "tissue's tracks that are DNase (raw DNase scores are several-fold larger than RNA).", "",
          tabs[RAW_LAB].join(comp[["n_tracks", "dnase_share"]]).round(4).to_markdown(), "",
          "Raw DNase only (assay-matched but uncalibrated):", "",
          tabs["raw DNase only (original scorer, uncalibrated)"].round(4).to_markdown(), ""]
    L += ["## b. Calibrated but pooled across assays", "",
          "Per-track percentile among the null variants (0.5 = typical random variant), averaged over all "
          "of a tissue's tracks regardless of assay.", "",
          tabs["b. calibrated pooled (recommended, null pct)"].round(4).to_markdown(), "",
          "Same with the original scorers:", "",
          tabs["b'. calibrated pooled (original scorers, null pct)"].round(4).to_markdown(), "",
          "How far immune loci exceed the null in each assay (mean null percentile over all tracks):", ""]
    ex = lt[lt.scorer.isin(REC) & ~lt.tissue.isin(NOT_TISSUES)].assign(w=lambda z: z.emp * z.n_tracks)
    ex = ex.groupby("scorer").apply(lambda z: z.w.sum() / z.n_tracks.sum(), include_groups=False).rename("mean null pct")
    L += [ex.round(4).to_frame().to_markdown(), ""]
    L += ["## c. Assay-matched calibrated", ""]
    for lab in MATCHED + ["c'. ATAC null pct", "c'. RNA null pct"]:
        L += [f"### {lab}", "", tabs[lab].round(4).to_markdown(), ""]
    L += ["### Per trait: rank of immune (and eye) under each method", "",
          tr.pivot_table(index=["trait", "n_loci"], columns="method", values="immune rank")
            [[m[0] for m in METHODS]].to_markdown(), "",
          "Top tissue per trait:", "",
          tr.pivot_table(index="trait", columns="method", values="top", aggfunc="first")[[m[0] for m in METHODS]].to_markdown(), ""]
    L += ["## d. Agreement with S-LDSC (Finucane 2018 Supp. Table 7)", "",
          "Spearman across the 11 shared categories between the S-LDSC category score (mean normalized tau*, as in "
          "step38; eye and `other` excluded) and each method's tissue score for that trait's loci.", "",
          piv.round(2).to_markdown(), "",
          "Paired comparison with raw pooled over the six traits:", "",
          paired.round(3).to_markdown(index=False), "",
          "S-LDSC top 3 categories per trait (mean tau*):", "",
          ref_top[["Phenotype", "category", "tau", "n_annot"]].round(3).to_markdown(index=False), "",
          "Rank each method gives the S-LDSC top category:", "",
          bench.pivot_table(index=["trait", "sldsc_top"], columns="method", values="rank_of_sldsc_top")
          [[m[0] for m in METHODS]].to_markdown(), ""]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {REPORT}")
    print(ranks.to_string())
    print(paired.round(3).to_string())


if __name__ == "__main__":
    main()
