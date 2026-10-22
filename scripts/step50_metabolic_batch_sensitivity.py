"""
step50_metabolic_batch_sensitivity.py

Sensitivity of the metabolic results to 199 loci that the published locus set lacks.

In the Ensembl query of June 2026 (step03), the first batch of 200 metabolic rsIDs returned no answer,
so 199 of the 200 most significant metabolic loci were marked "unresolved" and never analysed (the
200th was resolved). This step resolves them now, scores the ones not already scored as eye or kidney
loci, and repeats the main metabolic analyses with and without them.

  prepare   query Ensembl for the 199 rsIDs, resolve alleles as step03 does, drop loci within
            524,288 bp of a chromosome start, audit ALT alleles at multiallelic sites as step04 does
            (a never-observed ALT is replaced by the most frequent observed one, as in step34), and
            write the variants that still need scores
  (score)   python scripts/step34_full_rescoring.py --variants data/metabolic_batch/variants_to_score.csv
                                                    --out data/metabolic_batch_scoring
  analyze   metabolic results with the published loci (n = 5,150) and with the added loci: raw pooled
            summary, calibrated pooled summary (random null), DNase-only summaries (AlphaGenome quantile
            score, random-null and matched-null percentiles; mean and strong-effect fraction),
            specificity under the same six combinations, and the rank of the expected tissue for the six
            metabolic traits

Output: data/metabolic_batch/ (resolved loci, Ensembl answers, variants to score),
        data/analysis/step50_metabolic_batch.csv, reports/metabolic_batch_sensitivity.md

    python scripts/step50_metabolic_batch_sensitivity.py prepare
    python scripts/step50_metabolic_batch_sensitivity.py analyze
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

OUT = os.path.join(paths.DATA, "metabolic_batch")
RUN = os.path.join(paths.DATA, "metabolic_batch_scoring")
REPORT = os.path.join(paths.REPORTS, "metabolic_batch_sensitivity.md")
HALF_WINDOW = 1_048_576 // 2
BATCH = 200


def failed_batch():
    """Metabolic loci of the first Ensembl batch that came back unresolved."""
    m = pd.read_csv(os.path.join(paths.RESOLVED_DIR, "metabolic_loci_resolved.csv"))
    first = set(m.rsid.drop_duplicates().iloc[:BATCH])
    return m[m.rsid.isin(first) & (m.resolution == "unresolved")]


def prepare():
    import step03_resolve_alleles as s03
    import step04_multiallelic_audit as s04
    import step34_full_rescoring as s34

    os.makedirs(OUT, exist_ok=True)
    lost = failed_batch()
    cache_f = os.path.join(OUT, "ensembl_allele_strings.json")
    if os.path.exists(cache_f):
        answers = json.load(open(cache_f))
    else:
        answers = s03.query_ensembl(sorted(lost.rsid.unique()))
        json.dump(dict(sorted(answers.items())), open(cache_f, "w"), indent=0)
    cols = [c for c in lost.columns if c not in ("risk_allele", "ref_allele", "alt_allele",
                                                 "ensembl_allele_string", "resolution")]
    res = s03.resolve(lost[cols], answers)
    res["multiallelic"] = res.ensembl_allele_string.fillna("").map(
        lambda s: sum(a in ("A", "C", "G", "T") for a in s.split("/")) >= 3)
    res["excluded"] = np.where(res.resolution.str.startswith("unresolved"), res.resolution,
                               np.where(res.pos < HALF_WINDOW, "within 524 kb of chromosome start", ""))

    ok = res[res.excluded == ""].copy()
    risky = ok[ok.resolution.str.startswith("ensembl_alt") & ok.multiallelic]
    ok["alt_fixed"] = False
    if len(risky):
        aud = s04.audit(risky[s04.COLS])
        aud.to_csv(os.path.join(OUT, "multiallelic_alt_audit.csv"), index=False)
        bad = aud[aud.verdict.str.startswith("SUSPECT")].copy()
        if len(bad):
            bad["new_alt"] = bad.apply(s34.observed_alt, axis=1)
            fix = bad.set_index("rsid").new_alt
            hit = ok.rsid.isin(fix.index)
            ok.loc[hit, "alt_allele"] = ok.loc[hit, "rsid"].map(fix)
            ok.loc[hit, "alt_fixed"] = True
    res = res.merge(ok[["rsid", "alt_allele", "alt_fixed"]].rename(columns={"alt_allele": "alt_scored"}),
                    on="rsid", how="left")
    res.to_csv(os.path.join(OUT, "loci_resolved.csv"), index=False)

    scored = set(pd.read_csv(os.path.join(paths.DATA, "full_rescoring", "variants.csv")).id)
    todo = ok[~ok.rsid.isin(scored) & ok.alt_allele.isin(list("ACGT"))]
    v = pd.DataFrame({"id": todo.rsid, "set": "gwas", "CHR": todo.CHR.astype(int), "pos": todo.pos.astype(int),
                      "ref_allele": todo.ref_allele, "alt_allele": todo.alt_allele, "alt_fixed": todo.alt_fixed})
    v.to_csv(os.path.join(OUT, "variants_to_score.csv"), index=False)
    print(f"lost loci: {len(lost)}; resolved now: {(res.excluded == '').sum()} "
          f"({res.resolution.value_counts().to_dict()}); excluded: {res.excluded[res.excluded != ''].value_counts().to_dict()}")
    print(f"already scored as eye/kidney loci: {ok.rsid.isin(scored).sum()}; to score: {len(v)} "
          f"(ALT replaced at {int(v.alt_fixed.sum())}) -> {os.path.join(OUT, 'variants_to_score.csv')}")


# ── analysis ──────────────────────────────────────────────────────────────────────────────────────
THR = 0.95
DOMAINS = ["eye", "kidney", "metabolic"]
METABOLIC_TRAITS = ["T2D", "hypertension", "BMI", "LDL", "HDL", "triglycerides"]


def _combined_matrices():
    """Main-run matrices plus the added metabolic loci, same columns, calibrated against the same nulls."""
    import step42_uncertainty as s42
    main = s42.build_matrices()
    extra = s42.build_matrices(run_dir=RUN, tag="metabolic_batch")
    out = {}
    for k in ("raw", "q", "emp_rec"):
        a = pd.DataFrame(main[f"{k}_X"], index=main[f"{k}_ids"], columns=main[f"{k}_cols"])
        b = pd.DataFrame(extra[f"{k}_X"], index=extra[f"{k}_ids"], columns=extra[f"{k}_cols"]).reindex(columns=a.columns)
        m = pd.concat([a, b])
        out[f"{k}_X"], out[f"{k}_ids"], out[f"{k}_cols"] = m.values, m.index.values.astype(str), m.columns.values.astype(str)
    return out


def _matched_dnase():
    """DNase |q| as a percentile among the gene-distance-matched null, for main and added loci."""
    import step35_analysis as s35
    import step49_recommended_pipeline_check as s49
    X, ids, cols = s49.dnase_matched_matrix()
    a = pd.DataFrame(X, index=ids, columns=cols)
    m = pd.read_csv(os.path.join(s49.MATCH_RUN, "variants.csv"), keep_default_na=False)
    index, G = s35.null_grids(sorted(glob.glob(os.path.join(s49.MATCH_RUN, "parts", "*.parquet"))), set(m.id))
    mats = []
    for p in sorted(glob.glob(os.path.join(RUN, "parts", "*.parquet"))):
        d = pd.read_parquet(p)
        d = d[d.scorer == "dnase_rec"].copy()
        d["emp"] = s35.empirical(d, index, G)
        k = d.groupby(["id", "scorer", "track_name"]).cumcount().astype(str)
        d["col"] = d.scorer + "|" + d.track_name + "|" + k
        mats.append(d.pivot_table(index="id", columns="col", values="emp", aggfunc="first"))
    b = pd.concat(mats).reindex(columns=a.columns)
    m = pd.concat([a, b])
    return m.values, m.index.values.astype(str), m.columns.values.astype(str)


def _tissue_scores(X, tis):
    colmean = np.nanmean(X, axis=0)
    tissues = np.unique(tis)
    return pd.Series([np.nanmean(colmean[tis == t]) for t in tissues], index=tissues)


def _rank(score):
    return score.rank(ascending=False, method="first").astype(int)


def analyze():
    import step35_analysis as s35
    import step42_uncertainty as s42
    import step49_recommended_pipeline_check as s49
    from step36_benchmark_expected_tissue import EXPECTED

    M = _combined_matrices()
    dom = s35.domain_map(s35.read_variants())
    sets = {d: set(dom.loc[dom.domain == d, "id"]) for d in DOMAINS}
    added = pd.read_csv(os.path.join(OUT, "loci_resolved.csv"), keep_default_na=False)
    added = added[added.excluded == ""]
    met = {"published": sets["metabolic"], "with added loci": sets["metabolic"] | set(added.rsid)}
    rows = []

    def rec(analysis, version, **kw):
        rows.append({"analysis": analysis, "metabolic loci": version, "n": len(met[version]), **kw})

    # 1. raw pooled summary (mean-difference scorer, all assays, choroid plexus inside the eye)
    X, ids, tis = s42.select(M, "raw", s35.ORIG, published_eye=True, exclude=("other",))
    for v, idset in met.items():
        sc = _tissue_scores(X[np.isin(ids, list(idset))], tis).sort_values(ascending=False)
        rec("raw pooled summary", v, eye_rank=int(_rank(sc)["eye"]), top=sc.index[0],
            eye_over_next=sc["eye"] / sc.drop("eye").max())
    # 2. calibrated pooled summary, random null
    X, ids, tis = s42.select(M, "emp_rec", s35.REC, published_eye=False)
    for v, idset in met.items():
        sc = _tissue_scores(X[np.isin(ids, list(idset))], tis)
        rec("calibrated pooled (random null)", v, eye_rank=int(_rank(sc)["eye"]), top=_rank(sc).idxmin())

    # 3 + 4. DNase only: three calibrations x two summaries; absolute ranks and specificity
    calib = {"AlphaGenome |q|": s49.dnase_only(M["q_X"], M["q_ids"], M["q_cols"]),
             "null pct, random null": s49.dnase_only(M["emp_rec_X"], M["emp_rec_ids"], M["emp_rec_cols"]),
             "null pct, matched null": s49.dnase_only(*_matched_dnase())}
    for cname, (X, ids, tis) in calib.items():
        for metric in ["mean", "strong fraction"]:
            Y = X if metric == "mean" else np.where(np.isnan(X), np.nan, (X >= THR).astype(float))
            Y = np.nan_to_num(Y, nan=np.nanmean(Y))
            base = {d: _tissue_scores(Y[np.isin(ids, list(sets[d]))], tis) for d in ("eye", "kidney")}
            for v, idset in met.items():
                sc = {**base, "metabolic": _tissue_scores(Y[np.isin(ids, list(idset))], tis)}
                r = _rank(sc["metabolic"])
                rec(f"DNase {metric}, {cname}", v, eye_rank=int(r["eye"]), top=r.idxmin(),
                    islet_rank=int(r["pancreas_islet"]))
                spec = {d: sc[d] - np.mean([sc[e] for e in DOMAINS if e != d], axis=0) for d in DOMAINS}
                rs = {d: _rank(spec[d]) for d in DOMAINS}
                rec(f"specificity, DNase {metric}, {cname}", v, eye_rank=int(rs["metabolic"]["eye"]),
                    top=rs["metabolic"].idxmin(), islet_rank=int(rs["metabolic"]["pancreas_islet"]),
                    eye_loci_top=rs["eye"].idxmin(), kidney_rank_for_kidney_loci=int(rs["kidney"]["kidney"]))

    # 5. expected tissue per metabolic trait (DNase strong-effect fraction, AlphaGenome |q|, 12 tissues)
    X, ids, tis = s49.dnase_only(M["q_X"], M["q_ids"], M["q_cols"])
    S = np.where(np.isnan(X), np.nan, (X >= THR).astype(float))
    loci = pd.read_csv(s35.LOCI)[["rsid", "source_disease"]].drop_duplicates()
    extra = added[["rsid", "source_disease"]].drop_duplicates()
    for v, tab in {"published": loci, "with added loci": pd.concat([loci, extra]).drop_duplicates()}.items():
        for trait in METABOLIC_TRAITS:
            idset = set(tab.rsid[tab.source_disease == trait])
            sel = np.isin(ids, list(idset))
            r = _rank(_tissue_scores(S[sel], tis))
            rows.append({"analysis": f"expected tissue, {trait} -> {EXPECTED[trait]}", "metabolic loci": v,
                         "n": int(sel.sum()), "expected_rank": int(r[EXPECTED[trait]]), "top": r.idxmin()})

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(paths.DATA, "analysis", "step50_metabolic_batch.csv"), index=False)
    keys = ["eye_rank", "top", "islet_rank", "eye_loci_top", "kidney_rank_for_kidney_loci", "expected_rank"]
    same = []
    for a, g in t.groupby("analysis", sort=False):
        cols = [c for c in keys if c in g and g[c].notna().all()]
        same.append({"analysis": a, "unchanged": all(g[c].nunique() == 1 for c in cols)})
    same = pd.DataFrame(same)
    n_new_met = len(met["with added loci"]) - len(met["published"])
    already = len(set(added.rsid) & set(s35.read_variants().id))
    L = ["# Sensitivity to the metabolic loci missing from the published locus set", "",
         "Generated by `scripts/step50_metabolic_batch_sensitivity.py`. In June 2026 the first Ensembl batch of "
         "200 metabolic rsIDs (the 200 most significant metabolic loci) returned no answer, and 199 of them were "
         f"never analysed. Re-queried: {len(added)} resolve and pass the filters and are added to the metabolic loci "
         f"({len(met['published']):,} -> {len(met['with added loci']):,}; {n_new_met} new to the metabolic set). "
         f"{already} of them were already scored as eye or kidney loci; the other {len(added) - already} were scored "
         "for this check with the same scorers and calibrated against the same nulls. Ranks among 12 tissues "
         "(choroid plexus inside the eye for the raw pooled summary, as in the paper).", "",
         f"**Ranks and top tissues unchanged in {int(same.unchanged.sum())} of {len(same)} analyses.**", "",
         t.round(3).to_markdown(index=False), ""]
    if (~same.unchanged).any():
        L += ["Changed:", "", same[~same.unchanged].to_markdown(index=False), ""]
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(same.to_string(index=False))
    print(f"-> {REPORT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["prepare", "analyze"])
    a = ap.parse_args()
    if a.what == "prepare":
        prepare()
    else:
        analyze()
