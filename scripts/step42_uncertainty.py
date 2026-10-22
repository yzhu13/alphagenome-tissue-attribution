"""
step42_uncertainty.py

Uncertainty of the rankings: the paper asks readers to report uncertainty over variants
AND tracks, so its own rankings must carry it.

1. Joint bootstrap (resample loci, and resample tracks within each tissue) for:
   - Fig. 4a: the eye's rank under (i) raw pooled (original pipeline: eye = 5 DNase incl. choroid plexus),
     (ii) calibrated pooled (null-percentile, recommended scorers), (iii) calibrated DNase |q|,
     (iv) calibrated DNase strong-effect fraction (|q| >= 0.95), per disease domain;
   - the benchmark: rank of the independently expected tissue per trait (step36 traits).
2. The strong-effect metric at three thresholds (|q| >= 0.90 / 0.95 / 0.99), with a paired Wilcoxon
   test and a bootstrap CI for the mean rank difference vs the raw pipeline across the 9 traits.

Ranked set: 12 tissues (`other` and, for calibrated methods, the misfiled choroid plexus track excluded).
Tissue score = mean over the tissue's tracks of the track's mean over loci (identical to the per-locus
tissue mean averaged over loci, because every locus has every track), so a bootstrap only needs a
locus-weight vector and a track-resampling per tissue.

First run builds a cache (data/uncertainty/*.npz, a few minutes); later runs are fast.
Writes reports/uncertainty.md and data/uncertainty/*.csv.

    python scripts/step42_uncertainty.py [--n-boot 500]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step35_analysis as s35  # noqa: E402
from step36_benchmark_expected_tissue import EXPECTED, EXPECTED_SECONDARY  # noqa: E402

OUT = os.path.join(s35.NEW, "data", "uncertainty")
REPORT = os.path.join(s35.NEW, "reports", "uncertainty.md")
SEED = 20260922


# ── cache: locus x track matrices ────────────────────────────────────────────
def build_matrices(run_dir=s35.RUN, tag="main", null_grid_parts=None):
    """Matrices (variants x tracks) for: raw |score| of original scorers, |q| and null-percentile of
    recommended scorers, null-percentile of original scorers. Columns named 'scorer|track'."""
    f = os.path.join(OUT, f"matrices_{tag}.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    parts = sorted(glob.glob(os.path.join(run_dir, "parts", "*.parquet")))
    main_parts = sorted(glob.glob(os.path.join(s35.RUN, "parts", "*.parquet")))
    v = s35.read_variants()
    index, G = s35.null_grids(null_grid_parts or main_parts, set(v.id[v.set == "null"]))
    rows = {"raw": [], "q": [], "emp_rec": [], "emp_orig": []}
    for p in parts:
        d = pd.read_parquet(p)
        d["emp"] = s35.empirical(d, index, G)
        # RNA-seq tracks come as +/- strand pairs with the same track_name; number the copies so both
        # strands are kept as separate columns (the tidy output order is the same for every variant)
        k = d.groupby(["id", "scorer", "track_name"]).cumcount().astype(str)
        d["col"] = d.scorer + "|" + d.track_name + "|" + k
        o, r = d[d.scorer.isin(s35.ORIG)], d[d.scorer.isin(s35.REC)]
        rows["raw"].append(o.pivot_table(index="id", columns="col", values="raw_score", aggfunc="first").abs())
        rows["emp_orig"].append(o.pivot_table(index="id", columns="col", values="emp", aggfunc="first"))
        rows["q"].append(r.pivot_table(index="id", columns="col", values="quantile_score", aggfunc="first").abs())
        rows["emp_rec"].append(r.pivot_table(index="id", columns="col", values="emp", aggfunc="first"))
    out = {}
    for k, lst in rows.items():
        m = pd.concat(lst)
        out[f"{k}_X"] = m.values.astype("float32")
        out[f"{k}_ids"] = m.index.values.astype(str)
        out[f"{k}_cols"] = m.columns.values.astype(str)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f, **out)
    return out


def col_tissue(cols, published_eye):
    tissue = s35.track_map()
    t = pd.Series([tissue.get(c.split("|")[1], "other") for c in cols])
    if published_eye:
        t = t.replace({"eye_misfiled": "eye"})
    return t.values


def select(M, key, scorers, published_eye, exclude=("other", "eye_misfiled")):
    X, ids, cols = M[f"{key}_X"], M[f"{key}_ids"], M[f"{key}_cols"]
    keep = np.array([c.split("|")[0] in scorers for c in cols])
    X, cols = X[:, keep], cols[keep]
    tis = col_tissue(cols, published_eye)
    ok = ~np.isin(tis, exclude)
    return X[:, ok], ids, tis[ok]


# ── bootstrap ────────────────────────────────────────────────────────────────
def boot_ranks(X, tis, n_boot, rng, transform=None, over=("loci", "tracks")):
    """Observed tissue ranks and bootstrap rank matrix (n_boot x n_tissues)."""
    if transform is not None:
        X = transform(X)
    X = np.nan_to_num(X, nan=np.nanmean(X))
    tissues = np.unique(tis)
    groups = [np.where(tis == t)[0] for t in tissues]

    def score(colmean, pick):
        return np.array([colmean[g].mean() for g in pick])

    def ranks(s):
        return (-s).argsort().argsort() + 1

    colmean = X.mean(axis=0)
    obs = ranks(score(colmean, groups))
    R = np.empty((n_boot, len(tissues)), dtype=int)
    n = X.shape[0]
    for b in range(n_boot):
        if "loci" in over:
            w = rng.multinomial(n, np.full(n, 1 / n)).astype("float32")
            cm = w @ X / n
        else:
            cm = colmean
        pick = [rng.choice(g, len(g), replace=True) for g in groups] if "tracks" in over else groups
        R[b] = ranks(score(cm, pick))
    return tissues, obs, R


def summarise(tissues, obs, R):
    return pd.DataFrame({"tissue": tissues, "rank": obs, "p_rank1": (R == 1).mean(0),
                         "rank_lo": np.percentile(R, 2.5, axis=0), "rank_hi": np.percentile(R, 97.5, axis=0)}
                        ).sort_values("rank")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    B = ap.parse_args().n_boot
    rng = np.random.default_rng(SEED)
    M = build_matrices()
    atac = pd.read_csv(s35.LOCI)
    dom_ids = {d: set(atac.loc[atac.disease_class == d, "rsid"]) for d in ["eye", "kidney", "metabolic"]}
    traits = atac[["rsid", "source_disease"]].drop_duplicates()

    methods = {  # label: (matrix key, scorers, published eye?, transform)
        "raw, pooled (original pipeline)": ("raw", s35.ORIG, True, None),
        "calibrated, pooled (null pct)": ("emp_rec", s35.REC, False, None),
        "calibrated DNase, mean |q|": ("q", ["dnase_rec"], False, None),
        "calibrated DNase, strong fraction (|q|>=0.95)": ("q", ["dnase_rec"], False, lambda X: (X >= 0.95).astype("float32")),
    }
    L = ["# Uncertainty of the paper's rankings", "",
         f"Generated by `scripts/step42_uncertainty.py`. Joint bootstrap: {B} resamples of loci AND of tracks within "
         "each tissue. 12 ranked tissues (`other` excluded; calibrated methods also exclude the misfiled choroid "
         "plexus track; the raw method keeps it inside eye, as in the original pipeline). `p_rank1` = share of resamples in which "
         "the tissue ranks first; rank interval = 2.5–97.5 percentiles.", ""]

    # 1a. Fig. 4a ranks with uncertainty
    rows, full = [], []
    for d, ids in dom_ids.items():
        for lab, (key, sc, pub, tf) in methods.items():
            X, rid, tis = select(M, key, sc, pub)
            X = X[np.isin(rid, list(ids))]
            t, obs, R = boot_ranks(X, tis, B, rng, tf)
            s = summarise(t, obs, R)
            e = s.set_index("tissue").loc["eye"]
            top = s.iloc[0]
            rows.append({"domain": d, "method": lab, "eye rank": int(e["rank"]),
                         "eye 95% interval": f"{e.rank_lo:.0f}–{e.rank_hi:.0f}", "P(eye #1)": e.p_rank1,
                         "top tissue": top.tissue, "P(top #1)": top.p_rank1})
            full.append(s.assign(domain=d, method=lab))
    fig4 = pd.DataFrame(rows)
    fig4.to_csv(os.path.join(OUT, "fig4a_rank_uncertainty.csv"), index=False)
    pd.concat(full).to_csv(os.path.join(OUT, "fig4a_all_tissues.csv"), index=False)
    L += ["## 1a. Eye rank under each comparison (Fig. 4a), with uncertainty", "",
          fig4.round(3).to_markdown(index=False), ""]

    # 1b + 2. expected-tissue rank per trait, with uncertainty; thresholds
    bench = {"raw, pooled (original pipeline)": methods["raw, pooled (original pipeline)"],
             "calibrated DNase, mean |q|": methods["calibrated DNase, mean |q|"]}
    for thr in (0.90, 0.95, 0.99):
        bench[f"strong fraction |q|>={thr:.2f}"] = ("q", ["dnase_rec"], False,
                                                     (lambda th: (lambda X: (X >= th).astype("float32")))(thr))
    brow, bootranks = [], {}
    for trait, exp in {**EXPECTED, **EXPECTED_SECONDARY}.items():
        ids = set(traits.loc[traits.source_disease == trait, "rsid"])
        for lab, (key, sc, pub, tf) in bench.items():
            X, rid, tis = select(M, key, sc, pub)
            X = X[np.isin(rid, list(ids))]
            t, obs, R = boot_ranks(X, tis, B, rng, tf)
            j = list(t).index(exp)
            bootranks[(trait, lab)] = R[:, j]
            brow.append({"trait": trait, "expected": exp, "set": "primary" if trait in EXPECTED else "secondary",
                         "method": lab, "rank": int(obs[j]),
                         "interval": f"{np.percentile(R[:, j], 2.5):.0f}–{np.percentile(R[:, j], 97.5):.0f}",
                         "P(expected #1)": (R[:, j] == 1).mean()})
    bt = pd.DataFrame(brow)
    bt.to_csv(os.path.join(OUT, "benchmark_rank_uncertainty.csv"), index=False)
    for st in ["primary", "secondary"]:
        sub = bt[bt.set == st]
        L += [f"## 1b. Rank of the expected tissue ({st} traits), with 95% interval", "",
              sub.assign(cell=sub["rank"].astype(str) + " (" + sub.interval + ")")
                 .pivot(index=["trait", "expected"], columns="method", values="cell")[list(bench)].to_markdown(), ""]

    # 2. paired comparison vs raw across the 9 primary traits
    from scipy.stats import wilcoxon
    prim = list(EXPECTED)
    raw_lab = "raw, pooled (original pipeline)"
    L += ["## 2. Each method vs the raw pipeline across the 9 primary traits", "",
          "Mean rank of the expected tissue; difference = method − raw (negative = better). Wilcoxon signed-rank on "
          "the 9 observed rank pairs (zero differences dropped); bootstrap CI of the mean difference from the "
          "per-trait bootstrap ranks.", ""]
    rows = []
    for lab in bench:
        if lab == raw_lab:
            continue
        a = np.array([bt[(bt.trait == t) & (bt.method == raw_lab)]["rank"].item() for t in prim])
        b = np.array([bt[(bt.trait == t) & (bt.method == lab)]["rank"].item() for t in prim])
        diff_boot = np.mean([bootranks[(t, lab)] - bootranks[(t, raw_lab)] for t in prim], axis=0)
        try:
            p = wilcoxon(b, a).pvalue
        except ValueError:
            p = np.nan
        rows.append({"method": lab, "mean rank": b.mean(), "raw mean rank": a.mean(), "mean difference": (b - a).mean(),
                     "95% CI": f"{np.percentile(diff_boot, 2.5):.2f} to {np.percentile(diff_boot, 97.5):.2f}",
                     "better / same / worse (traits)": f"{(b < a).sum()} / {(b == a).sum()} / {(b > a).sum()}",
                     "Wilcoxon p": p})
    L += [pd.DataFrame(rows).round(3).to_markdown(index=False), ""]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "benchmark_paired_vs_raw.csv"), index=False)

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
