"""
Tissue attribution for variant scores from sequence-to-function models (e.g. AlphaGenome), done safely.

The functions follow the checklist in Box 2 of the paper:

  1. assay_composition()      tabulate assay types and track counts per tissue before summarising
  2. tissue_scores()          compare tissues WITHIN one assay; pooling assays is refused unless forced
  3. null_calibrate()         put every track on a common scale using a null set of variants
     background_sensitivity() and check how rankings change with a second null background
  4. tissue_scores(metric="strong_fraction") and bootstrap_ranks()
                              summarise by the fraction of strong effects; report uncertainty over
                              variants and tracks, not ranks alone
  5. top_tracks()             inspect the individual tracks behind a "winning" tissue
  6. tissue_specificity()     compare each tissue with itself across trait sets, so that properties
                              lifting a tissue for every trait cancel

Input is a long table with one row per (variant, track):

    variant_id | track_name | assay | score

`score` is the absolute effect you want to compare: |raw_score| for any scorer, or |quantile_score|
for AlphaGenome's recommended scorers. `from_alphagenome()` builds this table from the output of
`alphagenome.models.variant_scorers.tidy_scores`. Track metadata is a table with `track_name` and
`tissue` (your own grouping of biosamples into tissues).

Only numpy and pandas are required.
"""
import warnings

import numpy as np
import pandas as pd

__all__ = ["from_alphagenome", "assay_composition", "null_calibrate", "tissue_scores",
           "bootstrap_ranks", "background_sensitivity", "top_tracks", "tissue_specificity"]

_ASSAY_ALIASES = {"DNASE": "DNase", "ATAC": "ATAC", "RNA_SEQ": "RNA-seq", "CAGE": "CAGE", "PROCAP": "PRO-cap",
                  "CHIP_HISTONE": "ChIP-histone", "CHIP_TF": "ChIP-TF"}


def _assay_name(x):
    s = str(x).upper().replace("OUTPUTTYPE.", "")
    return _ASSAY_ALIASES.get(s, str(x))


# ── input ─────────────────────────────────────────────────────────────────────
def from_alphagenome(tidy, use="quantile", gene_summary="max"):
    """Convert `variant_scorers.tidy_scores(...)` output into the long table used here.

    use          "quantile" -> score = |quantile_score| (recommended scorers only);
                 "raw"      -> score = |raw_score| (any scorer; calibrate with null_calibrate()).
    gene_summary gene-level scorers (e.g. RNA-seq GeneMaskLFC) give one row per gene and track;
                 "max" keeps the largest |score| per track. NOTE: a max over genes is not on the
                 quantile scale any more (it is inflated towards 1 for every variant), so it must be
                 re-calibrated with null_calibrate() before it is compared with anything.
    """
    col = {"quantile": "quantile_score", "raw": "raw_score"}[use]
    d = pd.DataFrame({
        "variant_id": tidy["variant_id"].astype(str),
        "track_name": tidy["track_name"].astype(str),
        "assay": tidy["output_type"].map(_assay_name),
        "score": tidy[col].abs().astype(float),
    })
    if d.score.isna().all():
        raise ValueError(f"No {col} values: this scorer has no quantile scores; use use='raw' and null_calibrate().")
    key = ["variant_id", "track_name", "assay"]
    if d.duplicated(key).any():
        if gene_summary != "max":
            raise ValueError("gene_summary must be 'max'")
        d = d.groupby(key, as_index=False).score.max()
    return d


# ── check 1 ───────────────────────────────────────────────────────────────────
def assay_composition(scores, tracks):
    """Tracks per tissue and assay, with each tissue's share of each assay.

    A tissue whose tracks are all of one assay (e.g. five DNase tracks) will look extreme whenever
    assays are pooled. Look at this table before summarising anything.
    """
    t = scores[["track_name", "assay"]].drop_duplicates().merge(tracks[["track_name", "tissue"]], on="track_name")
    tab = t.pivot_table(index="tissue", columns="assay", values="track_name", aggfunc="count", fill_value=0)
    tab["n_tracks"] = tab.sum(axis=1)
    for a in [c for c in tab.columns if c != "n_tracks"]:
        tab[f"share_{a}"] = (tab[a] / tab.n_tracks).round(3)
    return tab.sort_values("n_tracks", ascending=False)


# ── check 3 ───────────────────────────────────────────────────────────────────
def null_calibrate(scores, null_scores):
    """Replace each score by its percentile among null variants on the same track (column `null_pct`).

    null_scores: the same long table for a set of null variants (common variants with no association
    to the trait; ideally matched to the test variants on distance to genes). After calibration a
    typical null variant scores 0.5 on every track, whatever the assay or scorer.
    """
    out = scores.copy()
    out["null_pct"] = np.nan
    ref = {k: np.sort(g.score.values) for k, g in null_scores.groupby("track_name")}
    for k, idx in out.groupby("track_name").groups.items():
        r = ref.get(k)
        if r is None or len(r) == 0:
            continue
        v = out.loc[idx, "score"].values
        # mid-rank percentile: ties count half
        out.loc[idx, "null_pct"] = (np.searchsorted(r, v, "left") + np.searchsorted(r, v, "right")) / (2 * len(r))
    missing = out.null_pct.isna().mean()
    if missing > 0:
        warnings.warn(f"{missing:.1%} of rows have no null distribution for their track and stay NaN.")
    return out


# ── checks 2 and 4 ────────────────────────────────────────────────────────────
def _summaries(df, value, metric, threshold):
    if metric == "mean":
        return df.groupby(["assay", "tissue", "variant_id"])[value].mean()
    if metric == "strong_fraction":
        return (df[value] >= threshold).groupby([df.assay, df.tissue, df.variant_id]).mean()
    if metric == "max":
        raise ValueError("metric='max' is not offered: the maximum over a tissue's tracks grows with the number "
                         "of tracks, so it ranks tissues by track count (in our data Spearman 0.84-0.93 with "
                         "track number). Use 'strong_fraction' (share of tracks above a threshold) instead.")
    raise ValueError("metric must be 'mean' or 'strong_fraction'")


def tissue_scores(scores, tracks, value="null_pct", metric="strong_fraction", threshold=0.95,
                  pool_assays=False, exclude=("other",)):
    """Per-assay tissue summary, ranked within each assay.

    value      "null_pct" (after null_calibrate) or "score" (e.g. AlphaGenome |quantile_score|).
               Raw scores are refused here: they are on different scales in different tracks.
    metric     "strong_fraction": share of tracks with value >= threshold (recommended; tissue
               differences are carried by strong effects, the mean is diluted by near-null tracks)
               "mean": mean value.
    pool_assays Pooling assays re-creates a composition bias even after calibration, because test
               variants usually depart from the null by different amounts in different assays.
               It is refused unless set to True, and then a warning is raised.

    Returns one row per (assay, tissue): summary, rank within assay, n_tracks, n_variants.
    """
    if value == "score" and "null_pct" not in scores and scores.score.max() > 1:
        raise ValueError("value='score' looks like raw scores (values > 1). Calibrate first with null_calibrate().")
    d = scores.merge(tracks[["track_name", "tissue"]], on="track_name")
    d = d[~d.tissue.isin(exclude)]
    if pool_assays:
        warnings.warn("Pooling assays: tissue ranks will depend on each tissue's assay mix. "
                      "Use only for comparison with the within-assay result.")
        d = d.assign(assay="ALL (pooled)")
    per_variant = _summaries(d, value, metric, threshold)
    out = per_variant.groupby(["assay", "tissue"]).agg(summary="mean", n_variants="size").reset_index()
    out = out.merge(d.groupby(["assay", "tissue"]).track_name.nunique().rename("n_tracks").reset_index())
    out["rank"] = out.groupby("assay").summary.rank(ascending=False, method="min").astype(int)
    return out.sort_values(["assay", "rank"]).reset_index(drop=True)


def bootstrap_ranks(scores, tracks, assay, n_boot=1000, over=("variants", "tracks"), seed=0, **kw):
    """Resample variants and/or tracks (within each tissue) and report how stable each tissue's rank is.

    Returns per tissue: observed rank, P(rank 1), 2.5% and 97.5% rank quantiles. A tissue that ranks
    first in, say, 60% of resamples is not "the" tissue; report the uncertainty.
    """
    rng = np.random.default_rng(seed)
    d = scores[scores.assay == assay].merge(tracks[["track_name", "tissue"]], on="track_name")
    variants = d.variant_id.unique()
    by_tissue = d.groupby("tissue").track_name.unique()
    obs = tissue_scores(scores[scores.assay == assay], tracks, **kw).set_index("tissue")["rank"]
    ranks = []
    for _ in range(n_boot):
        b = d
        if "variants" in over:
            pick = pd.Series(rng.choice(variants, len(variants), replace=True), name="variant_id")
            b = pick.to_frame().merge(b, on="variant_id")
        if "tracks" in over:
            keep = pd.Series([t for tis in by_tissue for t in rng.choice(tis, len(tis), replace=True)],
                             name="track_name")
            b = keep.to_frame().merge(b, on="track_name")
        r = tissue_scores(b.drop(columns="tissue"), tracks, **kw).set_index("tissue")["rank"]
        ranks.append(r)
    R = pd.concat(ranks, axis=1)
    return pd.DataFrame({"observed_rank": obs, "p_rank1": (R == 1).mean(axis=1),
                         "rank_q025": R.quantile(0.025, axis=1), "rank_q975": R.quantile(0.975, axis=1)}
                        ).sort_values("observed_rank")


def background_sensitivity(scores, tracks, null_a, null_b, labels=("null A", "null B"), **kw):
    """Tissue ranks within each assay under two null backgrounds (e.g. random vs gene-distance-matched).

    If ranks move between backgrounds, the tissue differences are too small to interpret on their own.
    """
    ra = tissue_scores(null_calibrate(scores, null_a), tracks, value="null_pct", **kw)
    rb = tissue_scores(null_calibrate(scores, null_b), tracks, value="null_pct", **kw)
    m = ra.merge(rb, on=["assay", "tissue"], suffixes=(f" [{labels[0]}]", f" [{labels[1]}]"))
    m["rank_change"] = m[f"rank [{labels[1]}]"] - m[f"rank [{labels[0]}]"]
    return m


# ── check 6 ───────────────────────────────────────────────────────────────────
def tissue_specificity(scores, tracks, trait_sets, assay, value="null_pct", metric="strong_fraction",
                       threshold=0.95, n_boot=0, seed=0, exclude=("other",)):
    """Compare each tissue with itself across trait sets, within one assay.

    trait_sets  table with `variant_id` and `trait` (at least two traits; three or more is better).
    For each trait, a tissue's specificity is its score for that trait minus its mean score for the
    other traits. Properties that lift a tissue for every set of GWAS loci (for example tracks that
    respond little to gene proximity, so that they stand out against a proximity-matched null) cancel;
    what remains is the tissue's response to this trait's loci in particular.

    n_boot > 0 adds a bootstrap over variants within each trait and tracks within each tissue:
    95% interval of the specificity and P(rank 1).
    Returns one row per (trait, tissue): specificity, rank within trait, n_tracks, [ci_lo, ci_hi, p_rank1].
    """
    if metric == "max":
        _summaries(None, value, "max", threshold)  # raises with the explanation
    d = scores[scores.assay == assay].merge(tracks[["track_name", "tissue"]], on="track_name")
    d = d[~d.tissue.isin(exclude)]
    if d.empty:
        raise ValueError(f"No scores for assay {assay!r}.")
    x = (d[value] >= threshold).astype(float) if metric == "strong_fraction" else d[value].astype(float)
    M = d.assign(x=x).pivot_table(index="variant_id", columns="track_name", values="x", aggfunc="mean")
    ts = trait_sets[trait_sets.variant_id.isin(M.index)]
    traits = sorted(ts.trait.unique())
    if len(traits) < 2:
        raise ValueError("Need at least two trait sets to compare a tissue with itself.")
    tissue_of = d.drop_duplicates("track_name").set_index("track_name").tissue.reindex(M.columns)
    tissues = sorted(tissue_of.unique())
    groups = [np.where(tissue_of.values == t)[0] for t in tissues]
    rows = {tr: M.index.get_indexer(ts.loc[ts.trait == tr, "variant_id"].unique()) for tr in traits}
    X = np.nan_to_num(M.values, nan=np.nanmean(M.values))

    def spec(colmeans, pick):
        sc = np.array([[colmeans[tr][g].mean() for g in pick] for tr in traits])  # traits x tissues
        others = (sc.sum(0, keepdims=True) - sc) / (len(traits) - 1)
        return sc - others

    obs = spec({tr: X[r].mean(0) for tr, r in rows.items()}, groups)
    out = pd.DataFrame([{"trait": tr, "tissue": t, "specificity": obs[i, j], "n_tracks": len(groups[j])}
                        for i, tr in enumerate(traits) for j, t in enumerate(tissues)])
    out["rank"] = out.groupby("trait").specificity.rank(ascending=False, method="min").astype(int)
    if n_boot:
        rng = np.random.default_rng(seed)
        B = np.empty((n_boot, len(traits), len(tissues)))
        for b in range(n_boot):
            cm = {tr: X[rng.choice(r, len(r))].mean(0) for tr, r in rows.items()}
            B[b] = spec(cm, [rng.choice(g, len(g)) for g in groups])
        top = (B == B.max(axis=2, keepdims=True))
        out["ci_lo"] = np.percentile(B, 2.5, axis=0).ravel()
        out["ci_hi"] = np.percentile(B, 97.5, axis=0).ravel()
        out["p_rank1"] = top.mean(0).ravel()
    return out.sort_values(["trait", "rank"]).reset_index(drop=True)


# ── check 5 ───────────────────────────────────────────────────────────────────
def top_tracks(scores, tracks, tissue, value="null_pct", n=10):
    """The tracks behind one tissue's score, strongest first, with their metadata.

    Check that every track really belongs to the tissue (we found a brain-ventricle cell type filed
    under "eye") and whether one or two tracks carry the whole tissue.
    """
    meta = tracks[["track_name"] + [c for c in tracks.columns if c not in scores.columns]]
    d = scores.merge(meta, on="track_name")
    d = d[d.tissue == tissue]
    keys = ["track_name", "assay"] + [c for c in meta.columns if c != "track_name"]
    per = d.groupby(keys)[value].agg(["mean", "median", "size"]).reset_index()
    return per.sort_values("mean", ascending=False).head(n)
