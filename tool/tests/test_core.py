"""
Synthetic tests for tissue_attribution. Run:  python tool/tests/test_core.py   (or pytest)

The simulation reproduces the paper's failure mode on purpose:
  tissue "dnase_only"  5 DNase tracks, no real signal
  tissue "true_hit"    5 DNase + 20 RNA tracks, real signal in its DNase tracks
  tissue "background"  5 DNase + 20 RNA tracks, no signal
DNase raw scores are ~5x larger than RNA raw scores, as in AlphaGenome.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tissue_attribution as ta  # noqa: E402

RNG = np.random.default_rng(1)
SCALE = {"DNase": 1.0, "RNA-seq": 0.2}


def make_tracks():
    rows = []
    for tissue, n_dnase, n_rna in [("dnase_only", 5, 0), ("true_hit", 5, 20), ("background", 5, 20)]:
        rows += [(f"{tissue}_dnase{i}", tissue, "DNase") for i in range(n_dnase)]
        rows += [(f"{tissue}_rna{i}", tissue, "RNA-seq") for i in range(n_rna)]
    return pd.DataFrame(rows, columns=["track_name", "tissue", "assay"])


def simulate(tracks, n_variants, signal=0.0, prefix="v"):
    rows = []
    for v in range(n_variants):
        for t in tracks.itertuples():
            s = abs(RNG.normal(0, SCALE[t.assay]))
            if signal and t.tissue == "true_hit" and t.assay == "DNase" and RNG.random() < 0.3:
                s += signal
            rows.append((f"{prefix}{v}", t.track_name, t.assay, s))
    return pd.DataFrame(rows, columns=["variant_id", "track_name", "assay", "score"])


def test_all():
    tracks = make_tracks()
    meta = tracks[["track_name", "tissue"]]
    test = simulate(tracks, 300, signal=2.0, prefix="gwas")
    null = simulate(tracks, 600, prefix="null")

    # the artifact: raw scores pooled across assays put the DNase-only tissue first
    raw = test.merge(meta).groupby("tissue").score.mean().sort_values(ascending=False)
    assert raw.index[0] == "dnase_only", raw

    # check 1: composition table flags the DNase-only tissue
    comp = ta.assay_composition(test, meta)
    assert comp.loc["dnase_only", "share_DNase"] == 1.0

    # check 3: calibration puts null variants at ~0.5 on every assay
    cal_null = ta.null_calibrate(null, null)
    assert all(abs(m - 0.5) < 0.02 for m in cal_null.groupby("assay").null_pct.mean()), \
        cal_null.groupby("assay").null_pct.mean()

    # checks 2 + 4: within-assay, calibrated, strong-effect fraction finds the true tissue
    cal = ta.null_calibrate(test, null)
    res = ta.tissue_scores(cal, meta, value="null_pct", metric="strong_fraction")
    dn = res[res.assay == "DNase"].set_index("tissue")
    assert dn["rank"].idxmin() == "true_hit", dn
    assert set(res.assay) == {"DNase", "RNA-seq"}  # never pooled by default

    # pooling assays is allowed only with a warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ta.tissue_scores(cal, meta, value="null_pct", pool_assays=True)
        assert any("Pooling assays" in str(x.message) for x in w)

    # the max over a tissue's tracks is refused (it ranks tissues by track count)
    try:
        ta.tissue_scores(cal, meta, value="null_pct", metric="max")
        raise AssertionError("metric='max' should be refused")
    except ValueError as e:
        assert "track count" in str(e)

    # raw scores are refused as `value="score"`
    try:
        ta.tissue_scores(test.assign(score=test.score * 10), meta, value="score")
        raise AssertionError("raw scores should be refused")
    except ValueError:
        pass

    # uncertainty and background sensitivity run and agree on the true tissue
    b = ta.bootstrap_ranks(cal, meta, assay="DNase", n_boot=30, value="null_pct")
    assert b.p_rank1.idxmax() == "true_hit", b
    sens = ta.background_sensitivity(test, meta, null, simulate(tracks, 600, prefix="nullB"))
    assert "rank_change" in sens

    # check 5
    top = ta.top_tracks(cal, tracks, "true_hit")
    assert (top.assay == "DNase").iloc[0]

    # AlphaGenome adapter: gene-level rows reduced to the max per track; missing quantiles caught
    tidy = pd.DataFrame({"variant_id": ["x"] * 3, "track_name": ["t1", "t1", "t2"],
                         "output_type": ["RNA_SEQ", "RNA_SEQ", "DNASE"],
                         "raw_score": [0.1, -0.3, 2.0], "quantile_score": [0.2, -0.9, 0.99]})
    a = ta.from_alphagenome(tidy)
    assert len(a) == 2 and a.loc[a.track_name == "t1", "score"].item() == 0.9
    try:
        ta.from_alphagenome(tidy.assign(quantile_score=np.nan))
        raise AssertionError("missing quantiles should raise")
    except ValueError:
        pass
    # check 6: a tissue lifted for EVERY trait set wins on absolute scores but not on specificity
    tr2 = pd.DataFrame([(f"{t}_d{i}", t, "DNase") for t in ["lifted", "specific", "plain"] for i in range(6)],
                       columns=["track_name", "tissue", "assay"])
    rows = []
    for trait in ["A", "B", "C"]:
        for v in range(150):
            for t in tr2.itertuples():
                p = 0.5 + 0.3 * (t.tissue == "lifted") + 0.25 * (t.tissue == "specific" and trait == "A")
                rows.append((f"{trait}{v}", t.track_name, "DNase", min(RNG.uniform(0, 1) + p - 0.5, 1.0)))
    sc = pd.DataFrame(rows, columns=["variant_id", "track_name", "assay", "null_pct"])
    sets = pd.DataFrame({"variant_id": sc.variant_id.unique()}).assign(trait=lambda x: x.variant_id.str[0])
    absolute = ta.tissue_scores(sc[sc.variant_id.str.startswith("A")], tr2, metric="mean")
    assert absolute.set_index("tissue")["rank"].idxmin() == "lifted", absolute
    spec = ta.tissue_specificity(sc, tr2, sets, assay="DNase", metric="mean", n_boot=50)
    a = spec[spec.trait == "A"].set_index("tissue")
    assert a["rank"].idxmin() == "specific" and a.loc["specific", "ci_lo"] > 0, a
    assert abs(a.loc["lifted", "specificity"]) < 0.03, a
    try:
        ta.tissue_specificity(sc, tr2, sets, assay="DNase", metric="max")
        raise AssertionError("metric='max' should be refused")
    except ValueError:
        pass
    print("all tests passed")


if __name__ == "__main__":
    test_all()
