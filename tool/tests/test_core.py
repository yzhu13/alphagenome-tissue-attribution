"""
Synthetic tests for tissue_attribution. Run:  python -m pytest tool/tests

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
import pytest

try:
    import tissue_attribution as ta
except ImportError:  # running from a checkout without installing the package
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import tissue_attribution as ta  # noqa: E402

SCALE = {"DNase": 1.0, "RNA-seq": 0.2}


def make_tracks():
    rows = []
    for tissue, n_dnase, n_rna in [("dnase_only", 5, 0), ("true_hit", 5, 20), ("background", 5, 20)]:
        rows += [(f"{tissue}_dnase{i}", tissue, "DNase") for i in range(n_dnase)]
        rows += [(f"{tissue}_rna{i}", tissue, "RNA-seq") for i in range(n_rna)]
    return pd.DataFrame(rows, columns=["track_name", "tissue", "assay"])


def simulate(rng, tracks, n_variants, signal=0.0, prefix="v"):
    rows = []
    for v in range(n_variants):
        for t in tracks.itertuples():
            s = abs(rng.normal(0, SCALE[t.assay]))
            if signal and t.tissue == "true_hit" and t.assay == "DNase" and rng.random() < 0.3:
                s += signal
            rows.append((f"{prefix}{v}", t.track_name, t.assay, s))
    return pd.DataFrame(rows, columns=["variant_id", "track_name", "assay", "score"])


@pytest.fixture(scope="module")
def study():
    """Test and null variants on the three-tissue design above."""
    rng = np.random.default_rng(1)
    tracks = make_tracks()
    test = simulate(rng, tracks, 300, signal=2.0, prefix="gwas")
    null = simulate(rng, tracks, 600, prefix="null")
    null_b = simulate(rng, tracks, 600, prefix="nullB")
    meta = tracks[["track_name", "tissue"]]
    return dict(tracks=tracks, meta=meta, test=test, null=null, null_b=null_b,
                cal=ta.null_calibrate(test, null))


def test_raw_pooled_scores_reproduce_the_artifact(study):
    raw = study["test"].merge(study["meta"]).groupby("tissue").score.mean().sort_values(ascending=False)
    assert raw.index[0] == "dnase_only", raw


def test_assay_composition_flags_single_assay_tissue(study):
    comp = ta.assay_composition(study["test"], study["meta"])
    assert comp.loc["dnase_only", "share_DNase"] == 1.0


def test_null_calibration_centres_null_variants(study):
    cal_null = ta.null_calibrate(study["null"], study["null"])
    means = cal_null.groupby("assay").null_pct.mean()
    assert all(abs(m - 0.5) < 0.02 for m in means), means


def test_within_assay_strong_fraction_finds_true_tissue(study):
    res = ta.tissue_scores(study["cal"], study["meta"], value="null_pct", metric="strong_fraction")
    dn = res[res.assay == "DNase"].set_index("tissue")
    assert dn["rank"].idxmin() == "true_hit", dn
    assert set(res.assay) == {"DNase", "RNA-seq"}  # never pooled by default


def test_pooling_assays_warns(study):
    with pytest.warns(UserWarning, match="Pooling assays"):
        ta.tissue_scores(study["cal"], study["meta"], value="null_pct", pool_assays=True)


def test_max_metric_is_refused(study):
    with pytest.raises(ValueError, match="track count"):
        ta.tissue_scores(study["cal"], study["meta"], value="null_pct", metric="max")


def test_raw_scores_are_refused(study):
    test = study["test"]
    with pytest.raises(ValueError):
        ta.tissue_scores(test.assign(score=test.score * 10), study["meta"], value="score")


def test_bootstrap_and_background_sensitivity(study):
    b = ta.bootstrap_ranks(study["cal"], study["meta"], assay="DNase", n_boot=30, value="null_pct")
    assert b.p_rank1.idxmax() == "true_hit", b
    sens = ta.background_sensitivity(study["test"], study["meta"], study["null"], study["null_b"])
    assert "rank_change" in sens


def test_top_tracks_returns_signal_assay_first(study):
    top = ta.top_tracks(study["cal"], study["tracks"], "true_hit")
    assert (top.assay == "DNase").iloc[0]


def test_alphagenome_adapter():
    tidy = pd.DataFrame({"variant_id": ["x"] * 3, "track_name": ["t1", "t1", "t2"],
                         "output_type": ["RNA_SEQ", "RNA_SEQ", "DNASE"],
                         "raw_score": [0.1, -0.3, 2.0], "quantile_score": [0.2, -0.9, 0.99]})
    a = ta.from_alphagenome(tidy)  # gene-level rows reduced to the max per track
    assert len(a) == 2 and a.loc[a.track_name == "t1", "score"].item() == 0.9
    with pytest.raises(ValueError):
        ta.from_alphagenome(tidy.assign(quantile_score=np.nan))


@pytest.fixture(scope="module")
def lifted_design():
    """A tissue lifted for EVERY trait set, and one specific to trait A."""
    rng = np.random.default_rng(2)
    tracks = pd.DataFrame([(f"{t}_d{i}", t, "DNase") for t in ["lifted", "specific", "plain"] for i in range(6)],
                          columns=["track_name", "tissue", "assay"])
    rows = []
    for trait in ["A", "B", "C"]:
        for v in range(150):
            for t in tracks.itertuples():
                p = 0.5 + 0.3 * (t.tissue == "lifted") + 0.25 * (t.tissue == "specific" and trait == "A")
                rows.append((f"{trait}{v}", t.track_name, "DNase", min(rng.uniform(0, 1) + p - 0.5, 1.0)))
    scores = pd.DataFrame(rows, columns=["variant_id", "track_name", "assay", "null_pct"])
    sets = pd.DataFrame({"variant_id": scores.variant_id.unique()}).assign(trait=lambda x: x.variant_id.str[0])
    return tracks, scores, sets


def test_lifted_tissue_wins_on_absolute_scores(lifted_design):
    tracks, scores, _ = lifted_design
    absolute = ta.tissue_scores(scores[scores.variant_id.str.startswith("A")], tracks, metric="mean")
    assert absolute.set_index("tissue")["rank"].idxmin() == "lifted", absolute


def test_specificity_removes_tissue_wide_lift(lifted_design):
    tracks, scores, sets = lifted_design
    spec = ta.tissue_specificity(scores, tracks, sets, assay="DNase", metric="mean", n_boot=50)
    a = spec[spec.trait == "A"].set_index("tissue")
    assert a["rank"].idxmin() == "specific" and a.loc["specific", "ci_lo"] > 0, a
    assert abs(a.loc["lifted", "specificity"]) < 0.03, a


def test_specificity_refuses_max(lifted_design):
    tracks, scores, sets = lifted_design
    with pytest.raises(ValueError):
        ta.tissue_specificity(scores, tracks, sets, assay="DNase", metric="max")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
