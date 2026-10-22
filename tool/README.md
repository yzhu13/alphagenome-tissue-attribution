# tissue_attribution

Safe tissue attribution for variant-effect scores from sequence-to-function models such as
AlphaGenome. It implements the six checks in Box 2 of *When every disease points to the eye: assay-composition artifacts in AlphaGenome-based tissue attribution* (Zhu et al., in preparation).

**Why.** Averaging raw scores over each tissue's tracks produced a false "eye hub" in our own study:
the eye ranked first for eye, kidney, metabolic and immune disease loci, and for random variants,
because its tracks were all DNase-seq, whose raw scores are several times larger than RNA-seq scores.
Calibrating scores fixes their scale but not this composition problem, because disease variants
depart from the null by different amounts in different assays. This package makes the safe path the
default: calibrate, compare within one assay, summarise by strong effects, report uncertainty.

## Install

```bash
pip install "git+https://github.com/yzhu13/alphagenome-tissue-attribution#subdirectory=tool"
```

Only `numpy` and `pandas` are needed, so you can also copy the `tissue_attribution/` folder next to your code.
Run the tests with `pip install "./tool[test]"` and `python -m pytest tool/tests`.

## Input

One row per (variant, track):

| variant_id | track_name | assay | score |
|---|---|---|---|

`score` is an absolute effect: `|quantile_score|` from AlphaGenome's recommended scorers, or
`|raw_score|` from any scorer (then calibrate it). Track metadata is a table with `track_name` and
`tissue`, your own grouping of biosamples.

## Use

```python
import tissue_attribution as ta
from alphagenome.models import variant_scorers as vs

scores = ta.from_alphagenome(vs.tidy_scores(results_for_test_variants))   # |quantile_score|
null   = ta.from_alphagenome(vs.tidy_scores(results_for_null_variants))   # same scorers, null variants

ta.assay_composition(scores, tracks)                    # 1. look at assay mix per tissue first
cal = ta.null_calibrate(scores, null)                   # 3. percentile vs null, per track
res = ta.tissue_scores(cal, tracks, value="null_pct",   # 2 + 4. within-assay ranking,
                       metric="strong_fraction")        #        fraction of strong effects
ta.bootstrap_ranks(cal, tracks, assay="DNase",          # 4. uncertainty over variants AND tracks
                   value="null_pct", n_boot=200)
ta.background_sensitivity(scores, tracks, null, null_matched)  # 3. does a second null change ranks?
ta.top_tracks(cal, tracks, tissue=res.iloc[0].tissue)   # 5. which tracks carry the winner?
ta.tissue_specificity(cal, tracks, trait_sets,          # 6. each tissue vs itself across traits
                      assay="DNase", n_boot=200)        #    (trait_sets: variant_id | trait)
```

`tissue_scores(..., pool_assays=True)` exists only to show the pooled result next to the
within-assay one; it raises a warning. Raw scores are refused by `tissue_scores`, and so is
`metric="max"`: the largest score among a tissue's tracks grows with the number of tracks, so it ranks
tissues by track count rather than biology (Spearman 0.84–0.93 in our data).

**Choosing a null.** Use common variants with no association to your traits, matched to your test
variants on distance to the nearest gene. In our data, an unmatched null overstated the RNA-seq excess
of GWAS variants about four-fold. A matched null can, however, lift a tissue for every trait: the four
eye DNase tracks respond little to gene proximity, so against a proximity-matched null the eye ranked
first for eye and metabolic loci alike. `tissue_specificity()` removes such tissue-wide effects by
comparing each tissue with itself across several trait sets; report it alongside the absolute ranks.

**Speed.** `bootstrap_ranks` re-summarises the full table per resample; for millions of rows use
`n_boot` of 50–200, or resample only tracks.

## Tests

```
python tests/test_core.py
```

Synthetic data reproduce the failure mode (a DNase-only tissue wins under raw pooling) and check that
the safe path finds the tissue with true signal. `validate_on_study_data.py` checks that the package
reproduces the paper's numbers (Fig. 4a) on the study data.
