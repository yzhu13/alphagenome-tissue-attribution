# When every disease points to the eye

[![tests](https://github.com/yzhu13/alphagenome-tissue-attribution/actions/workflows/tests.yml/badge.svg)](https://github.com/yzhu13/alphagenome-tissue-attribution/actions/workflows/tests.yml)

Code and data for *When every disease points to the eye: assay-composition artifacts in AlphaGenome-based
tissue attribution* (Zhu Y, Vrathasha V, Bradford Y, *et al.*, O'Brien JM, Setia-Verma S).

Sequence-to-function models such as AlphaGenome predict a variant's effect on thousands of tracks from
hundreds of biosamples, and averaging those predictions by tissue is a tempting way to ask where GWAS
variants act. We show that such a tissue score mostly measures which assays happen to represent each
tissue. This repository contains the complete pipeline behind the paper, from the GWAS Catalog to every
figure and every number in the text. It also contains `tissue_attribution`, a small package that runs
the checks of the paper's Box 2 on your own scores.

## Contents

| Path | Contents |
|---|---|
| `scripts/` | the pipeline, `step01`–`step50`, figures, and `run_pipeline.py`, which runs it in order |
| `scripts/paths.py` | every file location, relative to the repository root |
| `tool/` | the `tissue_attribution` package, its tests, and a script that reproduces Fig. 3a from the study data |
| `data/inputs/` | inputs built by steps 01–05: track catalogue, GWAS loci, allele resolution, multiallelic audit, locus table |
| `data/` (other folders) | variant lists, null sets and every summary table behind the figures |
| `reports/` | one Markdown report per analysis step; the numbers in the paper are taken from these |
| `figures_v2/` | Figs. 1–5 and Supplementary Figs. 1–6 (PDF, SVG with editable text, 300-dpi PNG) |

The per-variant score tables (AlphaGenome scores for 9,782 variants × 6 scorers, the matched-null and
immune-disease score tables, and the Enformer predictions) are too large for GitHub and are deposited on
Zenodo (DOI added on publication).

## Pipeline

| Stage | Step | What it does | Output |
|---|---|---|---|
| inputs | `step01_track_catalog.py` | lists AlphaGenome's 5,559 human tracks and assigns 13 tissue categories by name rules (API) | `data/inputs/track_catalog_full.csv` |
| | `step02_gwas_loci.py` | GWAS Catalog associations (P < 5 × 10⁻⁸) → 15 traits by keyword → 500-kb distance clumping → loci per domain | `data/inputs/gwas_loci/` |
| | `step03_resolve_alleles.py` | GRCh38 REF/ALT from Ensembl allele strings (answers cached in `data/inputs/ensembl_allele_strings.json`) | `data/inputs/resolved_loci/` |
| | `step04_multiallelic_audit.py` | flags ALT alleles never observed in any population at multiallelic sites (Ensembl) | `data/inputs/multiallelic_alt_audit.csv` |
| | `step05_locus_table.py` | 8,084 domain–locus pairs (7,782 loci), domains within ±500 kb, ocular ATAC overlap | `data/inputs/gwas_locus_table.csv` |
| nulls | `step33_build_null_variants.py` | 2,000 random common variants with no association (FinnGen R10) | `data/null_variants/` |
| | `step39_build_matched_null.py` | 2,000 null variants matched to the GWAS loci on distance to the nearest gene | `data/null_matched/` |
| scores | `step34_full_rescoring.py` | AlphaGenome, six scorers per variant (API, resumable) | `data/*/parts/*.parquet` (Zenodo) |
| | `step40_build_immune_loci.py` | 1,154 lead variants of six immune-mediated diseases (GWAS Catalog REST) | `data/immune_loci/` |
| | `step47_enformer_scoring.py` | Enformer on 410 variants (CPU, ~3.5 h) | `data/enformer/scores/` (Zenodo) |
| analysis | `step35`–`step50` | aggregation, permutation, calibration, benchmarks, bootstrap, specificity, Enformer analysis, metabolic-loci sensitivity | `data/analysis/`, `reports/` |
| figures | `fig_v2_fig1.py`, `fig_v2_main.py`, `fig_v2_supp.py` | figures, each checked by `figure_qa.py` | `figures_v2/` |
| audit | `step44_number_audit.py` | recomputes every number quoted in the paper from the tables | `reports/number_audit.md` |

`python scripts/run_pipeline.py --help` lists every command in order. Each script's docstring describes
its inputs, method and outputs.

Names used in the code and tables, and their names in the paper:

| In code and tables | In the paper |
|---|---|
| `orig`, `orig_all`, "original pipeline", "raw pooled" | the mean-difference scorer (CenterMaskScorer, 501 bp, DIFF_MEAN) and the raw pooled summary |
| `rec`, "recommended scorers", `abs_q` / \|q\| | AlphaGenome's recommended scorers and their quantile scores |
| `emp`, "null percentile" | a score's percentile among the null variants on the same track |
| `top5`, "strong-effect fraction" | fraction of locus–track pairs with \|q\| ≥ 0.95 |
| `eye_misfiled` | the choroid plexus track that the name rules place in the eye (kept apart in all calibrated analyses) |
| `null`, `null_matched` | the random and the distance-matched null variants |

## Reproducing the results

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py inputs nulls           # rebuild data/inputs and the null sets (third-party files below)
python scripts/run_pipeline.py analysis figures audit # every analysis, figure and number, from the score tables
python scripts/run_pipeline.py all --with-api         # everything, including the AlphaGenome and Enformer scoring
```

The `analysis` and `figures` stages start from the per-variant score tables. Download them from Zenodo
and unpack them at the repository root; they fill `data/full_rescoring/parts/`,
`data/null_matched_scoring/parts/`, `data/immune_scoring/parts/`, `data/metabolic_batch_scoring/parts/`
and `data/enformer/scores/`. Rerun the
`scores` stage only to rescore from scratch.

- **AlphaGenome** scoring needs `alphagenome==0.6.1` and an API key in the environment variable
  `ALPHAGENOME_API_KEY`. The model is served remotely: scores for the same variant obtained two months apart
  differed by up to 0.43%. The developers also recalibrated the quantile tables on 18 June 2026. All scores
  in the paper were computed on 21 September 2026.
- **Enformer** needs `torch` and `enformer-pytorch==0.8.12`. The weights (`EleutherAI/enformer-official-rough`)
  are downloaded from Hugging Face, and sequence windows are fetched from the UCSC Genome Browser REST API.
- **Third-party files** go in `data/external/`, or you can point to them with the environment variables
  listed in `scripts/paths.py`:

| File | Source | Used by |
|---|---|---|
| `gwas_catalog_associations_2024-12-19.tsv` | [GWAS Catalog downloads](https://www.ebi.ac.uk/gwas/docs/file-downloads), full associations, release 2024-12-19 | step02 |
| `finngen_R10_HTN.gz` | [FinnGen R10](https://www.finngen.fi/en/access_results) summary statistics, endpoint I9_HYPTENS | step33, step39 |
| `Homo_sapiens.GRCh38.116.chr.gtf.gz` | [Ensembl release 116](https://ftp.ensembl.org/pub/release-116/gtf/homo_sapiens/) | step39, step39b |
| `Finucane2018_MOESM3_ESM.xlsx` | Finucane *et al.* 2018 *Nat Genet* 50:621, Supplementary Tables | step38, step40b |
| ATAC-seq peak files | Vrathasha *et al.*, GEO GSE324792 (private until publication) | step05 |

Steps 01–05 rebuild the shipped `data/inputs/` exactly. step03 uses the cached Ensembl answers; step05
keeps the shipped ATAC overlap flags if the peak files are absent (`--keep-atac-flags`).
`step34_full_rescoring.py` rebuilds the scored variant list (`data/full_rescoring/variants.csv`) from
these inputs byte for byte.

**Metabolic loci missing from the published locus set.** In the Ensembl query of June 2026, the first
batch of 200 metabolic rsIDs (the 200 most significant metabolic loci) timed out, so 199 of those loci
are "unresolved" in `data/inputs/` and are not among the 7,782 analysed variants. `step03` now splits a
failing batch and retries. `step50_metabolic_batch_sensitivity.py` re-queries the 199 loci (195
resolve), scores the 131 not already scored, and repeats the metabolic analyses with them. No conclusion
changes (`reports/metabolic_batch_sensitivity.md`, Supplementary Table 3).

## Using the tool on your own scores

```bash
pip install "git+https://github.com/yzhu13/alphagenome-tissue-attribution#subdirectory=tool"
```

```python
import tissue_attribution as ta
scores = ta.from_alphagenome(tidy_scores)          # variant_id | track_name | assay | score
ta.assay_composition(scores, tracks)               # check 1: tracks per tissue and assay
cal = ta.null_calibrate(scores, null_scores)       # check 3: percentile against null variants
ta.tissue_scores(cal, tracks)                      # checks 2 + 4: within assay, strong-effect fraction
ta.bootstrap_ranks(cal, tracks, assay="DNase")     # check 4: uncertainty over loci and tracks
ta.top_tracks(cal, tracks, "eye")                  # check 5: the tracks behind a winning tissue
ta.tissue_specificity(cal, tracks, trait_sets, assay="DNase")  # check 6: each tissue vs itself across traits
```

`tissue_scores` refuses raw scores and refuses `metric="max"`, because the maximum over a tissue's
tracks ranks tissues by how many tracks they have. It also warns when assays are pooled. The tool needs
only numpy and pandas; run its tests with `pip install "./tool[test]"` and `python -m pytest tool/tests`.

## Data availability and licences

- **Code** (`scripts/`, `tool/`): MIT licence (`LICENSE`).
- **AlphaGenome output and everything derived from it** is provided under and subject to the
  [AlphaGenome Output Terms of Use](https://deepmind.google.com/science/alphagenome/output-terms). This
  covers the score tables on Zenodo and, in this repository, the tables in `data/`, `reports/` and
  `figures_v2/`, except the Enformer and variant-list files. It may not be used for commercial purposes
  or to train machine-learning models for predicting variant effects. See
  `ALPHAGENOME_OUTPUT_TERMS_NOTICE.txt`.
- **Enformer** results (`data/enformer/`, Supplementary Fig. 6) and the variant lists: CC BY 4.0.
- **ATAC-seq peaks** of primary human trabecular meshwork cells and iPSC-derived retinal ganglion cells
  (Vrathasha *et al.*, GEO GSE324792) are not redistributed. Only a per-locus overlap flag is included.
- **Other sources:** GWAS loci are derived from the NHGRI-EBI GWAS Catalog. S-LDSC benchmark values are
  from Finucane *et al.* 2018 and are not redistributed.

## Citation

See `CITATION.cff`. Please cite the paper if you use the code or the checklist.
