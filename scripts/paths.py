"""
paths.py

Every file location used by the pipeline, in one place. All paths are relative to the repository
root, so the code runs from any checkout:

  data/inputs/     small inputs built by steps 01-05 and shipped with the repository
  data/external/   third-party files that you download yourself (not shipped; see README.md)
  data/<name>/     tables written by the pipeline (scores, summaries, bootstrap draws)
  reports/         analysis reports, one per step
  figures_v2/      figures (PDF, SVG, PNG)

A third-party file kept somewhere else can be pointed to with the environment variable named next to
it, or with an optional scripts/paths_local.py (ignored by git) that assigns the same names.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(HERE)  # repository root
DATA = os.path.join(NEW, "data")
INPUTS = os.path.join(DATA, "inputs")
EXTERNAL = os.path.join(DATA, "external")
REPORTS = os.path.join(NEW, "reports")


def _external(env, name):
    return os.environ.get(env) or os.path.join(EXTERNAL, name)


# ── Inputs built by steps 01-05 (shipped) ─────────────────────────────────────────────────────────
TRACK_CATALOG = os.path.join(INPUTS, "track_catalog_full.csv")              # step01
GWAS_LOCI_DIR = os.path.join(INPUTS, "gwas_loci")                           # step02
RESOLVED_DIR = os.path.join(INPUTS, "resolved_loci")                        # step03
ALLELE_RESOLUTION = os.path.join(INPUTS, "variant_allele_resolution.csv")   # step03
AT_RISK_MULTIALLELIC = os.path.join(INPUTS, "at_risk_multiallelic_loci.csv")  # step04
MULTIALLELIC_AUDIT = os.path.join(INPUTS, "multiallelic_alt_audit.csv")     # step04
LOCUS_TABLE = os.path.join(INPUTS, "gwas_locus_table.csv")                  # step05

# ── Third-party files (not shipped) ───────────────────────────────────────────────────────────────
# GWAS Catalog full associations file, release of 2024-12-19 (https://www.ebi.ac.uk/gwas/docs/file-downloads)
GWAS_CATALOG_TSV = _external("GWAS_CATALOG_TSV", "gwas_catalog_associations_2024-12-19.tsv")
# FinnGen R10 summary statistics for hypertension (I9_HYPTENS), https://www.finngen.fi/en/access_results
FINNGEN_HTN = _external("FINNGEN_HTN", "finngen_R10_HTN.gz")
# Ensembl release 116 gene annotation, https://ftp.ensembl.org/pub/release-116/gtf/homo_sapiens/
ENSEMBL_GTF = _external("ENSEMBL_GTF", "Homo_sapiens.GRCh38.116.chr.gtf.gz")
# Finucane et al. 2018, Supplementary Table (S-LDSC tissue enrichments), Nat Genet 50:621
FINUCANE_XLSX = _external("FINUCANE_XLSX", "Finucane2018_MOESM3_ESM.xlsx")
# ATAC-seq reproducible peaks, Vrathasha et al. (GEO GSE324792; private until publication)
ATAC_HTMC = _external("ATAC_HTMC", "Vrathasha_hTMC_reproducible_peaks.bed.gz")
ATAC_RGC = _external("ATAC_RGC", "Vrathasha_iPSC_RGC_reproducible_peaks.bed.gz")
# GWAS Catalog REST pages for the immune-mediated diseases, saved by step40
GWAS_CATALOG_IMMUNE = os.path.join(EXTERNAL, "gwas_catalog_immune")

try:  # machine-specific overrides, never committed
    import paths_local as _local
except ImportError:
    _local = None
if _local is not None:
    globals().update({k: v for k, v in vars(_local).items() if k.isupper()})
