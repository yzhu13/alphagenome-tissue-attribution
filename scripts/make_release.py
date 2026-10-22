"""
make_release.py

Builds the public repository in release/ (push this folder to GitHub) and, with --zenodo, the archive
of per-variant score tables for Zenodo in release_zenodo/.

The release keeps the working folder's layout (scripts/, tool/, data/, reports/, figures_v2/), so the
pipeline runs from a fresh clone without changes. Included: the pipeline scripts listed in SCRIPTS, the
tissue_attribution tool and its tests, data/inputs/ (steps 01-05), variant lists and null sets, every
summary table (<= MAX_MB), the analysis reports, the figures, and README, LICENSE, CITATION.cff,
requirements.txt, .gitignore and the AlphaGenome Output Terms notice (from release_template/).

Left out on purpose:
  - the API key (never in any file; scripts read ALPHAGENOME_API_KEY from the environment)
  - GSE324792 ATAC peak files (only the per-locus overlap flags in data/inputs/gwas_locus_table.csv)
  - per-variant score tables and caches larger than MAX_MB (Zenodo)
  - third-party files in data/external/
  - manuscript drafts, the manuscript builders and internal reviews
  - exploratory steps not used for the paper's numbers (step31, step32) and the superseded
    first-draft figure scripts (step37, step41)

Every text file is scanned for secrets and for absolute paths of this computer; the build fails if
either is found. MANIFEST.csv lists every file with its size and SHA-256.

    python scripts/make_release.py            # release/
    python scripts/make_release.py --zenodo   # also release_zenodo/score_tables.zip
"""
import argparse
import hashlib
import os
import re
import shutil
import sys
import time
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

NEW = paths.NEW
REL = os.path.join(NEW, "release")
ZEN = os.path.join(NEW, "release_zenodo")
TEMPLATE = os.path.join(NEW, "release_template")
MAX_MB = 5

SCRIPTS = [
    "paths.py", "run_pipeline.py",
    "step01_track_catalog.py", "step02_gwas_loci.py", "step03_resolve_alleles.py",
    "step04_multiallelic_audit.py", "step05_locus_table.py",
    "step33_build_null_variants.py", "step34_full_rescoring.py", "step35_analysis.py",
    "step36_benchmark_expected_tissue.py", "step38_benchmark_sldsc.py", "step39_build_matched_null.py",
    "step39b_matched_null_analysis.py", "step40_build_immune_loci.py", "step40b_immune_analysis.py",
    "step42_uncertainty.py", "step43_sensitivity.py", "step44_number_audit.py",
    "step45_aggregation_variants.py", "step46_grouping_sensitivity.py", "step47_enformer_scoring.py",
    "step48_enformer_analysis.py", "step49_recommended_pipeline_check.py", "step50_metabolic_batch_sensitivity.py",
    "enformer_tissue_map.py",
    "make_supp_table2.py", "figstyle.py", "figure_qa.py", "fig_v2_fig1.py", "fig_v2_main.py", "fig_v2_supp.py",
    "make_release.py",
]
DATA_DIRS = ["data/inputs", "data/null_variants", "data/null_matched", "data/immune_loci", "data/metabolic_batch",
             "data/analysis", "data/immune_analysis", "data/null_matched_analysis", "data/uncertainty"]
DATA_FILES = ["data/full_rescoring/variants.csv", "data/null_matched_scoring/variants.csv",
              "data/immune_scoring/variants.csv", "data/metabolic_batch_scoring/variants.csv",
              "data/enformer/targets_human.txt",
              "data/enformer/track_tissue_map.csv", "data/enformer/variants_enformer.csv"]
DATA_GLOBS = [("data/enformer", "step48_")]
SKIP_EXT = (".parquet", ".npz")  # caches, rebuilt from the score tables
SKIP_NAMES = ("aggregated_parts.txt",)  # cache stamps: shipped without their cache they would mislead
REPORTS = ["aggregation_variants.md", "analysis_A1_A7.md", "benchmark_expected_tissue.md", "benchmark_sldsc.md",
           "enformer_validation.md", "grouping_sensitivity.md", "immune_generalization.md", "matched_null.md",
           "recommended_pipeline_check.md", "sensitivity.md", "uncertainty.md", "metabolic_batch_sensitivity.md"]
ZENODO_DIRS = ["data/full_rescoring/parts", "data/null_matched_scoring/parts", "data/immune_scoring/parts",
               "data/metabolic_batch_scoring/parts", "data/enformer/scores"]

SECRET = re.compile(r"[0-9a-z]{16,}token|reviewer token|ALPHAGENOME_API_KEY\s*=\s*['\"][^'\"]|AIza[0-9A-Za-z_-]{20,}", re.I)
_HOME = os.path.expanduser("~")  # this computer's home and project folders, built at run time
_LOCAL = {_HOME, _HOME.replace("\\", "/"), os.path.basename(_HOME), os.path.dirname(NEW),
          os.path.dirname(NEW).replace("\\", "/")}
LOCAL_PATH = re.compile("|".join(re.escape(t) for t in sorted(_LOCAL, key=len, reverse=True) if len(t) > 3), re.I)

LICENSE = """MIT License

Copyright (c) 2026 Yan Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

AG_NOTICE = """Legally Binding Terms of Use

By using this information, you agree to AlphaGenome Output Terms of Use found at
http://deepmind.google.com/science/alphagenome/output-terms

This repository and the associated Zenodo record contain output of the AlphaGenome API (Google
DeepMind) and material derived from it: per-track variant scores, null-calibrated percentiles, and
tissue- and domain-level summaries, tables and figures. Modifications made by us: calibration of each
track's scores against null variants, and aggregation of per-track scores to tissues and disease
domains (see the paper's Methods and scripts/). Use restrictions of the AlphaGenome Output Terms of Use
apply to all of this material, including that it must not be used in connection with commercial
activities or to train machine-learning models, except as those Terms allow.

Files NOT derived from AlphaGenome: scripts/ and tool/ (code, MIT licence); data/enformer/ and
figures_v2/SuppFig6_enformer.* (Enformer); the variant lists in data/ (variants.csv, null_variants.csv,
null_matched_variants.csv, immune_variants.csv) and the GWAS loci in data/inputs/gwas_loci/ and
data/inputs/resolved_loci/.
"""


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def rmtree(path):
    for _ in range(5):  # a sync client or antivirus can hold a handle for a moment
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(2)
    shutil.rmtree(path)


def build_release():
    if os.path.exists(REL):
        rmtree(REL)
    os.makedirs(REL)
    skipped = []

    for f in SCRIPTS:
        copy(os.path.join(HERE, f), os.path.join(REL, "scripts", f))
    for root, dirs, files in os.walk(os.path.join(NEW, "tool")):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache")]
        for f in files:
            src = os.path.join(root, f)
            copy(src, os.path.join(REL, os.path.relpath(src, NEW)))

    def add(rel):
        src = os.path.join(NEW, *rel.split("/"))
        mb = os.path.getsize(src) / 1e6
        if rel.endswith(SKIP_NAMES):
            return
        if rel.endswith(SKIP_EXT) or mb > MAX_MB:
            skipped.append((rel, f"{mb:.1f} MB" + (", cache" if rel.endswith(SKIP_EXT) else ", Zenodo")))
        else:
            copy(src, os.path.join(REL, *rel.split("/")))

    for d in DATA_DIRS:
        for root, dirs, files in os.walk(os.path.join(NEW, *d.split("/"))):
            dirs[:] = [x for x in dirs if not x.startswith(".")]
            for f in sorted(files):
                if not f.startswith("."):
                    add(os.path.relpath(os.path.join(root, f), NEW).replace("\\", "/"))
    for rel in DATA_FILES:
        add(rel)
    for d, prefix in DATA_GLOBS:
        for f in sorted(os.listdir(os.path.join(NEW, d))):
            if f.startswith(prefix):
                add(f"{d}/{f}")
    for f in REPORTS:
        add(f"reports/{f}")
    for f in sorted(os.listdir(os.path.join(NEW, "figures_v2"))):
        add(f"figures_v2/{f}")

    for f in os.listdir(TEMPLATE):
        copy(os.path.join(TEMPLATE, f), os.path.join(REL, f))
    for name, text in [("LICENSE", LICENSE), ("ALPHAGENOME_OUTPUT_TERMS_NOTICE.txt", AG_NOTICE)]:
        with open(os.path.join(REL, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    return skipped


def scan_and_manifest():
    rows, flagged = [], []
    for root, _, files in os.walk(REL):
        for f in files:
            p = os.path.join(root, f)
            b = open(p, "rb").read()
            rel = os.path.relpath(p, REL).replace("\\", "/")
            rows.append({"path": rel, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
            if "peaks.bed" in f:
                flagged.append((rel, "ATAC peak file"))
            if f.rsplit(".", 1)[-1] in {"py", "md", "txt", "csv", "json", "cff", "svg"} or f.startswith("."):
                text = b.decode("utf-8", "ignore")
                for pat, what in [(SECRET, "secret"), (LOCAL_PATH, "local path")]:
                    hit = pat.search(text)
                    if hit and not (what == "secret" and f == "make_release.py"):  # its own pattern
                        flagged.append((rel, f"{what}: {hit.group(0)[:40]}"))
    m = pd.DataFrame(rows).sort_values("path")
    m.to_csv(os.path.join(REL, "MANIFEST.csv"), index=False)
    return m, flagged


def build_zenodo():
    os.makedirs(ZEN, exist_ok=True)
    out = os.path.join(ZEN, "score_tables.zip")
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # parquet is already compressed
        for d in ZENODO_DIRS:
            for root, _, files in os.walk(os.path.join(NEW, *d.split("/"))):
                for f in sorted(files):
                    p = os.path.join(root, f)
                    z.write(p, os.path.relpath(p, NEW).replace("\\", "/"))
                    n += 1
        z.writestr("ALPHAGENOME_OUTPUT_TERMS_NOTICE.txt", AG_NOTICE)
    print(f"zenodo: {n} files, {os.path.getsize(out) / 1e6:.0f} MB -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", action="store_true", help="also write release_zenodo/score_tables.zip")
    args = ap.parse_args()
    skipped = build_release()
    m, flagged = scan_and_manifest()
    print(f"{len(m)} files, {m.bytes.sum() / 1e6:.1f} MB -> {REL}")
    if skipped:
        print("not included:")
        for a, b in skipped:
            print(f"  {a}: {b}")
    print("secret / local-path scan:", "CLEAN" if not flagged else "")
    for a, b in flagged:
        print(f"  FLAGGED {a}: {b}")
    if args.zenodo:
        build_zenodo()
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
