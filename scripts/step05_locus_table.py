"""
step05_locus_table.py

Builds the table of GWAS loci analysed in the paper: one row per domain-locus pair (8,084 rows,
7,782 unique loci), with the number of disease domains near each locus and whether it lies in open
chromatin of two ocular cell types.

  loci        every resolved locus of each domain (step03), first row per rsID within a domain,
              excluding loci within 524,288 bp (half the AlphaGenome input window) of a chromosome
              start
  n_disease_cats / cats_present
              disease domains with a locus within +-500 kb (counting the locus itself)
  in_hTMC_atac / in_iPSC_RGC_atac
              any ATAC-seq reproducible peak within +-500 bp of the variant, for primary human
              trabecular meshwork cells and iPSC-derived retinal ganglion cells (Vrathasha et al.,
              GEO GSE324792)

The peak files are not public yet. Without them (or with --keep-atac-flags) the script rebuilds every
other column and keeps the two overlap flags of the shipped table.

Row order: when the table already exists, rows keep its order (new rows go to the end). The shipped
order is the one used for the paper; it matters only because bootstrap resamples are drawn by row.

Input:  data/inputs/resolved_loci/<domain>_loci_resolved.csv (step03), ATAC peaks (paths.ATAC_*)
Output: data/inputs/gwas_locus_table.csv

    python scripts/step05_locus_table.py
    python scripts/step05_locus_table.py --keep-atac-flags
"""
import argparse
import gzip
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

DOMAINS = ("eye", "kidney", "metabolic")
HALF_WINDOW = 1_048_576 // 2   # AlphaGenome input length / 2
CLUSTER_WINDOW = 500_000
PEAK_WINDOW = 500
FLAGS = ["in_hTMC_atac", "in_iPSC_RGC_atac"]


def domain_loci(domain):
    r = pd.read_csv(os.path.join(paths.RESOLVED_DIR, f"{domain}_loci_resolved.csv"))
    r = r[~r.resolution.astype(str).str.startswith("unresolved")].dropna(subset=["CHR", "pos"])
    r = r[r.pos >= HALF_WINDOW].drop_duplicates("rsid")
    out = r[["rsid", "CHR", "pos", "source_disease"]].copy()
    out["disease_class"] = domain
    out["CHR"] = out.CHR.astype(int)
    out["pos"] = out.pos.astype(int)
    return out


def count_domains(df):
    by_chr = {c: g for c, g in df.groupby("CHR")}
    n, cats = [], []
    for row in df.itertuples(index=False):
        g = by_chr[row.CHR]
        s = set(g.disease_class[(g.pos >= row.pos - CLUSTER_WINDOW) & (g.pos <= row.pos + CLUSTER_WINDOW)])
        n.append(len(s))
        cats.append("|".join(sorted(s)))
    return df.assign(n_disease_cats=n, cats_present=cats)


def load_bed(path):
    peaks = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) >= 3:
                peaks.setdefault(parts[0].replace("chr", ""), []).append((int(parts[1]), int(parts[2])))
    return {c: np.array(v, dtype=int) for c, v in peaks.items()}


def overlaps(df, peaks):
    hit = []
    for row in df.itertuples(index=False):
        arr = peaks.get(str(row.CHR))
        lo, hi = row.pos - PEAK_WINDOW, row.pos + PEAK_WINDOW
        hit.append(arr is not None and len(arr) > 0 and bool(((lo < arr[:, 1]) & (hi > arr[:, 0])).any()))
    return np.array(hit, dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-atac-flags", action="store_true",
                    help="take the two ATAC overlap flags from the existing table instead of the peak files")
    args = ap.parse_args()

    loci = count_domains(pd.concat([domain_loci(d) for d in DOMAINS], ignore_index=True))
    peaks = (paths.ATAC_HTMC, paths.ATAC_RGC)
    if not args.keep_atac_flags and not all(os.path.exists(p) for p in peaks):
        print("ATAC peak files not found; keeping the overlap flags of the shipped table")
        args.keep_atac_flags = True
    if args.keep_atac_flags:
        old = pd.read_csv(paths.LOCUS_TABLE).drop_duplicates("rsid").set_index("rsid")
        for f in FLAGS:
            loci[f] = loci.rsid.map(old[f]).astype(bool)
    else:
        for f, path in zip(FLAGS, peaks):
            loci[f] = overlaps(loci, load_bed(path))
    if os.path.exists(paths.LOCUS_TABLE):  # keep the shipped row order
        old = pd.read_csv(paths.LOCUS_TABLE, usecols=["disease_class", "rsid"])
        rank = {k: i for i, k in enumerate(zip(old.disease_class, old.rsid))}
        key = [rank.get(k, len(rank) + i) for i, k in enumerate(zip(loci.disease_class, loci.rsid))]
        loci = loci.iloc[np.argsort(key, kind="stable")].reset_index(drop=True)
    loci.to_csv(paths.LOCUS_TABLE, index=False)
    print(f"{len(loci):,} domain-locus rows, {loci.rsid.nunique():,} unique loci "
          f"{loci.disease_class.value_counts().to_dict()} -> {paths.LOCUS_TABLE}")


if __name__ == "__main__":
    main()
