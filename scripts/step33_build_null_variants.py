"""
step33_build_null_variants.py

Builds the random null variant set: common SNVs with no association to the traits in this
study, used to show that the raw-score pipeline ranks the eye first even for random variants.

Source: FinnGen R10 hypertension summary statistics (hg38, REF/ALT, allele frequency), already local.
Filters: autosomal SNVs · 0.05 <= alt allele frequency <= 0.95 · HTN P > 0.5 · outside the MHC
(chr6:25-34 Mb) · at least 100 kb from every GWAS locus scored in this study. Then a fixed-seed
random sample of 2,000.

    python scripts/step33_build_null_variants.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

NEW = paths.NEW
FINNGEN = paths.FINNGEN_HTN       # FinnGen R10 I9_HYPTENS summary statistics
GWAS_LOCI = paths.LOCUS_TABLE     # step05
OUT = os.path.join(NEW, "data", "null_variants", "null_variants.csv")

SEED = 20260921
N = 2000
MIN_DIST = 100_000
MHC = (6, 25_000_000, 34_000_000)

gwas = pd.read_csv(GWAS_LOCI).drop_duplicates("rsid")
gwas_pos = {int(c): np.sort(g.pos.astype(int).values) for c, g in gwas.groupby("CHR")}


def far_from_gwas(chrom, pos):
    p = gwas_pos.get(chrom)
    if p is None:
        return np.ones(len(pos), bool)
    i = np.searchsorted(p, pos)
    left = np.abs(pos - p[np.clip(i - 1, 0, len(p) - 1)])
    right = np.abs(p[np.clip(i, 0, len(p) - 1)] - pos)
    return np.minimum(left, right) >= MIN_DIST


keep = []
cols = ["#chrom", "pos", "ref", "alt", "rsids", "pval", "af_alt"]
for chunk in pd.read_csv(FINNGEN, sep="\t", usecols=cols, chunksize=2_000_000,
                         dtype={"#chrom": str, "rsids": str}):
    c = chunk[chunk["#chrom"].str.fullmatch(r"\d+") & (chunk.ref.str.len() == 1) & (chunk.alt.str.len() == 1)
              & chunk.af_alt.between(0.05, 0.95) & (chunk.pval > 0.5)].copy()
    c["CHR"] = c["#chrom"].astype(int)
    c = c[c.CHR <= 22]
    c = c[~((c.CHR == MHC[0]) & c.pos.between(MHC[1], MHC[2]))]
    ok = np.zeros(len(c), bool)
    for chrom, idx in c.groupby("CHR").indices.items():
        ok[idx] = far_from_gwas(chrom, c.pos.values[idx])
    keep.append(c[ok])
    print(f"  chunk done, {sum(len(k) for k in keep):,} candidates so far", flush=True)

cand = pd.concat(keep, ignore_index=True)
null = cand.sample(N, random_state=SEED).sort_values(["CHR", "pos"])
null = pd.DataFrame({
    "id": [f"null_{c}_{p}_{r}_{a}" for c, p, r, a in zip(null.CHR, null.pos, null.ref, null.alt)],
    "rsid": null.rsids.fillna(""), "CHR": null.CHR, "pos": null.pos,
    "ref_allele": null.ref, "alt_allele": null.alt, "af_alt": null.af_alt, "htn_pval": null.pval})
os.makedirs(os.path.dirname(OUT), exist_ok=True)
null.to_csv(OUT, index=False)
print(f"{len(cand):,} candidates -> sampled {len(null)} -> {OUT}")
