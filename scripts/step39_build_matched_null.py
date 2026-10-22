"""
step39_build_matched_null.py

Builds a second null variant set, matched to the GWAS loci on distance to the nearest
protein-coding TSS. Why: the first null set (step33) is random common SNVs, which sit further from
genes than GWAS loci. Relative to it, GWAS loci are shifted more in RNA-seq (mean null-percentile
0.66) than in DNase (0.54). This set tests whether that RNA > DNase gap just reflects gene proximity.

Gene annotation: Ensembl release 116 GTF, primary chromosomes (= GENCODE release 50 comprehensive
gene set; Ensembl and GENCODE human gene models are identical, GENCODE 50 <-> Ensembl 116):
    data/external/Homo_sapiens.GRCh38.116.chr.gtf.gz   (141,042,687 bytes, downloaded 2026-09-21)
    https://ftp.ensembl.org/pub/release-116/gtf/homo_sapiens/Homo_sapiens.GRCh38.116.chr.gtf.gz
    (the GENCODE FTP host ftp.ebi.ac.uk refused connections from this machine on 2026-09-21; the
    Ensembl copy was used instead; its checksum matched Ensembl's CHECKSUMS file)
TSS = gene start (+ strand) or gene end (- strand) of every gene with gene_biotype protein_coding.

Candidate pool: FinnGen R10 hypertension sumstats with the same filters as step33 (autosomal SNVs,
0.05 <= af_alt <= 0.95, HTN P > 0.5, outside the MHC, >= 100 kb from every study locus), minus the
variants already in the first null set, one variant per position.
Matching: 20 quantile bins of log10(distance + 1) defined on the 7,782 GWAS loci; from each bin
draw as many candidates as the GWAS share of that bin x 2,000. Fixed seed 20260921.

Output: data/null_matched/null_matched_variants.csv
        (id, set, CHR, pos, ref_allele, alt_allele, af_alt, htn_pval, dist_tss; set = "null_matched")
Prints matching quality (distance quantiles GWAS / random null / matched null, KS statistics).

    python scripts/step39_build_matched_null.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

NEW = paths.NEW
FINNGEN = paths.FINNGEN_HTN       # FinnGen R10 I9_HYPTENS summary statistics
GWAS_LOCI = paths.LOCUS_TABLE     # step05
GTF = paths.ENSEMBL_GTF
NULL1 = os.path.join(NEW, "data", "null_variants", "null_variants.csv")
OUT = os.path.join(NEW, "data", "null_matched", "null_matched_variants.csv")

SEED = 20260921
N = 2000
N_BINS = 20
MIN_DIST = 100_000
MHC = (6, 25_000_000, 34_000_000)


def load_tss(gtf=GTF):
    """{chromosome (int, autosomes): sorted array of protein-coding gene TSSs (1-based)}."""
    genes = []
    for ch in pd.read_csv(gtf, sep="\t", comment="#", header=None, usecols=[0, 2, 3, 4, 6, 8],
                          names=["chrom", "feature", "start", "end", "strand", "attr"],
                          dtype={"chrom": str}, quoting=3, chunksize=500_000):
        g = ch[(ch.feature == "gene") & ch.attr.str.contains('gene_biotype "protein_coding"', regex=False)]
        genes.append(g[["chrom", "start", "end", "strand"]])
    g = pd.concat(genes)
    g = g[g.chrom.str.fullmatch(r"\d+")]
    g["tss"] = np.where(g.strand == "+", g.start, g.end)
    return {int(c): np.sort(x.tss.values) for c, x in g.groupby("chrom")}


def dist_to_tss(chrom, pos, tss):
    """Absolute distance (bp) to the nearest TSS on the same chromosome; vectorised per chromosome."""
    chrom, pos = np.asarray(chrom, int), np.asarray(pos, np.int64)
    out = np.full(len(pos), np.nan)
    for c in np.unique(chrom):
        m = chrom == c
        t = tss.get(int(c))
        if t is None or len(t) == 0:
            continue
        p = pos[m]
        i = np.searchsorted(t, p)
        left = np.abs(p - t[np.clip(i - 1, 0, len(t) - 1)])
        right = np.abs(t[np.clip(i, 0, len(t) - 1)] - p)
        out[m] = np.minimum(left, right)
    return out


def quantile_table(sets, qs=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)):
    return pd.DataFrame({k: np.quantile(v, qs) for k, v in sets.items()}, index=[f"q{int(q * 100)}" for q in qs])


def main():
    tss = load_tss()
    print(f"protein-coding TSSs (autosomes): {sum(len(v) for v in tss.values()):,}")

    gwas = pd.read_csv(GWAS_LOCI).drop_duplicates("rsid")
    gwas["dist_tss"] = dist_to_tss(gwas.CHR, gwas.pos, tss)
    null1 = pd.read_csv(NULL1)
    null1["dist_tss"] = dist_to_tss(null1.CHR, null1.pos, tss)
    null1_keys = set(zip(null1.CHR, null1.pos))
    gwas_pos = {int(c): np.sort(g.pos.astype(int).values) for c, g in gwas.groupby("CHR")}

    def far_from_gwas(chrom, pos):  # same rule as step33
        p = gwas_pos.get(chrom)
        if p is None:
            return np.ones(len(pos), bool)
        i = np.searchsorted(p, pos)
        left = np.abs(pos - p[np.clip(i - 1, 0, len(p) - 1)])
        right = np.abs(p[np.clip(i, 0, len(p) - 1)] - pos)
        return np.minimum(left, right) >= MIN_DIST

    keep = []
    cols = ["#chrom", "pos", "ref", "alt", "pval", "af_alt"]
    for chunk in pd.read_csv(FINNGEN, sep="\t", usecols=cols, chunksize=2_000_000, dtype={"#chrom": str}):
        c = chunk[chunk["#chrom"].str.fullmatch(r"\d+") & (chunk.ref.str.len() == 1) & (chunk.alt.str.len() == 1)
                  & chunk.af_alt.between(0.05, 0.95) & (chunk.pval > 0.5)].copy()
        c["CHR"] = c["#chrom"].astype(int)
        c = c[c.CHR <= 22]
        c = c[~((c.CHR == MHC[0]) & c.pos.between(MHC[1], MHC[2]))]
        ok = np.zeros(len(c), bool)
        for chrom, idx in c.groupby("CHR").indices.items():
            ok[idx] = far_from_gwas(chrom, c.pos.values[idx])
        keep.append(c[ok].drop(columns="#chrom"))
        print(f"  chunk done, {sum(len(k) for k in keep):,} candidates so far", flush=True)
    cand = pd.concat(keep, ignore_index=True)
    n0 = len(cand)
    cand = cand[[k not in null1_keys for k in zip(cand.CHR, cand.pos)]]
    cand = cand.drop_duplicates(["CHR", "pos"])
    cand["dist_tss"] = dist_to_tss(cand.CHR, cand.pos, tss)
    cand = cand[cand.dist_tss.notna()]
    print(f"candidates: {n0:,} -> {len(cand):,} after removing first-null positions and duplicate positions")

    # 20 quantile bins of log10(distance + 1) on the GWAS loci; open-ended outer bins
    lg, lc = np.log10(gwas.dist_tss + 1), np.log10(cand.dist_tss + 1)
    edges = np.unique(np.quantile(lg, np.linspace(0, 1, N_BINS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    gb = pd.cut(lg, edges, labels=False, include_lowest=True)
    cb = pd.cut(lc, edges, labels=False, include_lowest=True)
    share = pd.Series(gb).value_counts(normalize=True).sort_index()
    target = np.floor(share * N).astype(int)
    rem = N - target.sum()  # largest-remainder rounding so the total is exactly N
    target[(share * N - target).sort_values(ascending=False).index[:rem]] += 1
    rng = np.random.default_rng(SEED)
    picks = []
    for b, k in target.items():
        pool = np.flatnonzero(cb.values == b)
        if len(pool) < k:
            raise SystemExit(f"bin {b}: only {len(pool)} candidates for {k} draws")
        picks.append(rng.choice(pool, k, replace=False))
    m = cand.iloc[np.concatenate(picks)].sort_values(["CHR", "pos"])
    out = pd.DataFrame({
        "id": [f"nullm_{c}_{p}_{r}_{a}" for c, p, r, a in zip(m.CHR, m.pos, m.ref, m.alt)],
        "set": "null_matched", "CHR": m.CHR.values, "pos": m.pos.values,
        "ref_allele": m.ref.values, "alt_allele": m.alt.values, "af_alt": m.af_alt.values,
        "htn_pval": m.pval.values, "dist_tss": m.dist_tss.astype(int).values})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"sampled {len(out)} -> {OUT}\n")

    # matching quality
    sets = {"GWAS": gwas.dist_tss.values, "random null": null1.dist_tss.values,
            "matched null": out.dist_tss.values, "candidate pool": cand.dist_tss.values}
    print("distance to nearest protein-coding TSS (bp):")
    print(quantile_table(sets).round(0).astype(int).to_string())
    for k in ["random null", "matched null", "candidate pool"]:
        r = ks_2samp(sets["GWAS"], sets[k])
        print(f"KS GWAS vs {k:14s}: D = {r.statistic:.3f}, P = {r.pvalue:.2g}")
    maf = lambda a: np.minimum(a, 1 - a)
    print(f"MAF median: random null {np.median(maf(null1.af_alt)):.3f}, matched null {np.median(maf(out.af_alt)):.3f}")


if __name__ == "__main__":
    main()
