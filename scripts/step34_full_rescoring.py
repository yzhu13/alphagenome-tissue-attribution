"""
step34_full_rescoring.py

Scores every GWAS locus in the study (7,782) plus the 2,000 null variants
from step33, in one pass, with six scorers per API call:

  recommended (calibrated):  DNase, ATAC  CenterMaskScorer 501 bp DIFF_LOG2_SUM
                             RNA-seq      GeneMaskLFCScorer
  original pipeline (raw):   DNase, ATAC, RNA-seq  CenterMaskScorer 501 bp DIFF_MEAN

Keeps raw_score and quantile_score per track (DIFF_MEAN has no quantile scores: they are NaN).
The gene-level RNA scorer returns one row per gene x track; only the gene with the largest
|quantile_score| per track is kept.

Multiallelic loci whose ALT was never observed in any population (670, from the 2026-08-03 audit)
are scored with the most frequent observed non-REF allele instead; unrecoverable ones are dropped.

Output: data/full_rescoring/variants.csv, parts/part_NNNNN.parquet (100 variants each), done.txt.
Resumable: rerun the same command after an interruption.

    python -u scripts/step34_full_rescoring.py            # run / resume
    python -u scripts/step34_full_rescoring.py --limit 5  # quick test

Scoring another variant set (added 2026-09-21 so that side analyses never touch the main run's
files): pass a CSV with columns id, set, CHR, pos, ref_allele, alt_allele and its own output folder.

    python -u scripts/step34_full_rescoring.py --variants data/X/variants_in.csv --out data/X_scoring
"""
import argparse
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

NEW = paths.NEW
RES_DIR = paths.RESOLVED_DIR          # step03
LOCI = paths.LOCUS_TABLE              # step05
MULTI_F = paths.MULTIALLELIC_AUDIT    # step04
NULL_F = os.path.join(NEW, "data", "null_variants", "null_variants.csv")
OUT = os.path.join(NEW, "data", "full_rescoring")
PARTS = os.path.join(OUT, "parts")
VARIANTS_F = os.path.join(OUT, "variants.csv")
DONE_F = os.path.join(OUT, "done.txt")

FLUSH_EVERY = 100
MAX_RETRIES = 4


def get_key():
    key = os.environ.get("ALPHAGENOME_API_KEY")
    if not key and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = winreg.QueryValueEx(k, "ALPHAGENOME_API_KEY")[0]
        except OSError:
            key = None
    if not key:
        sys.exit("ALPHAGENOME_API_KEY is not set (see README.md, Reproducing the analysis).")
    return key


def observed_alt(row):
    """For a never-observed ALT, pick the most frequent observed non-REF allele, else None."""
    seen = {}
    for part in str(row.alleles_seen).split(","):
        if ":" in part:
            a, f = part.split(":")
            seen[a.strip()] = float(f)
    cands = [a for a in str(row.ensembl_allele_string).split("/") if a != row.ref_allele and a in seen]
    return max(cands, key=seen.get) if cands else None


def build_variants():
    if os.path.exists(VARIANTS_F):
        return pd.read_csv(VARIANTS_F)
    loci = pd.read_csv(LOCI).drop_duplicates("rsid")[["rsid", "CHR", "pos"]]
    al = pd.concat(pd.read_csv(os.path.join(RES_DIR, f"{d}_loci_resolved.csv"))
                   for d in ("eye", "kidney", "metabolic"))
    al = al[~al.resolution.astype(str).str.startswith("unresolved")].drop_duplicates("rsid")
    g = loci.merge(al[["rsid", "ref_allele", "alt_allele"]], on="rsid")
    g["alt_fixed"] = False
    multi = pd.read_csv(MULTI_F)
    bad = multi[multi.verdict.str.startswith("SUSPECT")].copy()
    bad["new_alt"] = bad.apply(observed_alt, axis=1)
    fix = bad.set_index("rsid").new_alt
    hit = g.rsid.isin(fix.index)
    g.loc[hit, "alt_allele"] = g.loc[hit, "rsid"].map(fix)
    g.loc[hit, "alt_fixed"] = True
    g = g[g.ref_allele.isin(list("ACGT")) & g.alt_allele.isin(list("ACGT")) & (g.ref_allele != g.alt_allele)]
    g = g.assign(id=g.rsid, set="gwas")
    n = pd.read_csv(NULL_F).assign(set="null", alt_fixed=False)
    v = pd.concat([n[["id", "set", "CHR", "pos", "ref_allele", "alt_allele", "alt_fixed"]],
                   g[["id", "set", "CHR", "pos", "ref_allele", "alt_allele", "alt_fixed"]]], ignore_index=True)
    v["CHR"] = v.CHR.astype(int)
    v["pos"] = v.pos.astype(int)
    os.makedirs(OUT, exist_ok=True)
    v.to_csv(VARIANTS_F, index=False)
    print(f"variants: {len(v)}  (null {sum(v.set == 'null')}, gwas {sum(v.set == 'gwas')}, "
          f"ALT corrected {int(v.alt_fixed.sum())}, dropped as unrecoverable {int(hit.sum() - g.alt_fixed.sum())})")
    return v


def scorer_code(s):
    s = str(s)
    if s.startswith("GeneMaskLFC"):
        return "rna_rec"
    assay = "dnase" if "DNASE" in s else "atac" if "ATAC" in s else "rna"
    return f"{assay}_rec" if "DIFF_LOG2_SUM" in s else f"{assay}_orig"


def compress(df, vid):
    df = df.assign(scorer=df.variant_scorer.map(scorer_code))
    rna = df[df.scorer == "rna_rec"]
    if len(rna):
        top = rna.loc[rna.quantile_score.abs().groupby(rna.track_name).idxmax()]
        df = pd.concat([df[df.scorer != "rna_rec"], top])
    out = pd.DataFrame({"id": vid, "scorer": df.scorer.values, "track_name": df.track_name.values,
                        "raw_score": df.raw_score.astype("float32").values,
                        "quantile_score": df.quantile_score.astype("float32").values,
                        "gene_name": df.gene_name.where(df.scorer == "rna_rec").values})
    return out


def main():
    global OUT, PARTS, VARIANTS_F, DONE_F
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--variants", help="CSV of variants to score instead of the study set")
    ap.add_argument("--out", help="output folder (required with --variants)")
    args = ap.parse_args()
    if args.variants and not args.out:
        sys.exit("--variants needs --out, so the main run's files are never touched")
    if args.out:
        OUT = os.path.abspath(args.out)
        PARTS, VARIANTS_F, DONE_F = (os.path.join(OUT, "parts"), os.path.join(OUT, "variants.csv"),
                                     os.path.join(OUT, "done.txt"))
    if args.variants and not os.path.exists(VARIANTS_F):
        vin = pd.read_csv(args.variants, keep_default_na=False)
        need = ["id", "set", "CHR", "pos", "ref_allele", "alt_allele"]
        missing = [c for c in need if c not in vin.columns]
        if missing:
            sys.exit(f"--variants file lacks columns {missing}")
        vin = vin.assign(alt_fixed=vin.get("alt_fixed", False))
        vin["CHR"], vin["pos"] = vin.CHR.astype(int), vin.pos.astype(int)
        os.makedirs(OUT, exist_ok=True)
        vin[need + ["alt_fixed"]].to_csv(VARIANTS_F, index=False)

    from alphagenome.data import genome
    from alphagenome.models import dna_client, variant_scorers as vs
    from alphagenome.models.dna_model import ModelVersion
    from alphagenome.models.dna_output import OutputType
    import alphagenome, importlib.metadata as md

    v = pd.read_csv(VARIANTS_F, keep_default_na=False) if args.variants else build_variants()
    os.makedirs(PARTS, exist_ok=True)
    done = set(open(DONE_F).read().split()) if os.path.exists(DONE_F) else set()
    todo = v[~v.id.isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"alphagenome {md.version('alphagenome')} | {time.strftime('%Y-%m-%d %H:%M')} | "
          f"{len(done)} done, {len(todo)} to go", flush=True)

    model = dna_client.create(get_key(), model_version=ModelVersion.ALL_FOLDS)
    half = dna_client.SEQUENCE_LENGTH_1MB // 2
    R = vs.RECOMMENDED_VARIANT_SCORERS
    scorers = [R["DNASE"], R["ATAC"], R["RNA_SEQ"]] + [
        vs.CenterMaskScorer(requested_output=o, width=501, aggregation_type=vs.AggregationType.DIFF_MEAN)
        for o in (OutputType.DNASE, OutputType.ATAC, OutputType.RNA_SEQ)]

    buf, buf_ids, failed = [], [], []
    part_no = len([f for f in os.listdir(PARTS) if f.endswith(".parquet")])
    t0 = time.time()

    def flush():
        nonlocal part_no, buf, buf_ids
        if not buf:
            return
        pd.concat(buf).to_parquet(os.path.join(PARTS, f"part_{part_no:05d}.parquet"), index=False)
        with open(DONE_F, "a") as f:
            f.write("\n".join(buf_ids) + "\n")
        part_no += 1
        buf, buf_ids = [], []

    for i, r in enumerate(todo.itertuples(), 1):
        chrom = f"chr{r.CHR}"
        # Keep the window exactly 1 Mb: near a chromosome start, shift it right instead of clipping
        # (a clipped window has an unsupported length and the call fails).
        start = max(0, r.pos - half)
        for attempt in range(MAX_RETRIES):
            try:
                res = model.score_variant(
                    interval=genome.Interval(chromosome=chrom, start=start, end=start + 2 * half),
                    variant=genome.Variant(chromosome=chrom, position=r.pos,
                                           reference_bases=r.ref_allele, alternate_bases=r.alt_allele),
                    variant_scorers=scorers)
                buf.append(compress(vs.tidy_scores(res), r.id))
                buf_ids.append(r.id)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    failed.append(r.id)
                    print(f"  FAILED {r.id}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                else:
                    time.sleep(10 * 2 ** attempt)
        if len(buf) >= FLUSH_EVERY:
            flush()
        if i == 1 or i % 250 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el / i:.2f}s/variant  ~{el / i * (len(todo) - i) / 60:.0f} min left"
                  f"  failed {len(failed)}", flush=True)
    flush()
    if failed:
        pd.Series(failed, name="id").to_csv(os.path.join(OUT, "failed.csv"), index=False)
    print(f"finished {time.strftime('%Y-%m-%d %H:%M')}; failed {len(failed)}", flush=True)


if __name__ == "__main__":
    main()
