"""
step40_build_immune_loci.py

Benchmark B2 (generalisation): lead loci for six immune-mediated diseases whose
heritability is strongly enriched in immune cells (Finucane 2018 Supp. Table 7 covers all six).

Source: NHGRI-EBI GWAS Catalog, REST API v2 (data release 2026-09-13, EFO v3.93.0, genes
GRCh38.p14; 1,192,604 associations / 7,817 studies in the release), one query per trait, the
exact trait only (no child traits):

    https://www.ebi.ac.uk/gwas/rest/api/v2/associations?efo_id=<ID>&size=100&page=<N>

    CD   Crohn disease                  MONDO_0005011   Finucane: "Crohn's disease"
    UC   ulcerative colitis             MONDO_0005101   Finucane: "Ulcerative colitis"
    RA   rheumatoid arthritis           MONDO_0008383   Finucane: "Rheumatoid Arthritis"
    MS   multiple sclerosis             MONDO_0005301   Finucane: "Multiple sclerosis"
    SLE  systemic lupus erythematosus   MONDO_0007915   Finucane: "Lupus"
    T1D  type 1 diabetes mellitus       MONDO_0005147   Finucane: "Type 1 Diabetes"

Raw pages are saved as data/external/gwas_catalog_immune/<TRAIT>_page<NNN>.json (downloaded
2026-09-21; sizes and totals are written to download_log.csv in the same folder, catalog metadata
to catalog_metadata.json). Pages already on disk are not downloaded again.
Downloaded 2026-09-21: CD 17 pages / 1,607 associations / 1.60 MB; UC 14 / 1,323 / 1.34 MB;
RA 32 / 3,183 / 3.10 MB; MS 11 / 1,039 / 0.96 MB; SLE 16 / 1,562 / 1.52 MB; T1D 15 / 1,465 / 1.40 MB
(total 105 pages, 9.9 MB); Ensembl slim cache 3.2 MB for 4,174 rsIDs.
Result: 1,154 loci (no sampling needed); 621 multiallelic in Ensembl, ALT chosen by frequency.

Allele resolution: Ensembl REST (GRCh38), POST https://rest.ensembl.org/variation/homo_sapiens?pops=1
with <= 200 rsIDs per request. Only the fields used here are kept, in
data/external/gwas_catalog_immune/ensembl_variation_slim.json (mapping, var_class, and per allele
the frequency in the global populations gnomADg:ALL, ALFA total, 1000G phase 3 ALL, and the max over
all populations).

Rules
  1. Associations with P < 5e-8 (from mantissa/exponent, so P below double precision still sorts),
     single-SNP (no haplotypes / interactions), rsID given.
  2. Resolve with Ensembl: var_class SNP, one mapping on chromosomes 1-22 (the scorer and the null
     background are autosomal). REF = first allele of the forward-strand allele string. ALT = the
     single non-reference allele if biallelic; at multiallelic sites the most frequent non-reference
     allele (gnomAD genomes, else ALFA, else 1000G, else max over any population). An allele with no
     observed frequency > 0 is never chosen; unresolved variants are dropped. (Resolution happens
     before clumping so that a dropped lead does not remove its locus.)
  3. Per trait, clump by distance: sort by P, keep a variant unless a kept one lies within 500 kb.
  4. Union over traits (a variant that leads for several traits lists them all, e.g. "CD;UC").
  5. If the union exceeds 2,000 loci, a fixed-seed (20260921) sample of 1,500, stratified by the
     trait in which the variant has its smallest P.
  The MHC is kept (immune loci live there); it contributes at most a handful of clumped leads per trait.

Output: data/immune_loci/immune_variants.csv  (id, set="immune", CHR, pos, ref_allele, alt_allele, trait)
        data/immune_loci/immune_trait_counts.csv

    python scripts/step40_build_immune_loci.py
"""
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

NEW = paths.NEW
RAW = paths.GWAS_CATALOG_IMMUNE
OUT = os.path.join(NEW, "data", "immune_loci")
ENS_F = os.path.join(RAW, "ensembl_variation_slim.json")

API = "https://www.ebi.ac.uk/gwas/rest/api/v2"
ENSEMBL = "https://rest.ensembl.org/variation/homo_sapiens?pops=1"
TRAITS = {  # code: (catalog id, catalog label, Finucane 2018 phenotype)
    "CD": ("MONDO_0005011", "Crohn disease", "Crohn's disease"),
    "UC": ("MONDO_0005101", "ulcerative colitis", "Ulcerative colitis"),
    "RA": ("MONDO_0008383", "rheumatoid arthritis", "Rheumatoid Arthritis"),
    "MS": ("MONDO_0005301", "multiple sclerosis", "Multiple sclerosis"),
    "SLE": ("MONDO_0007915", "systemic lupus erythematosus", "Lupus"),
    "T1D": ("MONDO_0005147", "type 1 diabetes mellitus", "Type 1 Diabetes"),
}
PAGE = 100
LOG10_P = math.log10(5e-8)
CLUMP_KB = 500
MAX_UNION, N_SAMPLE, SEED = 2000, 1500, 20260921
GLOBAL_POPS = ["gnomADg:ALL", "ALFA:SAMN10492705", "1000GENOMES:phase_3:ALL"]
AUTOSOMES = {str(i) for i in range(1, 23)}


def get(url, tries=6):
    for a in range(tries):
        try:
            r = requests.get(url, timeout=300)
            if r.status_code == 200:
                return r
            print(f"  HTTP {r.status_code} for {url}; retry", flush=True)
        except requests.RequestException as e:
            print(f"  {type(e).__name__} for {url}; retry", flush=True)
        time.sleep(10 * 2 ** a)
    raise RuntimeError(f"giving up on {url}")


# ── 1. download ───────────────────────────────────────────────────────────────
def download():
    os.makedirs(RAW, exist_ok=True)
    meta_f = os.path.join(RAW, "catalog_metadata.json")
    if not os.path.exists(meta_f):
        with open(meta_f, "w") as f:
            f.write(get(f"{API}/metadata").text)
    log = []
    for code, (efo, _, _) in TRAITS.items():
        page, total_pages = 0, None
        while total_pages is None or page < total_pages:
            url = f"{API}/associations?efo_id={efo}&size={PAGE}&page={page}"
            fn = os.path.join(RAW, f"{code}_page{page:03d}.json")
            if not os.path.exists(fn):
                txt = get(url).text
                json.loads(txt)  # fail loudly on a truncated page
                with open(fn, "w", encoding="utf-8") as f:
                    f.write(txt)
                time.sleep(0.5)
            d = json.load(open(fn, encoding="utf-8"))
            total_pages = d["page"]["totalPages"]
            log.append({"trait": code, "efo_id": efo, "page": page, "url": url,
                        "bytes": os.path.getsize(fn), "total_associations": d["page"]["totalElements"]})
            page += 1
        print(f"{code}: {total_pages} pages", flush=True)
    pd.DataFrame(log).to_csv(os.path.join(RAW, "download_log.csv"), index=False)


def parse():
    rows = []
    for code in TRAITS:
        for fn in sorted(glob.glob(os.path.join(RAW, f"{code}_page*.json"))):
            for a in json.load(open(fn, encoding="utf-8")).get("_embedded", {}).get("associations", []):
                if a.get("multi_snp_haplotype") or a.get("snp_interaction"):
                    continue
                snps = a.get("snp_allele") or []
                if len(snps) != 1 or not str(snps[0].get("rs_id", "")).startswith("rs"):
                    continue
                m, e = a.get("pvalue_mantissa"), a.get("pvalue_exponent")
                if m is None or e is None or m <= 0:
                    continue
                rows.append({"trait": code, "rsid": snps[0]["rs_id"].strip(), "log10p": math.log10(m) + e,
                             "accession": a.get("accession_id"), "association_id": a.get("association_id")})
    d = pd.DataFrame(rows)
    print(f"single-SNP rsID associations: {len(d):,}; P < 5e-8: {(d.log10p < LOG10_P).sum():,}")
    return d[d.log10p < LOG10_P]


# ── 2. Ensembl ────────────────────────────────────────────────────────────────
def slim(v):
    freq = {}
    for p in v.get("populations", []):
        a, f = p.get("allele"), p.get("frequency")
        if a is None or f is None:
            continue
        x = freq.setdefault(a, {"any_max": 0.0})
        x["any_max"] = max(x["any_max"], float(f))
        if p["population"] in GLOBAL_POPS:
            x[p["population"]] = float(f)
    return {"var_class": v.get("var_class"),
            "mappings": [{k: m.get(k) for k in ("seq_region_name", "start", "end", "allele_string", "strand",
                                                 "assembly_name")} for m in v.get("mappings", [])],
            "freq": freq}


def ensembl(ids):
    cache = json.load(open(ENS_F)) if os.path.exists(ENS_F) else {}
    todo = [i for i in ids if i not in cache]
    s = requests.Session()
    hdr = {"Content-Type": "application/json", "Accept": "application/json"}
    for k in range(0, len(todo), 200):
        batch = todo[k:k + 200]
        for a in range(6):
            try:
                r = s.post(ENSEMBL, headers=hdr, data=json.dumps({"ids": batch}), timeout=300)
            except requests.RequestException:
                time.sleep(10 * 2 ** a)
                continue
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 10)) + 1)
                continue
            if r.status_code == 200:
                break
            time.sleep(10 * 2 ** a)
        else:
            raise RuntimeError("Ensembl batch failed repeatedly")
        got = r.json()
        for i in batch:
            cache[i] = slim(got[i]) if i in got else None
        print(f"  Ensembl {min(k + 200, len(todo))}/{len(todo)}", flush=True)
        with open(ENS_F, "w") as f:
            json.dump(cache, f)
        time.sleep(1)
    return cache


COMP = str.maketrans("ACGT", "TGCA")


def resolve(v):
    """-> (CHR, pos, ref, alt, n_alleles) or a reason string."""
    if v is None:
        return "not in Ensembl"
    if v["var_class"] != "SNP":
        return f"var_class {v['var_class']}"
    maps = [m for m in v["mappings"] if m["seq_region_name"] in AUTOSOMES and m["assembly_name"] == "GRCh38"]
    if len(maps) != 1:
        return "no unique autosomal GRCh38 mapping"
    m = maps[0]
    alleles = m["allele_string"].split("/")
    if m["strand"] == -1:
        alleles = [a.translate(COMP) for a in alleles]
    if any(len(a) != 1 or a not in "ACGT" for a in alleles) or m["start"] != m["end"]:
        return "not a single-base SNV"
    ref, cands = alleles[0], [a for a in alleles[1:] if a != alleles[0]]
    fr = v["freq"]
    observed = [a for a in cands if fr.get(a, {}).get("any_max", 0) > 0]
    if not observed:
        return "no observed non-reference allele"
    if len(cands) == 1:
        alt = observed[0]
    else:
        alt = None
        for pop in GLOBAL_POPS + ["any_max"]:
            f = {a: fr[a].get(pop, 0) for a in observed}
            if max(f.values()) > 0:
                alt = max(f, key=f.get)
                break
    return (int(m["seq_region_name"]), int(m["start"]), ref, alt, len(alleles))


# ── 3-5. clump, union, sample ─────────────────────────────────────────────────
def clump(d):
    kept = []
    for r in d.sort_values("log10p").itertuples():
        if not any(k.CHR == r.CHR and abs(k.pos - r.pos) <= CLUMP_KB * 1000 for k in kept):
            kept.append(r)
    return pd.DataFrame(kept).drop(columns="Index")


def main():
    download()
    sig = parse()
    lead = sig.groupby(["trait", "rsid"]).log10p.min().reset_index()
    cache = ensembl(sorted(lead.rsid.unique()))
    res = {i: resolve(cache.get(i)) for i in lead.rsid.unique()}
    reasons = pd.Series({i: (r if isinstance(r, str) else "resolved") for i, r in res.items()}).value_counts()
    print("Ensembl resolution:\n" + reasons.to_string())
    ok = {i: r for i, r in res.items() if not isinstance(r, str)}
    lead = lead[lead.rsid.isin(ok)].copy()
    for j, c in enumerate(["CHR", "pos", "ref_allele", "alt_allele", "n_alleles"]):
        lead[c] = lead.rsid.map(lambda i: ok[i][j])

    counts = []
    clumped = []
    for code in TRAITS:
        c = clump(lead[lead.trait == code])
        clumped.append(c)
        counts.append({"trait": code, "catalog_id": TRAITS[code][0], "finucane_phenotype": TRAITS[code][2],
                       "sig_assoc": int((sig.trait == code).sum()),
                       "sig_rsids": int(sig.loc[sig.trait == code, "rsid"].nunique()),
                       "resolved_snvs": int((lead.trait == code).sum()), "clumped_leads": len(c)})
    cl = pd.concat(clumped, ignore_index=True)
    order = {t: i for i, t in enumerate(TRAITS)}
    u = (cl.sort_values("log10p").groupby("rsid")
         .agg(CHR=("CHR", "first"), pos=("pos", "first"), ref_allele=("ref_allele", "first"),
              alt_allele=("alt_allele", "first"), n_alleles=("n_alleles", "first"),
              best_trait=("trait", "first"), trait=("trait", lambda s: ";".join(sorted(s, key=order.get))))
         .reset_index())
    print(f"union of clumped leads: {len(u):,} variants; multiallelic: {(u.n_alleles > 2).sum()}")
    sampled = False
    if len(u) > MAX_UNION:
        rng = np.random.default_rng(SEED)
        n_t = (u.best_trait.value_counts() / len(u) * N_SAMPLE).round().astype(int)
        u = pd.concat([g.iloc[rng.choice(len(g), min(n_t[t], len(g)), replace=False)]
                       for t, g in u.groupby("best_trait")])
        sampled = True
    u = u.sort_values(["CHR", "pos"]).rename(columns={"rsid": "id"}).assign(set="immune")
    os.makedirs(OUT, exist_ok=True)
    u[["id", "set", "CHR", "pos", "ref_allele", "alt_allele", "trait"]].to_csv(
        os.path.join(OUT, "immune_variants.csv"), index=False)
    ct = pd.DataFrame(counts)
    ct["in_final_set"] = [int(u.trait.str.split(";").apply(lambda s, t=t: t in s).sum()) for t in TRAITS]
    ct["final_best_trait"] = [int((u.best_trait == t).sum()) for t in TRAITS]
    ct.to_csv(os.path.join(OUT, "immune_trait_counts.csv"), index=False)
    print(ct.to_string(index=False))
    print(f"final: {len(u):,} variants (sampled: {sampled}); shared by >1 trait: {(u.trait.str.contains(';')).sum()}")


if __name__ == "__main__":
    main()
