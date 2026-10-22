"""
step03_resolve_alleles.py

Assigns GRCh38 reference and alternate alleles to every GWAS locus, using the allele string that
Ensembl reports for the rsID (REST endpoint variation/human, first mapping on a primary chromosome).

  ref_allele = first allele of the Ensembl allele string
  alt_allele = the GWAS risk allele, if it is a valid base, differs from REF and is in the allele
               string                                             (resolution "risk_allele");
               otherwise the first other single base in the string ("ensembl_alt", or
               "ensembl_alt_risk_eq_ref" when a risk allele was reported but equals REF)
  loci without an Ensembl record, with a non-SNV reference or with no single-base ALT are kept but
  flagged ("unresolved", "unresolved_non_snv", "unresolved_no_alt") and are not scored.

At multiallelic sites the first listed ALT can be an allele that was catalogued but never observed;
step04 audits those choices and step34 replaces them.

The Ensembl answers are cached per domain in data/inputs/ensembl_allele_strings.json, so the shipped
tables are rebuilt offline and exactly. The cache records what Ensembl returned in June 2026. For the
metabolic domain the first batch of 200 rsIDs (the 200 most significant metabolic loci) returned
nothing, so 199 of those loci are "unresolved" in the published tables and were not analysed.
--refresh queries Ensembl again for every rsID without an answer; this resolves those loci but
changes the locus set, and the current Ensembl release may differ from the one used for the paper.

Input:  data/inputs/gwas_loci/<domain>_all_loci_catalog.csv (step02)
Output: data/inputs/resolved_loci/<domain>_loci_resolved.csv
        data/inputs/variant_allele_resolution.csv (all domains, with a multiallelic flag)

    python scripts/step03_resolve_alleles.py            # from the cache
    python scripts/step03_resolve_alleles.py --refresh  # query Ensembl for rsIDs without an answer
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

DOMAINS = ("eye", "kidney", "metabolic")
ENSEMBL_URL = "https://rest.ensembl.org/variation/human?content-type=application/json"
BATCH = 200
CACHE = os.path.join(paths.INPUTS, "ensembl_allele_strings.json")


def parse_risk_allele(x):
    """'rs123-A' -> 'A'; 'rs123-?' -> None. The test is a substring test on "ACGT", as in the code
    that produced the published tables (so 'rs123-' gives '', which counts as no risk allele)."""
    if pd.isna(x):
        return None
    parts = str(x).split("-")
    return parts[-1] if len(parts) >= 2 and parts[-1] in "ACGT" else None


def _post(chunk, tries=3, timeout=120):
    """One POST to Ensembl; None if every attempt fails."""
    req = urllib.request.Request(ENSEMBL_URL, data=json.dumps({"ids": chunk}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # time-out, network hiccup or rate limit
            print(f"  retry {attempt} for {len(chunk)} ids ({e})", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    return None


def _query(chunk):
    """POST a batch; if it keeps failing, split it in half and query each half (a single slow rsID
    can make a whole batch time out, which is what lost 199 metabolic loci in June 2026)."""
    data = _post(chunk)
    if data is not None:
        return data
    if len(chunk) == 1:
        print(f"  no answer for {chunk[0]}", file=sys.stderr)
        return {}
    half = len(chunk) // 2
    return {**_query(chunk[:half]), **_query(chunk[half:])}


def query_ensembl(rsids):
    """rsID -> allele string of its first mapping on a primary chromosome."""
    out = {}
    for i in range(0, len(rsids), BATCH):
        chunk = rsids[i:i + BATCH]
        data = _query(chunk)
        for rid in chunk:
            for m in (data.get(rid) or {}).get("mappings") or []:
                region, astr = m.get("seq_region_name"), m.get("allele_string")
                if region and astr and "/" in astr and (region.isdigit() or region in ("X", "Y", "MT")):
                    out[rid] = astr
                    break
        print(f"  Ensembl: {min(i + BATCH, len(rsids))}/{len(rsids)}")
        time.sleep(1)
    return out


def resolve(df, allele_map):
    df = df.copy()
    df["risk_allele"] = df["STRONGEST SNP-RISK ALLELE"].apply(parse_risk_allele)
    ref_l, alt_l, astr_l, res_l = [], [], [], []
    for _, row in df.iterrows():
        astr = allele_map.get(row["rsid"])
        ref = alt = None
        if not astr:
            res = "unresolved"
        else:
            alleles = astr.split("/")
            risk = row["risk_allele"]
            if alleles[0] not in "ACGT":  # same test as the original code: a multi-base REF is not an SNV
                res = "unresolved_non_snv"
            elif risk and risk != alleles[0] and risk in alleles:
                ref, alt, res = alleles[0], risk, "risk_allele"
            else:
                cands = [a for a in alleles[1:] if a in "ACGT" and a != alleles[0]]
                if cands:
                    ref, alt = alleles[0], cands[0]
                    res = "ensembl_alt" if not risk else "ensembl_alt_risk_eq_ref"
                else:
                    res = "unresolved_no_alt"
        ref_l.append(ref)
        alt_l.append(alt)
        astr_l.append(astr)
        res_l.append(res)
    df["ref_allele"], df["alt_allele"] = ref_l, alt_l
    df["ensembl_allele_string"], df["resolution"] = astr_l, res_l
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="query Ensembl for rsIDs missing from the cache")
    args = ap.parse_args()

    catalogs = {d: pd.read_csv(os.path.join(paths.GWAS_LOCI_DIR, f"{d}_all_loci_catalog.csv"))
                .dropna(subset=["rsid", "CHR", "pos"]) for d in DOMAINS}
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}  # domain -> rsID -> allele string
    if args.refresh or not cache:
        for d, df in catalogs.items():
            known = cache.setdefault(d, {})
            todo = [r for r in df.rsid.unique() if r not in known]
            print(f"{d}: querying Ensembl for {len(todo)} rsIDs")
            known.update(query_ensembl(todo))
        with open(CACHE, "w") as fh:
            json.dump({d: dict(sorted(m.items())) for d, m in cache.items()}, fh, indent=0)

    os.makedirs(paths.RESOLVED_DIR, exist_ok=True)
    prov = []
    for d, df in catalogs.items():
        out = resolve(df, cache.get(d, {}))
        out.to_csv(os.path.join(paths.RESOLVED_DIR, f"{d}_loci_resolved.csv"), index=False)
        ok = out.resolution.isin(["risk_allele", "ensembl_alt", "ensembl_alt_risk_eq_ref"]).sum()
        print(f"{d}: {ok}/{len(out)} resolved  {out.resolution.value_counts().to_dict()}")
        p = out[["rsid", "CHR", "pos", "STRONGEST SNP-RISK ALLELE", "risk_allele", "ref_allele", "alt_allele",
                 "ensembl_allele_string", "resolution", "DISEASE/TRAIT", "pval"]].copy()
        p.insert(0, "domain", d)
        p["multiallelic"] = p.ensembl_allele_string.fillna("").map(  # three or more single-base alleles
            lambda s: sum(a in ("A", "C", "G", "T") for a in s.split("/")) >= 3)
        prov.append(p)
    pd.concat(prov, ignore_index=True).to_csv(paths.ALLELE_RESOLUTION, index=False)
    print(f"-> {paths.RESOLVED_DIR}, {paths.ALLELE_RESOLUTION}")


if __name__ == "__main__":
    main()
