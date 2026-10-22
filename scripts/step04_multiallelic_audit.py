"""
step04_multiallelic_audit.py

Checks the alternate alleles chosen by step03 at multiallelic sites.

Where step03 took the ALT from the GWAS risk allele, that allele was observed in a GWAS. Where it took
the first other allele of the Ensembl allele string at a site with more than two catalogued alleles,
the chosen allele may never have been observed in any population. Those loci are "at risk":
first record per rsID in variant_allele_resolution.csv (domains in the order eye, kidney, metabolic)
with resolution ensembl_alt* and a multiallelic allele string (2,044 loci).

For each at-risk locus, Ensembl population frequencies (variation/homo_sapiens?pops=1) are queried
and the scored ALT is classified as
  ok                           seen with a non-zero frequency in at least one population
  SUSPECT_alt_never_observed   no population reports it (step34 then scores the most frequent
                               observed non-reference allele instead)
  unknown_no_population_data   Ensembl has no frequency data for the rsID

The shipped audit (data/inputs/multiallelic_alt_audit.csv) was queried on 2026-08-03; rerunning
queries the current Ensembl release.

Input:  data/inputs/variant_allele_resolution.csv (step03)
Output: data/inputs/at_risk_multiallelic_loci.csv, data/inputs/multiallelic_alt_audit.csv

    python scripts/step04_multiallelic_audit.py --list-only   # rebuild the at-risk list only
    python scripts/step04_multiallelic_audit.py               # list + Ensembl audit
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

ENSEMBL = "https://rest.ensembl.org/variation/homo_sapiens?pops=1"
BATCH = 100
PAUSE = 0.4
COLS = ["rsid", "CHR", "pos", "ref_allele", "alt_allele", "risk_allele", "ensembl_allele_string", "resolution"]


def at_risk_loci():
    v = pd.read_csv(paths.ALLELE_RESOLUTION)
    first = v.drop_duplicates("rsid", keep="first")
    risky = first.resolution.astype(str).str.startswith("ensembl_alt") & (first.multiallelic == True)  # noqa: E712
    return first.loc[risky, COLS].reset_index(drop=True)


def fetch(ids, tries=4):
    body = json.dumps({"ids": list(ids)}).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(ENSEMBL, data=body, headers={
                "Content-Type": "application/json", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:  # rate limited
                time.sleep(int(e.headers.get("Retry-After", 5)) + 1)
                continue
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def alleles_with_frequency(record):
    """Alleles with a non-zero frequency in at least one population, with the largest frequency."""
    best = {}
    for p in record.get("populations", []):
        a, f = p.get("allele"), p.get("frequency")
        if a is not None and f is not None and f > 0:
            best[a] = max(best.get(a, 0.0), float(f))
    return best


def audit(target):
    payload = {}
    ids = target.rsid.tolist()
    for i in range(0, len(ids), BATCH):
        payload.update(fetch(ids[i:i + BATCH]))
        print(f"  {min(i + BATCH, len(ids)):>5}/{len(ids)} fetched", flush=True)
        time.sleep(PAUSE)
    rows = []
    for _, t in target.iterrows():
        rec = payload.get(t.rsid)
        if not rec:
            rows.append({**t.to_dict(), "verdict": "not_returned", "alt_freq": None, "alleles_seen": ""})
            continue
        freqs = alleles_with_frequency(rec)
        verdict = ("unknown_no_population_data" if not freqs else
                   "ok" if t.alt_allele in freqs else "SUSPECT_alt_never_observed")
        rows.append({**t.to_dict(), "verdict": verdict, "alt_freq": freqs.get(t.alt_allele),
                     "alleles_seen": ",".join(f"{a}:{f:.3f}" for a, f in sorted(freqs.items()))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-only", action="store_true", help="rebuild the at-risk list without querying Ensembl")
    args = ap.parse_args()
    target = at_risk_loci()
    target.to_csv(paths.AT_RISK_MULTIALLELIC, index=False)
    print(f"at-risk multiallelic loci: {len(target)} -> {paths.AT_RISK_MULTIALLELIC}")
    if args.list_only:
        return
    res = audit(target)
    res.to_csv(paths.MULTIALLELIC_AUDIT, index=False)
    print(res.verdict.value_counts().to_string())
    print(f"-> {paths.MULTIALLELIC_AUDIT}")


if __name__ == "__main__":
    main()
