"""
step02_gwas_loci.py

Builds the GWAS loci of the 15 traits in three disease domains (eye, kidney, metabolic) from the GWAS
Catalog full associations file (release 2024-12-19, GRCh38 positions).

  1. Keep associations with P < 5e-8 on chromosomes 1-22 with an rsID and a position.
  2. Assign associations to traits by keyword matching on MAPPED_TRAIT and DISEASE/TRAIT.
     The trait text is lower-cased before case-sensitive matching, so keywords that contain capital
     letters ("LDL", "HDL", "T2D", "BMI", "eGFR", "UACR", "ESRD") never match; those traits are
     captured through their lower-case keywords only (Supplementary Table 2, make_supp_table2.py).
  3. Per trait, keep the strongest association per rsID and clump by distance: walk the
     associations from the smallest P upwards and drop every association within 500 kb of one
     already kept.
  4. Per domain, combine the traits in the order listed in DISEASES, sort by P and keep the first
     row per rsID, so a locus shared by two traits of one domain is counted once, under the trait
     with the smaller P. (The sort is pandas' default quicksort; ties in P are resolved as in the
     published data, which this script reproduces exactly.)

Input:  data/external/gwas_catalog_associations_2024-12-19.tsv (paths.GWAS_CATALOG_TSV)
Output: data/inputs/gwas_loci/<trait>_loci_catalog.csv      (independent loci per trait)
        data/inputs/gwas_loci/<domain>_all_loci_catalog.csv (loci per domain)

    python scripts/step02_gwas_loci.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

WINDOW = 500_000
P_MAX = 5e-8
DOMAINS = ("eye", "kidney", "metabolic")

DISEASES = {
    # EYE
    "glaucoma":    {"cat": "eye", "kw": ["glaucoma", "open-angle", "intraocular pressure"]},
    "AMD":         {"cat": "eye", "kw": ["macular degeneration", "age-related macular"]},
    "myopia":      {"cat": "eye", "kw": ["myopia", "refractive error", "spherical equivalent",
                                         "high myopia"]},
    "IOP":         {"cat": "eye", "kw": ["intraocular pressure", "ocular pressure"]},
    "cataract":    {"cat": "eye", "kw": ["cataract", "lens opacity", "nuclear sclerosis"]},
    "diabetic_retinopathy": {"cat": "eye",
                             "kw": ["diabetic retinopathy", "retinopathy", "diabetic macular"]},
    # KIDNEY
    "CKD":         {"cat": "kidney", "kw": ["chronic kidney disease", "kidney disease",
                                            "renal disease", "end-stage renal", "ESRD"]},
    "eGFR":        {"cat": "kidney", "kw": ["glomerular filtration", "eGFR", "kidney function",
                                            "renal function", "serum creatinine"]},
    "UACR":        {"cat": "kidney", "kw": ["albumin", "urine albumin", "UACR",
                                            "albumin-creatinine", "proteinuria", "microalbuminuria"]},
    # METABOLIC
    "T2D":         {"cat": "metabolic", "kw": ["type 2 diabetes", "T2D",
                                               "diabetes mellitus type 2", "non-insulin"]},
    "hypertension": {"cat": "metabolic", "kw": ["hypertension", "blood pressure",
                                                "systolic blood pressure", "diastolic blood pressure",
                                                "pulse pressure"]},
    "BMI":         {"cat": "metabolic", "kw": ["body mass index", "BMI", "obesity",
                                               "overweight", "waist circumference", "body weight"]},
    "LDL":         {"cat": "metabolic", "kw": ["LDL", "low-density lipoprotein",
                                               "LDL cholesterol"]},
    "HDL":         {"cat": "metabolic", "kw": ["HDL", "high-density lipoprotein",
                                               "HDL cholesterol"]},
    "triglycerides": {"cat": "metabolic", "kw": ["triglyceride", "triglycerides",
                                                 "serum triglycerides"]},
}

COLUMNS = ["SNPS", "CHR_ID", "CHR_POS", "STRONGEST SNP-RISK ALLELE", "P-VALUE", "PVALUE_MLOG", "OR or BETA",
           "MAPPED_TRAIT", "DISEASE/TRAIT", "MAPPED_TRAIT_URI", "STUDY ACCESSION", "DATE",
           "INITIAL SAMPLE SIZE", "REPLICATION SAMPLE SIZE", "PUBMEDID"]


def load_catalog(path):
    cat = pd.read_csv(path, sep="\t", low_memory=False, on_bad_lines="skip", usecols=COLUMNS)
    cat["pval"] = pd.to_numeric(cat["P-VALUE"], errors="coerce")
    cat["CHR"] = pd.to_numeric(cat["CHR_ID"], errors="coerce")
    cat["pos"] = pd.to_numeric(cat["CHR_POS"], errors="coerce")
    cat["trait_text"] = (cat["MAPPED_TRAIT"].fillna("") + " | " + cat["DISEASE/TRAIT"].fillna("")).str.lower()
    sig = cat[(cat["pval"] < P_MAX) & cat["CHR"].between(1, 22) & cat["pos"].notna()].copy()
    sig["rsid"] = sig["SNPS"].astype(str).str.extract(r"(rs\d+)")[0]
    return sig.dropna(subset=["rsid", "CHR", "pos"])


def clump(df, window=WINDOW):
    """Distance clumping: keep the smallest P, drop everything within `window` of it, repeat."""
    df_s = df.sort_values("pval").copy()
    selected, excluded = [], set()
    for idx, row in df_s.iterrows():
        if idx in excluded:
            continue
        selected.append(idx)
        near = df_s[(df_s["CHR"] == row["CHR"]) & (df_s.index != idx) &
                    ((df_s["pos"] - row["pos"]).abs() <= window)].index
        excluded.update(near)
    return df_s.loc[selected].sort_values(["CHR", "pos"]).reset_index(drop=True)


def trait_loci(sig, disease, info):
    mask = pd.Series(False, index=sig.index)
    for kw in info["kw"]:
        mask |= sig["trait_text"].str.contains(kw, na=False)
    subset = sig[mask].sort_values("pval").drop_duplicates("rsid")
    ind = clump(subset) if len(subset) else pd.DataFrame(columns=subset.columns)
    ind["disease"] = disease
    ind["disease_cat"] = info["cat"]
    ind["locus_id"] = [f"{disease}_L{i + 1:04d}" for i in range(len(ind))]
    return ind


def main():
    if not os.path.exists(paths.GWAS_CATALOG_TSV):
        sys.exit(f"GWAS Catalog file not found: {paths.GWAS_CATALOG_TSV} (see README, Data sources)")
    out = paths.GWAS_LOCI_DIR
    os.makedirs(os.path.join(out, "by_trait"), exist_ok=True)
    sig = load_catalog(paths.GWAS_CATALOG_TSV)
    print(f"genome-wide significant associations on chr1-22 with an rsID: {len(sig):,}")

    per_trait = {}
    for disease, info in DISEASES.items():
        ind = trait_loci(sig, disease, info)
        ind.to_csv(os.path.join(out, "by_trait", f"{disease}_loci_catalog.csv"), index=False)
        per_trait[disease] = ind
        print(f"  {disease:22s} {info['cat']:9s} {len(ind):5d} independent loci")

    for domain in DOMAINS:
        traits = [d for d, info in DISEASES.items() if info["cat"] == domain]
        dom = pd.concat([per_trait[d] for d in traits], ignore_index=True)
        dom = dom.sort_values("pval").drop_duplicates("rsid")
        dom["source_disease"] = dom["disease"]
        dom.to_csv(os.path.join(out, f"{domain}_all_loci_catalog.csv"), index=False)
        print(f"{domain}: {len(dom):,} loci")


if __name__ == "__main__":
    main()
