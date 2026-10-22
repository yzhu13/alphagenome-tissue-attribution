"""
step38_benchmark_sldsc.py

Benchmark B1, quantitative version. Does the tissue ranking from AlphaGenome
agree with independent heritability enrichment, and does calibration improve the agreement?

Reference: Finucane et al. 2018 Nat Genet, Supplementary Table 7 (S-LDSC on Roadmap + EN-TEx
chromatin annotations: DNase and five histone marks, 111 tissues/cell types, 48 traits).
File: data/external/Finucane2018_MOESM3_ESM.xlsx (Springer static supplementary content,
downloaded 2026-09-21).

Traits in both studies: BMI, LDL, HDL, triglycerides, type 2 diabetes, hypertension.

Mapping: Finucane's 111 tissues are mapped to our tissue categories following the same rules as our
track catalog (fibroblasts -> skin except lung fibroblasts -> lung; breast / mammary / placenta and
reproductive organs -> reproductive; adipose and muscle -> muscle_adipose). Adrenal, thyroid and
tibial nerve fall in our catch-all "other" and are dropped, as is eye (absent from the reference).
11 categories remain. A category's S-LDSC score is the MEAN over its annotations of the normalized
coefficient (tau*; primary). The maximum is reported as a sensitivity check only: it rises with the
number of annotations in a category (Spearman up to 0.45 here; immune has 27 annotations, kidney 1),
the same composition bias this paper is about. With the mean, the reference reproduces the textbook
tissue for every trait (BMI brain; LDL/HDL/TG liver; T2D islet; hypertension kidney).

Per trait: Spearman correlation, across the 11 categories, between the S-LDSC category score and
each AlphaGenome method's tissue score for that trait's loci.

    python scripts/step38_benchmark_sldsc.py
"""
import os
import sys

import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step35_analysis as s35  # noqa: E402
from step36_benchmark_expected_tissue import METHODS, NOT_TISSUES  # noqa: E402

REF = s35.paths.FINUCANE_XLSX
OUT = os.path.join(s35.NEW, "reports", "benchmark_sldsc.md")
TRAITS = {"BMI": "BMI", "LDL": "LDL", "HDL": "HDL", "triglycerides": "Triglycerides",
          "T2D": "Type 2 diabetes", "hypertension": "Hypertension"}
FIG_CAT = {"Blood/Immune": "immune", "CNS": "brain", "Cardiovascular": "heart_vascular", "Digestive": "gut",
           "Liver": "liver", "Pancreas": "pancreas_islet", "Adipose": "muscle_adipose"}


def category(name, fig_cat):
    n = name.lower()
    if fig_cat in FIG_CAT:
        return FIG_CAT[fig_cat]
    if fig_cat == "Musculoskeletal/Connective":
        if "lung" in n:
            return "lung"
        if "fibroblast" in n or "dermal" in n:
            return "skin"
        return "muscle_adipose"
    # "Other"
    if "kidney" in n:
        return "kidney"
    if "lung" in n:
        return "lung"
    if any(k in n for k in ("keratinocyte", "melanocyte", "skin")):
        return "skin"
    if any(k in n for k in ("breast", "mammary", "hmec", "placenta", "ovary", "prostate", "testis", "uterus", "vagina")):
        return "reproductive"
    return None  # adrenal, thyroid, nerve -> our catch-all "other": dropped


def reference():
    d = pd.read_excel(REF, "Supp Table 7", header=1)
    d["category"] = [category(t, c) for t, c in zip(d["Tissue/cell type"], d["Tissue category for figures"])]
    d = d.dropna(subset=["category"])
    g = d.groupby(["Phenotype", "category"]).agg(tau=("Coefficient (normalized)", "mean"),
                                                   tau_max=("Coefficient (normalized)", "max"),
                                                   n_annot=("Coefficient (normalized)", "size"))
    return g.reset_index(), d


def main():
    ref, mapped = reference()
    lt, _ = s35.aggregate()
    frames = {"lt": lt, "pub": s35.as_published(lt)}
    traits = pd.read_csv(s35.LOCI)[["rsid", "source_disease"]].drop_duplicates()
    rows, tables = [], {}
    for ours, theirs in TRAITS.items():
        r = ref[ref.Phenotype == theirs].set_index("category")
        ids = traits.loc[traits.source_disease == ours, "rsid"]
        for label, fk, scorer, col in METHODS:
            x = frames[fk]
            x = x[(x.scorer == scorer) & x.id.isin(ids) & ~x.tissue.isin(NOT_TISSUES | {"eye"})]
            if x.empty:
                continue
            m = x.groupby("tissue")[col].mean()
            both = pd.concat([m.rename("ag"), r.tau, r.tau_max], axis=1, join="inner")
            rows.append({"trait": ours, "method": label, "n_loci": x.id.nunique(), "n_categories": len(both),
                         "rho_tau": spearmanr(both.ag, both.tau).correlation,
                         "rho_tau_max": spearmanr(both.ag, both.tau_max).correlation,
                         "sldsc_top": both.tau.idxmax(), "method_top": both.ag.idxmax(),
                         "rank_of_sldsc_top": int(both.ag.rank(ascending=False)[both.tau.idxmax()])})
            tables[(ours, label)] = both
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(s35.AN, "B1_sldsc_benchmark.csv"), index=False)

    L = ["# B1 (quantitative): agreement with independent S-LDSC tissue enrichment", "",
         "Generated by `scripts/step38_benchmark_sldsc.py`. Reference: Finucane et al. 2018 *Nat Genet* "
         "Supp. Table 7 (S-LDSC, Roadmap + EN-TEx chromatin, all marks). Category score = MEAN normalized "
         "coefficient (tau*) over the category's annotations (primary); `rho_tau_max` uses the max, which is "
         "biased toward categories with many annotations (sensitivity only). "
         "Spearman across the shared tissue categories (eye and `other` excluded).", "",
         f"Reference annotations mapped: {mapped['Tissue/cell type'].nunique()} tissues/cell types into "
         f"{mapped.category.nunique()} categories.", ""]
    if res.empty:
        L.append("No GWAS loci for these traits scored yet.")
    else:
        piv = res.pivot_table(index=["trait", "n_loci"], columns="method", values="rho_tau")
        piv = piv[[m[0] for m in METHODS if m[0] in piv.columns]]
        L += ["## Spearman rho with S-LDSC (tau*), per trait", "", piv.round(2).to_markdown(), ""]
        summ = res.groupby("method").agg(mean_rho_tau=("rho_tau", "mean"), mean_rho_tau_max=("rho_tau_max", "mean"),
                                         mean_rank_of_sldsc_top=("rank_of_sldsc_top", "mean"),
                                         top_agrees=("method_top", lambda s: int((s == res.loc[s.index, "sldsc_top"]).sum())),
                                         n_traits=("trait", "size"))
        L += ["## Summary", "", summ.round(2).to_markdown(), "",
              "`mean_rank_of_sldsc_top`: where the method ranks the category that S-LDSC ranks first (1 = best).", ""]
        raw_lab = METHODS[0][0]
        for label in [m[0] for m in METHODS[1:]]:
            a = piv.get(raw_lab)
            b = piv.get(label)
            if a is not None and b is not None and len(a.dropna()) >= 5:
                try:
                    p = wilcoxon(b, a).pvalue
                    L.append(f"- {label} vs raw: rho higher in {(b > a).sum()}/{len(a)} traits (Wilcoxon p={p:.2g})")
                except ValueError:
                    pass
        tops = res.pivot_table(index=["trait", "sldsc_top"], columns="method", values="method_top", aggfunc="first")
        L += ["", "## Top category: S-LDSC vs each method", "", tops.to_markdown(), ""]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
