"""
make_supp_table2.py

Supplementary Table 2: the keywords that assigned GWAS Catalog associations to the 15 traits in the
analysis (DISEASES in step02_gwas_loci.py). The original code lower-cased the
trait text but matched keywords case-sensitively, so keywords containing capital letters never
matched; the table flags them.

    python scripts/make_supp_table2.py
"""
import ast
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step35_analysis as s35  # noqa: E402

SRC = os.path.join(HERE, "step02_gwas_loci.py")
OUT = os.path.join(s35.AN, "supp_table2_trait_keywords.csv")
DOMAIN = {"eye": "Eye", "kidney": "Kidney", "metabolic": "Metabolic"}


def main():
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "DISEASES")
    diseases = ast.literal_eval(node.value)
    rows = [{"domain": DOMAIN[v["cat"]], "trait": k,
             "keywords used": "; ".join(w for w in v["kw"] if w == w.lower()),
             "keywords that never matched (contain capitals)": "; ".join(w for w in v["kw"] if w != w.lower())}
            for k, v in diseases.items()]
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
