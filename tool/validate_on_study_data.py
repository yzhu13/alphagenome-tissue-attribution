"""
Check that the released tool reproduces the paper's numbers on the study data (DNase, recommended scorer).

Expected (from scripts/step35_analysis.py and Fig. 4a): eye rank among 12 tissues (`other` and the
misfiled choroid plexus track excluded) under calibrated DNase |quantile|: #2 / #5 / #3 for eye /
kidney / metabolic loci; and the eye's null-percentile ranks under the random null match step35 A5.

    python tool/validate_on_study_data.py
"""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(NEW, "scripts"))
import tissue_attribution as ta  # noqa: E402
import step35_analysis as s35  # noqa: E402  (study paths and track catalog only)

parts = sorted(glob.glob(os.path.join(s35.RUN, "parts", "*.parquet")))
d = pd.concat(pd.read_parquet(p, columns=["id", "scorer", "track_name", "quantile_score"],
                              filters=[("scorer", "==", "dnase_rec")]) for p in parts)
scores = pd.DataFrame({"variant_id": d.id, "track_name": d.track_name, "assay": "DNase",
                       "score": d.quantile_score.abs()})
tracks = s35.track_map().rename("tissue").rename_axis("track_name").reset_index()
variants = s35.read_variants()
null_ids = set(variants.id[variants.set == "null"])
dom = s35.domain_map(variants)
excl = ("other", "eye_misfiled")

print("Composition (check 1), eye row:")
print(ta.assay_composition(scores, tracks).loc[["eye"]].to_string())

null = scores[scores.variant_id.isin(null_ids)]
cal = ta.null_calibrate(scores, null)
rows = []
for dname in ["eye", "kidney", "metabolic"]:
    ids = set(dom.loc[dom.domain == dname, "id"])
    q = ta.tissue_scores(scores[scores.variant_id.isin(ids)], tracks, value="score", metric="mean", exclude=excl)
    s5 = ta.tissue_scores(scores[scores.variant_id.isin(ids)], tracks, value="score", metric="strong_fraction",
                          exclude=excl)
    e = ta.tissue_scores(cal[cal.variant_id.isin(ids)], tracks, value="null_pct", metric="mean", exclude=excl)
    rows.append({"domain": dname,
                 "eye rank, mean |q| (Fig 4a: 2/5/3)": int(q.set_index("tissue").loc["eye", "rank"]),
                 "eye rank, strong-effect fraction": int(s5.set_index("tissue").loc["eye", "rank"]),
                 "eye rank, null percentile": int(e.set_index("tissue").loc["eye", "rank"]),
                 "top tissue (mean |q|)": q.iloc[0].tissue})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
expected = [2, 5, 3]
ok = out["eye rank, mean |q| (Fig 4a: 2/5/3)"].tolist() == expected
print("MATCHES Fig 4a" if ok else f"MISMATCH: expected {expected}")
sys.exit(0 if ok else 1)
