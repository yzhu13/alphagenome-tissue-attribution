"""
run_pipeline.py

Runs the analysis end to end, in order. Each stage can be run on its own.

  stage      needs                                      steps
  ---------  -----------------------------------------  ------------------------------------------------
  inputs     GWAS Catalog file (data/external)           02 GWAS loci, 03 alleles (Ensembl cache),
             (step01 needs the AlphaGenome API, so it      04 at-risk list, 05 locus table
             runs only with --with-api)
  nulls      FinnGen R10 hypertension summary            33 random null, 39 distance-matched null
             statistics, Ensembl GTF (data/external)
  scores     AlphaGenome API key; Enformer on CPU        34 GWAS + random null, 34 matched null,
             (many hours)                                  40 immune loci + 34 immune, 47 Enformer,
                                                           50 re-queried metabolic loci + 34 for them
  analysis   score tables from `scores` (or Zenodo)      35, 36, 38, 39b, 40b, 42, 43, 45, 46, 48, 49, 50,
                                                           Supplementary Table 2
  figures    tables from `analysis`                      Fig. 1-5, Supplementary Figs. 1-6
  audit      tables from `analysis`, a manuscript text   44 (every number quoted in the paper)

    python scripts/run_pipeline.py analysis figures     # rerun everything downstream of the scores
    python scripts/run_pipeline.py all --with-api       # everything, including AlphaGenome calls
    python scripts/run_pipeline.py analysis --dry-run   # print the commands only
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

STAGES = {
    "inputs": [["step01_track_catalog.py"], ["step02_gwas_loci.py"], ["step03_resolve_alleles.py"],
               ["step04_multiallelic_audit.py", "--list-only"], ["step05_locus_table.py"]],
    "nulls": [["step33_build_null_variants.py"], ["step39_build_matched_null.py"]],
    "scores": [["step34_full_rescoring.py"],
               ["step34_full_rescoring.py", "--variants", "data/null_matched/null_matched_variants.csv",
                "--out", "data/null_matched_scoring"],
               ["step40_build_immune_loci.py"],
               ["step34_full_rescoring.py", "--variants", "data/immune_loci/immune_variants.csv",
                "--out", "data/immune_scoring"],
               ["step47_enformer_scoring.py"],
               ["step50_metabolic_batch_sensitivity.py", "prepare"],
               ["step34_full_rescoring.py", "--variants", "data/metabolic_batch/variants_to_score.csv",
                "--out", "data/metabolic_batch_scoring"]],
    "analysis": [["step35_analysis.py"], ["step36_benchmark_expected_tissue.py"], ["step38_benchmark_sldsc.py"],
                 ["step39b_matched_null_analysis.py"], ["step40b_immune_analysis.py"], ["step42_uncertainty.py"],
                 ["step43_sensitivity.py"], ["step45_aggregation_variants.py"], ["step46_grouping_sensitivity.py"],
                 ["enformer_tissue_map.py"], ["step48_enformer_analysis.py"], ["step49_recommended_pipeline_check.py"],
                 ["step50_metabolic_batch_sensitivity.py", "analyze"], ["make_supp_table2.py"]],
    "figures": [["fig_v2_fig1.py"], ["fig_v2_main.py"], ["fig_v2_supp.py"]],
    "audit": [["step44_number_audit.py"]],
}
API_STEPS = {"step01_track_catalog.py", "step34_full_rescoring.py", "step40_build_immune_loci.py",
             "step47_enformer_scoring.py"}
API_COMMANDS = {("step50_metabolic_batch_sensitivity.py", "prepare")}  # Ensembl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=list(STAGES) + ["all"])
    ap.add_argument("--with-api", action="store_true",
                    help="also run steps that call AlphaGenome, the GWAS Catalog REST API or download sequence")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stages = list(STAGES) if "all" in args.stages else args.stages
    root = os.path.dirname(HERE)
    for stage in stages:
        for cmd in STAGES[stage]:
            if (cmd[0] in API_STEPS or tuple(cmd[:2]) in API_COMMANDS) and not args.with_api:
                print(f"[{stage}] skip {' '.join(cmd)} (needs --with-api)")
                continue
            full = [sys.executable, "-u", os.path.join(HERE, cmd[0])] + cmd[1:]
            print(f"[{stage}] {' '.join(cmd)}", flush=True)
            if args.dry_run:
                continue
            t = time.time()
            r = subprocess.run(full, cwd=root)
            if r.returncode:
                sys.exit(f"[{stage}] {cmd[0]} failed with exit code {r.returncode}")
            print(f"[{stage}] {cmd[0]} done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
