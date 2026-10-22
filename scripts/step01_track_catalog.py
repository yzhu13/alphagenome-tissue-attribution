"""
step01_track_catalog.py

Lists every human output track of AlphaGenome and assigns each to one of 13 tissue categories.

A track's category comes from priority-ordered regular expressions applied to its biosample name and
track name; the first category with a matching pattern wins, and tracks matching none are "other".
Because the rules work on names, not on ontology relationships, a few tracks land in an unexpected
category. The eye patterns include "choroid", for example, so the choroid plexus epithelial cell
track (CL:0000706, a brain-ventricle cell type) is placed in the eye; the analyses treat it
separately (step35, "eye_misfiled").

Needs the AlphaGenome client (alphagenome==0.6.1) and an API key in ALPHAGENOME_API_KEY.

    python scripts/step01_track_catalog.py                 # query the API, write the catalogue
    python scripts/step01_track_catalog.py --recategorise  # re-apply the rules to the shipped catalogue
                                                           # and report agreement (no API call)

Output: data/inputs/track_catalog_full.csv, data/inputs/track_catalog_summary.csv
"""
import argparse
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

OUTPUT_TYPES = ["atac", "dnase", "chip_histone", "chip_tf", "rna_seq", "cage", "splice_junctions",
                "splice_site_usage", "contact_maps", "procap"]

# Priority-ordered rules: the first category with a matching pattern wins.
TISSUE_RULES = [
    ("eye", [
        r"retina", r"RPE", r"retinal.pigment", r"RGC", r"optic",
        r"ocular", r"cornea", r"sclera", r"trabecular.meshwork",
        r"lens\b", r"vitreous", r"choroid", r"photoreceptor",
        r"\brod\b", r"\bcone\b", r"Muller.glia", r"iris", r"ciliary",
        r"ARPE", r"iPS.RPE", r"retinal.organoid"
    ]),
    ("brain", [
        r"\bbrain\b", r"neuron", r"cortex", r"hippocampus", r"cerebellum",
        r"astrocyte", r"oligodendrocyte", r"microglia", r"\bneural\b",
        r"neuronal", r"\bCNS\b", r"cerebral", r"frontal.lobe",
        r"temporal.lobe", r"substantia.nigra", r"caudate", r"putamen",
        r"thalamus", r"amygdala", r"hypothalamus", r"spinal.cord",
        r"dorsal.root.ganglion", r"neuroblast", r"motor.neuron",
        r"interneuron", r"GABAergic", r"glutamatergic", r"dopaminergic",
        r"iPSC.neuron", r"neuronal.stem"
    ]),
    ("heart_vascular", [
        r"\bheart\b", r"cardiac", r"cardiomyocyte", r"\baorta\b", r"aortic",
        r"endotheli", r"\bvascular\b", r"\bartery\b", r"arterial",
        r"\bvein\b", r"coronary", r"myocardium", r"ventricle",
        r"\batrium\b", r"atrial", r"HUVEC", r"smooth.muscle",
        r"pericyte", r"HMEC", r"iPS.CM", r"iPSC.cardiomyocyte"
    ]),
    ("pancreas_islet", [
        r"pancrea", r"\bislet\b", r"beta.cell", r"alpha.cell",
        r"acinar", r"insulin", r"glucagon", r"exocrine.pancrea"
    ]),
    ("liver", [
        r"\bliver\b", r"hepat", r"hepatocyte", r"\bHepG2\b",
        r"HuH", r"biliary", r"liver.organoid"
    ]),
    ("immune", [
        r"\bT.cell\b", r"\bB.cell\b", r"\bNK.cell\b", r"macrophage",
        r"monocyte", r"dendritic.cell", r"neutrophil", r"lymphocyte",
        r"eosinophil", r"basophil", r"mast.cell", r"plasma.cell",
        r"thymocyte", r"\bspleen\b", r"\bthymus\b", r"bone.marrow",
        r"\bPBMC\b", r"leukocyte", r"\bCD4\b", r"\bCD8\b",
        r"\bTreg\b", r"germinal.center", r"natural.killer",
        r"lymphoid", r"myeloid", r"erythroid", r"megakaryocyte"
    ]),
    ("lung", [
        r"\blung\b", r"pulmonary", r"bronch", r"alveol", r"pneumocyte",
        r"\bairway\b", r"\bA549\b", r"\bBEAS\b", r"NHBE"
    ]),
    ("kidney", [
        r"\bkidney\b", r"\brenal\b", r"\bnephron\b", r"podocyte",
        r"tubul", r"glomerul", r"collecting.duct", r"HEK293"
    ]),
    ("muscle_adipose", [
        r"\bmuscle\b", r"skeletal.muscle", r"myoblast", r"myotube",
        r"adipose", r"adipocyte", r"\bfat\b", r"subcutaneous.fat",
        r"SGBS", r"3T3"
    ]),
    ("gut", [
        r"intestin", r"\bcolon\b", r"\brectum\b", r"duodenum",
        r"\bileum\b", r"jejunum", r"gastric", r"\bstomach\b",
        r"esophag", r"\bCaco\b", r"\bHCT\b", r"HT29", r"sigmoid"
    ]),
    ("reproductive", [
        r"\bovary\b", r"\btestis\b", r"\buterus\b", r"prostate",
        r"\bbreast\b", r"mammary", r"placenta", r"endometrium", r"cervix"
    ]),
    ("skin", [
        r"\bskin\b", r"keratinocyte", r"fibroblast", r"melanocyte",
        r"\bdermis\b", r"epidermis", r"foreskin", r"HaCaT"
    ]),
]


def categorise(row):
    text = f"{row.get('biosample_name', '')} {row.get('name', '')}".lower()
    for category, patterns in TISSUE_RULES:
        if any(re.search(p, text, re.IGNORECASE) for p in patterns):
            return category
    return "other"


def fetch_catalog():
    from alphagenome.models import dna_client
    from alphagenome.models.dna_model import ModelVersion, Organism

    key = os.environ.get("ALPHAGENOME_API_KEY")
    if not key:
        sys.exit("Set ALPHAGENOME_API_KEY before running this step.")
    model = dna_client.create(key, model_version=ModelVersion.ALL_FOLDS)
    metadata = model.output_metadata(organism=Organism.HOMO_SAPIENS)
    frames = []
    for ot in OUTPUT_TYPES:
        df = getattr(metadata, ot).copy()
        df["output_type"] = ot
        if "biosample_name" not in df.columns:
            df["biosample_name"] = ""
        frames.append(df)
    full = pd.concat(frames, ignore_index=True, sort=False)
    full["biosample_name"] = full["biosample_name"].fillna("").astype(str)
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recategorise", action="store_true",
                    help="re-apply the rules to data/inputs/track_catalog_full.csv and report agreement")
    args = ap.parse_args()

    if args.recategorise:
        cat = pd.read_csv(paths.TRACK_CATALOG)
        cat["biosample_name"] = cat["biosample_name"].fillna("").astype(str)
        again = cat.apply(categorise, axis=1)
        same = (again == cat["tissue_category"]).mean()
        print(f"{len(cat)} tracks; rules reproduce the stored category for {same:.1%}")
        sys.exit(0 if same == 1 else 1)

    full = fetch_catalog()
    full["tissue_category"] = full.apply(categorise, axis=1)
    os.makedirs(paths.INPUTS, exist_ok=True)
    full.to_csv(paths.TRACK_CATALOG, index=False)
    xtab = pd.crosstab(full["tissue_category"], full["output_type"])
    xtab.to_csv(os.path.join(paths.INPUTS, "track_catalog_summary.csv"))
    print(f"{len(full)} tracks -> {paths.TRACK_CATALOG}")
    print(xtab.to_string())


if __name__ == "__main__":
    main()
