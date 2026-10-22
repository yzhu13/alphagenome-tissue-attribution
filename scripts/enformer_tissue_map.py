"""
enformer_tissue_map.py

Assigns Enformer's 5,313 human tracks (targets_human.txt) to the tissue categories used for AlphaGenome,
by keyword matching on the biosample description. Rules, applied in order (first match wins):
  - cancer / immortalised cell lines, stem cells, in-vitro differentiated cells and treated samples -> other
    (as for AlphaGenome, where the eye category holds primary eye cells and tissues only)
  - choroid plexus -> brain; extraocular muscle -> muscle_adipose
  - otherwise the first tissue whose keywords occur in the description.
Writes data/enformer/track_tissue_map.csv (index, assay, biosample, tissue) for inspection.

    python scripts/enformer_tissue_map.py
"""
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step35_analysis as s35  # noqa: E402

TARGETS = os.path.join(s35.NEW, "data", "enformer", "targets_human.txt")
OUT = os.path.join(s35.NEW, "data", "enformer", "track_tissue_map.csv")

OTHER_FIRST = [
    r"cell line", r"\bline\b", r"hepg2", r"k562", r"gm\d", r"hela", r"mcf-?7", r"a549", r"hek293", r"imr-90",
    r"sk-n-", r"hct116", r"panc1", r"pc-3", r"pc-9", r"lncap", r"22rv1", r"c4-2b", r"vcap", r"caco-2",
    r"weri-rb", r"y79", r"be2c", r"kelly", r"karpas", r"loucy", r"dohh2", r"ocly", r"hap-1", r"hues",
    r"h1\b", r"h7\b", r"h9\b", r"es cell", r"stem cell", r"ips", r"induced pluripotent", r"embryonic stem",
    r"originated from", r"derived from", r"differentiated", r"treated", r"carcinoma", r"sarcoma", r"leukemia",
    r"lymphoma", r"myeloma", r"blastoma", r"cancer", r"tumor", r"adenocarcinoma", r"melanoma", r"glioma",
    r"mesothelioma", r"wi38", r"bj\b", r"ag0\d", r"hff", r"hacat", r"t47d", r"ishikawa", r"ht1080", r"rko",
    r"dnd-41", r"jurkat", r"kms-11", r"mm\.1s", r"nci-h", r"hl-60", r"nb4", r"calu3", r"thp-1", r"u-?937",
    r"transformed", r"immortal", r"hues\d", r"ntera", r"hepatocellular", r"hek",
]
TISSUE_KW = [
    ("eye", ["retina", "eye,", "eye female", "eye embryo", "eye - vitreous", "ciliary", "lens epithelial", "corneal", "cornea", "trabecular", "iris pigment",
             "optic nerve", "retinal pigment", "choroid"]),
    ("brain", ["brain", "cortex", "cerebell", "hippocamp", "neur", "astrocyte", "spinal cord", "caudate",
               "putamen", "thalamus", "hypothalamus", "medulla oblongata", "pons", "globus pallidus",
               "substantia nigra", "amygdala", "nucleus accumbens", "gyrus", "occipital", "parietal",
               "temporal lobe", "frontal", "insula", "pituitary", "olfactory", "glia", "meninge",
               "dura mater", "locus coeruleus", "cingulate", "corpus callosum", "pineal", "diencephalon",
               "cerebrum", "brainstem", "choroid plexus", "neural"]),
    ("heart_vascular", ["heart", "cardiac", "cardio", "aort", "arter", "vein", "venous", "endothelial",
                        "vascular", "atrium", "atrial", "ventric", "myocard", "coronary", "pericyte",
                        "smooth muscle cell of the pulmonary artery", "umbilical vein", "blood vessel"]),
    ("pancreas_islet", ["pancrea", "islet"]),
    ("liver", ["liver", "hepat"]),
    ("immune", ["t cell", "t-cell", "b cell", "b-cell", "natural killer", "nk cell", "monocyte", "macrophage",
                "dendritic", "neutrophil", "leukocyte", "lymphocyte", "thymus", "spleen", "tonsil",
                "lymph node", "peyer", "cd4", "cd8", "cd14", "cd19", "cd34", "cd56", "mast cell",
                "basophil", "eosinophil", "blood", "bone marrow", "treg", "regulatory t", "thymocyte",
                "immune", "erythro", "megakaryo", "granulocyte", "plasma cell"]),
    ("lung", ["lung", "bronch", "trache", "alveol", "pulmonary", "airway", "pleura"]),
    ("kidney", ["kidney", "renal", "nephr", "glomerul", "podocyte", "ureter", "bladder", "urothel",
                "mesangial", "proximal tubule", "urinary"]),
    ("muscle_adipose", ["muscle", "myoblast", "myotube", "skeletal", "adipo", "fat", "preadipocyte",
                        "psoas", "gastrocnemius", "quadriceps", "diaphragm", "tongue", "satellite cell"]),
    ("gut", ["stomach", "gastric", "colon", "colonic", "intestin", "rectum", "rectal", "duoden", "jejun",
             "ileum", "esophag", "oesophag", "small bowel", "large bowel", "caecum", "cecum", "appendix",
             "sigmoid", "mucosa of", "gut", "digestive", "gall bladder", "gallbladder", "bile duct"]),
    ("reproductive", ["uter", "ovar", "testis", "testes", "prostate", "placenta", "vagina", "cervix",
                      "endometri", "mammary", "breast", "fallopian", "oviduct", "chorion", "amnion",
                      "trophoblast", "penis", "foreskin", "epididym", "seminal", "myometri", "sperm",
                      "decidua", "umbilical cord", "vulva"]),
    ("skin", ["skin", "keratinocyte", "melanocyte", "dermal", "dermis", "epiderm", "fibroblast of skin",
              "hair follicle", "sebocyte"]),
]


def biosample(desc):
    parts = desc.split(":")
    if parts[0] == "CHIP" and len(parts) >= 3:
        return ":".join(parts[2:])
    return ":".join(parts[1:])


def assign(b):
    s = b.lower()
    if any(re.search(p, s) for p in OTHER_FIRST):
        return "other"
    if "choroid plexus" in s:
        return "brain"  # brain-ventricle epithelium, not eye (misfiled as eye in our AlphaGenome catalog)
    if s.startswith("eye - muscle"):
        return "muscle_adipose"  # extraocular skeletal muscle
    for tissue, kws in TISSUE_KW:
        if any(k in s for k in kws):
            return tissue
    return "other"


def build():
    t = pd.read_csv(TARGETS, sep="\t")
    t["assay"] = t.description.str.split(":").str[0].replace({"ATAC": "DNASE"})  # 10 ATAC tracks: chromatin
    t["biosample"] = t.description.map(biosample)
    t["tissue"] = t.biosample.map(assign)
    m = t[["index", "assay", "biosample", "tissue", "description"]]
    m.to_csv(OUT, index=False)
    return m


if __name__ == "__main__":
    m = build()
    print(m.pivot_table(index="tissue", columns="assay", values="index", aggfunc="count", fill_value=0))
    print(m[m.tissue == "eye"].description.tolist())
