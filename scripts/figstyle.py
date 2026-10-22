"""
figstyle.py

One visual system for every figure of the paper, following journal figure practice:

  - final physical size: double column 183 mm, single column 89 mm; figures are drawn at final size
  - Arial, 7 pt body text, 6 pt ticks, 8 pt bold lowercase panel letters; nothing below 5 pt
  - thin lines (axes 0.6 pt, ticks 0.6 pt), left + bottom spines only, no chart titles inside figures
    (the title lives in the legend), frameless legends, direct labels where possible
  - one restrained palette used identically in all figures (colour-blind-safe Okabe-Ito hues):
      neutral greys for "other tissues" and scaffolding, vermillion for the eye (the tissue the story
      follows), and one family per scoring method; diverging blue-white-red for signed quantities
  - editable text in vector output (pdf.fonttype 42, svg.fonttype none); PDF + SVG + 300-dpi PNG

save() writes all three formats and runs figure_qa.check() on the PDF.
"""
import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

MM = 1 / 25.4
DOUBLE, SINGLE = 183 * MM, 89 * MM
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures_v2")

INK, INK2, MUTED = "#1A1A1A", "#4D4D4D", "#8C8C8C"
GRID, NEUTRAL, NEUTRAL_LIGHT = "#E6E6E6", "#BDBDBD", "#DDDDDD"
EYE = "#D55E00"              # vermillion: the eye, throughout
IMMUNE = "#009E73"           # bluish green: immune cells (immune benchmark)
RAW = "#4D4D4D"              # original pipeline (raw scores): dark grey baseline
CAL = "#56B4E9"              # calibrated, mean |q|: sky blue
STRONG = "#0072B2"           # calibrated, strong-effect fraction: deep blue (recommended)
ASSAY = {"DNase": "#0072B2", "ATAC": "#56B4E9", "RNA-seq": "#E69F00",   # assay identity (Fig. 2)
         "CAGE": "#CC79A7", "ChIP": "#999999"}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2166AC", "#F7F7F7", "#B2182B"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#08306B", "#4A87C0", "#C6DBEF", "#F4F8FC"])

TISSUE_LABEL = {"eye": "Eye", "brain": "Brain", "heart_vascular": "Heart/vascular", "pancreas_islet": "Pancreatic islet",
                "liver": "Liver", "immune": "Immune", "lung": "Lung", "kidney": "Kidney",
                "muscle_adipose": "Muscle/adipose", "gut": "Gut", "reproductive": "Reproductive", "skin": "Skin",
                "other": "Other", "eye_misfiled": "Choroid plexus (misfiled)"}


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 4,
        "legend.frameon": False, "legend.handlelength": 1.2, "lines.linewidth": 1.0,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300, "figure.dpi": 150,
        "svg.hashsalt": "tissue-attribution",  # stable element ids, so reruns give identical SVG files
    })


def panel_label(ax, letter, x=-0.02, y=None, fig=None):
    """Bold lowercase panel letter at the top-left of an axes (figure coordinates). y: figure y of the
    panel's header line, so that the letter sits on the same baseline as the panel title."""
    fig = fig or ax.figure
    bb = ax.get_position()
    fig.text(bb.x0 + x, (bb.y1 + 0.012) if y is None else y, letter, fontsize=8, fontweight="bold",
             ha="left", va="bottom")


def tissue(t):
    return TISSUE_LABEL.get(t, t)


def save(fig, name, qa=True):
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, name)
    fig.savefig(base + ".pdf", metadata={"CreationDate": None, "ModDate": None})  # no timestamps:
    fig.savefig(base + ".svg", metadata={"Date": None})                          # reruns give identical files
    fig.savefig(base + ".png", dpi=300)
    plt.close(fig)
    if qa:
        import figure_qa
        return figure_qa.check(base + ".pdf")
    return None
