"""
figure_qa.py

Rendered-figure checks on the exported PDF (our own implementation of the usual journal preflight):

  1. size      page width/height in mm (double column 183 mm, single 89 mm; height <= 247 mm)
  2. glyphs    every text span >= 5 pt (reports the smallest)
  3. overlaps  pairs of text spans whose boxes intersect (text colliding with text)
  4. strokes   text crossing a drawn line or curve (e.g. a label sitting on a data line)
  5. clipping  text running outside the page

Prints a one-line verdict and returns a dict. Use:

    python scripts/figure_qa.py figures_v2/Fig1.pdf
"""
import itertools
import sys

import fitz  # PyMuPDF

PT_PER_MM = 72 / 25.4
MIN_PT = 5.0


def check(pdf, width_mm=None, verbose=True):
    doc = fitz.open(pdf)
    page = doc[0]
    w, h = page.rect.width / PT_PER_MM, page.rect.height / PT_PER_MM
    spans = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for s in line["spans"]:
                if s["text"].strip():
                    spans.append((fitz.Rect(s["bbox"]), s["size"], s["text"].strip()))
    small = [(sz, t) for _, sz, t in spans if sz < MIN_PT - 0.05]
    overlaps = []
    for (r1, _, t1), (r2, _, t2) in itertools.combinations(spans, 2):
        inter = r1 & r2
        if not inter.is_empty and inter.width > 0.5 and inter.height > 0.5:
            overlaps.append((t1, t2))
    outside = [t for r, _, t in spans if not page.rect.contains(r)]
    crossings = []
    for d in page.get_drawings():
        if d.get("color") is None or d.get("fill") is not None:   # only unfilled strokes (lines, curves)
            continue
        for item in d["items"]:
            if item[0] != "l":
                continue
            p1, p2 = item[1], item[2]
            seg = fitz.Rect(min(p1.x, p2.x), min(p1.y, p2.y), max(p1.x, p2.x), max(p1.y, p2.y))
            for r, _, t in spans:
                core = fitz.Rect(r.x0 + 0.5, r.y0 + 1.0, r.x1 - 0.5, r.y1 - 1.0)   # ignore touching edges
                if core.is_empty:
                    continue
                if seg.width < 0.01:
                    seg = fitz.Rect(seg.x0 - 0.01, seg.y0, seg.x0 + 0.01, seg.y1)
                if seg.height < 0.01:
                    seg = fitz.Rect(seg.x0, seg.y0 - 0.01, seg.x1, seg.y0 + 0.01)
                if core.intersects(seg):
                    crossings.append(t)
    crossings = sorted(set(crossings))
    res = {"width_mm": round(w, 1), "height_mm": round(h, 1), "n_text": len(spans),
           "min_pt": round(min(sz for _, sz, _ in spans), 2) if spans else None,
           "below_5pt": small, "overlaps": overlaps, "crossings": crossings, "outside": outside}
    ok = not small and not overlaps and not crossings and not outside and h <= 247 and (width_mm is None or abs(w - width_mm) < 1)
    res["verdict"] = "PASS" if ok else "FIX"
    if verbose:
        print(f"[QA] {pdf.split('/')[-1].split(chr(92))[-1]}: {res['verdict']} | {w:.0f} x {h:.0f} mm | "
              f"{len(spans)} text runs, smallest {res['min_pt']} pt | overlaps {len(overlaps)} | "
              f"text on lines {len(crossings)} | below 5 pt {len(small)} | outside page {len(outside)}")
        for t1, t2 in overlaps[:10]:
            print(f"      overlap: {t1!r} <> {t2!r}")
        for t in crossings[:10]:
            print(f"      text on a line: {t!r}")
        for t in outside[:10]:
            print(f"      outside page: {t!r}")
        for sz, t in small[:10]:
            print(f"      below 5 pt ({sz:.1f}): {t.encode('ascii', 'replace').decode()!r}")
    return res


if __name__ == "__main__":
    for f in sys.argv[1:]:
        check(f)
