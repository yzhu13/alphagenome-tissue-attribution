"""
step47_enformer_scoring.py

Does the assay-composition artefact appear in a second sequence-to-function model?
Scores a random subset of our variants with Enformer (Avsec et al. 2021; PyTorch port
`enformer-pytorch` 0.8.12, weights EleutherAI/enformer-official-rough, CC-BY-4.0) on CPU.

Variants: 200 random-null variants + 70 loci from each disease domain (eye, kidney, metabolic), drawn
with a fixed seed from data/full_rescoring/variants.csv. For each variant the 196,608-bp hg38 window
centred on it is fetched from the UCSC REST API (no genome download; shifted inwards near chromosome
ends, padded with N beyond them) and REF and ALT sequences are run through Enformer.

Stored per variant and track (5,313 human tracks):
  ref_c, alt_c   predicted coverage summed over the 4 central 128-bp bins (512 bp, ~ the 501-bp
                 window used for AlphaGenome)
  sad            sum over all 896 bins of (alt - ref)

Checkpointed: one .npz per variant in data/enformer/scores/; rerun to resume.
Analysis in step48_enformer_analysis.py.

    python scripts/step47_enformer_scoring.py
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step35_analysis as s35  # noqa: E402

OUT = os.path.join(s35.NEW, "data", "enformer")
SC = os.path.join(OUT, "scores")
SEQ_LEN, HALF = 196_608, 98_304
CENTER_BINS = slice(446, 450)  # bin 448 starts at the variant
N_NULL, N_DOMAIN, SEED = 200, 70, 0
CHROM_SIZES_URL = "https://api.genome.ucsc.edu/list/chromosomes?genome=hg38"


def pick_variants():
    path = os.path.join(OUT, "variants_enformer.csv")
    if os.path.exists(path):
        return pd.read_csv(path, keep_default_na=False)
    v = s35.read_variants()
    dom = s35.domain_map(v)
    rng = np.random.default_rng(SEED)
    rows = []
    taken = set()
    for d, n in [("null", N_NULL), ("eye", N_DOMAIN), ("kidney", N_DOMAIN), ("metabolic", N_DOMAIN)]:
        ids = sorted(set(dom.loc[dom.domain == d, "id"]) - taken)
        pick = rng.choice(ids, n, replace=False)
        taken |= set(pick)
        rows += [(i, d) for i in pick]
    p = pd.DataFrame(rows, columns=["id", "pick_domain"]).merge(v, on="id")
    p.to_csv(path, index=False)
    return p


def get_json(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(2 + 3 * k)
    raise RuntimeError(f"failed: {url}")


def fetch_window(chrom, pos, sizes):
    """196,608 bp with the variant at index HALF (0-based); N beyond chromosome ends."""
    start = pos - 1 - HALF  # 0-based start
    lo, hi = max(start, 0), min(start + SEQ_LEN, sizes[chrom])
    seq = get_json(f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={chrom};start={lo};end={hi}")["dna"]
    return "N" * (lo - start) + seq.upper() + "N" * (start + SEQ_LEN - hi)


def one_hot(seq):
    lut = np.zeros((256, 4), np.float32)
    for i, b in enumerate("ACGT"):
        lut[ord(b), i] = 1
    return lut[np.frombuffer(seq.encode(), np.uint8)]


def main():
    from enformer_pytorch import from_pretrained
    os.makedirs(SC, exist_ok=True)
    p = pick_variants()
    sizes = get_json(CHROM_SIZES_URL)["chromosomes"]
    torch.set_num_threads(os.cpu_count())
    model = from_pretrained("EleutherAI/enformer-official-rough", use_tf_gamma=False).eval()
    todo = [r for r in p.itertuples() if not os.path.exists(os.path.join(SC, f"{r.id}.npz"))]
    print(f"{len(p) - len(todo)} done, {len(todo)} to score", flush=True)
    t0 = time.time()
    for k, r in enumerate(todo):
        chrom = f"chr{r.CHR}"
        seq = fetch_window(chrom, int(r.pos), sizes)
        ref, alt = r.ref_allele, r.alt_allele
        got = seq[HALF:HALF + len(ref)]
        flag = "ok"
        if got != ref:
            if seq[HALF:HALF + len(alt)] == alt:
                ref, alt, flag = alt, ref, "swapped"
            else:
                flag = f"mismatch:{got}"
        alt_seq = (seq[:HALF] + alt + seq[HALF + len(ref):])[:SEQ_LEN].ljust(SEQ_LEN, "N")
        x = torch.from_numpy(np.stack([one_hot(seq), one_hot(alt_seq)]))
        with torch.no_grad():
            y = model(x)["human"].numpy()  # (2, 896, 5313)
        np.savez_compressed(os.path.join(SC, f"{r.id}.npz"),
                            ref_c=y[0, CENTER_BINS].sum(0).astype(np.float32),
                            alt_c=y[1, CENTER_BINS].sum(0).astype(np.float32),
                            sad=(y[1] - y[0]).sum(0).astype(np.float32), flag=flag)
        el = time.time() - t0
        print(f"{k + 1}/{len(todo)} {r.id} {flag} {el / (k + 1):.0f}s/variant, "
              f"~{el / (k + 1) * (len(todo) - k - 1) / 3600:.1f} h left", flush=True)


if __name__ == "__main__":
    main()
