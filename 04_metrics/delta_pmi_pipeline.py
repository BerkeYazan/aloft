#!/usr/bin/env python
"""
delta_pmi_pipeline.py · v1.3 · 2025-07-18
Compute Δ-PMI (document PMI – background PMI) for the ALOFT corpus.

Changes v1.3
------------
- Removed on-the-fly creation of 'T50 Quote-Free Context'. This column
  is now expected to be pre-calculated in the input CSV.

Author : <your-name>
Licence: MIT
"""

from __future__ import annotations
import argparse, math, pickle, random, re, sys
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import spacy
from tqdm.auto import tqdm

# ------------------------- PARAMETERS ------------------------- #
WINDOW = 5           # ±5-token window for PMI
SEED   = 42
random.seed(SEED)

BG_COLS = [
    "Google Books Page Text",
    "T50 Quote-Free Context",
    "Non-Literary Baseline",
]

DEFAULT_TEXT_COLS = [
    "Goodreads Sample Quote",
    "Goodreads Popular Quote",
    "Google Books Length Matched Snippet",
    "T50 Quote-Free Context Length Matched",
    "T50 Quote",
]

TOKEN_RE = re.compile(r"^[a-z']+$", re.I)

# ------------------------- HELPERS ---------------------------- #

def iter_lemmas(nlp, texts: Iterable[str]) -> Iterable[List[str]]:
    """Yield clean lemmas for each text."""
    for doc in nlp.pipe(texts, batch_size=500, n_process=2):
        yield [
            t.lemma_.lower()
            for t in doc
            if t.is_alpha and TOKEN_RE.match(t.lemma_)
        ]

def sliding_pairs(tokens: List[str], width: int = WINDOW):
    """Generate ordered pairs within window (excluding self-pairs)."""
    for i, w in enumerate(tokens):
        for j in range(max(0, i - width), min(len(tokens), i + width + 1)):
            if i != j:
                yield w, tokens[j]

def collect_counts(token_stream) -> tuple[Counter, Counter]:
    uni, bi = Counter(), Counter()
    for toks in tqdm(token_stream, desc="Counting"):
        uni.update(toks)
        bi.update(sliding_pairs(toks))
    return uni, bi

def delta_ppmi(tokens, bg_uni, bg_bi, N_bg, α=0.5):
    U, B = Counter(tokens), Counter(sliding_pairs(tokens))
    N_d  = len(tokens)
    if N_d < 2:
        return np.nan
    vals = []
    for (w1, w2), c12 in B.items():
        if w1 not in bg_uni or w2 not in bg_uni:
            continue
        # Local PMI
        p_d   = c12 / N_d
        p1_d  = U[w1] / N_d
        p2_d  = U[w2] / N_d
        pmi_d = math.log2(p_d / (p1_d * p2_d))
        # BG PMI with add-α smoothing
        p_bg   = (bg_bi.get((w1, w2), 0) + α) / (N_bg + α)
        p1_bg  = bg_uni[w1] / N_bg
        p2_bg  = bg_uni[w2] / N_bg
        pmi_bg = math.log2(p_bg / (p1_bg * p2_bg))
        vals.append(pmi_d - pmi_bg)
    return np.mean(vals) if vals else np.nan

# ------------------------- MAIN ---------------------------- #

def main(args):
    np.random.seed(SEED)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. spaCy initialisation
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.max_length = 2_000_000
    print("spaCy loaded")

    # 2. CSV load
    df = pd.read_csv(args.csv)
    print(f"CSV loaded, {len(df):,} rows")

    missing_cols = {"T50 Full Context", "T50 Quote"} - set(df.columns)
    if missing_cols:
        sys.exit(f"ERROR: Missing columns {missing_cols} in CSV.")

    # 3. Optional sample
    if args.sample < 1.0:
        df = df.sample(frac=args.sample, random_state=SEED)
        print(f"Pilot run: {len(df):,} rows ({args.sample*100:.0f} %)")

    # 4. Build background model
    bg_texts = chain.from_iterable(df[col].fillna("").tolist() for col in BG_COLS)
    bg_uni, bg_bi = collect_counts(iter_lemmas(nlp, bg_texts))
    N_bg = sum(bg_uni.values())
    print(f"BG counts: {N_bg:,} tokens · {len(bg_uni):,} types")

    pickle.dump({"uni": bg_uni, "bi": bg_bi, "N": N_bg},
                open(out_dir / "bg_counters.pkl", "wb"))

    # 5. Δ-PMI for target columns
    targets = args.cols if args.cols else DEFAULT_TEXT_COLS
    targets = [c for c in targets if c in df.columns]

    results = {c: [] for c in targets}
    for col in targets:
        print(f"Scoring column -> {col}")
        for toks in iter_lemmas(nlp, df[col].fillna("")):
            results[col].append(delta_ppmi(toks, bg_uni, bg_bi, N_bg))

    pd.DataFrame(results).to_csv(out_dir / "delta_pmi_scores.csv", index=False)
    print(f"Results saved -> {out_dir/'delta_pmi_scores.csv'}")

# ------------------------- CLI ------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to ALOFT.csv")
    p.add_argument("--sample", type=float, default=1.0,
                   help="Fraction of rows for pilot (e.g. 0.10)")
    p.add_argument("--cols", nargs="+",
                   help="Override default target columns")
    p.add_argument("--out", default="delta_pmi_outputs",
                   help="Output directory")
    main(p.parse_args())
