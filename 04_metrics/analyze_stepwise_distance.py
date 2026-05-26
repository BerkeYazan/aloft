#!/usr/bin/env python3
"""
Fast Stepwise Semantic Distance Analysis with Aggregation (>= 2-sentence docs)
-------------------------------------------------------------------------------
Performs a pairwise comparison between two groups of corpora, where each group
can be composed of one or more source columns from the main data CSV.

- **Excludes one-sentence documents** (they are skipped, not scored).
- Aggregates texts from multiple source columns into a single group.
- Mann-Whitney U (auto method) + Cliff’s δ effect size.
- Saves: TXT report and a PNG comparison plot.

Usage:
    python 04_metrics/analyze_stepwise_distance.py \\
        --test_corpora "Goodreads Popular Quote" "Goodreads Sample Quote" \\
        --baseline_corpora "Google Books Length Matched Snippet" "Non-Literary Baseline" \\
        --output_dir "data/outputs/runtime/stepwise_distance/quotes_vs_baselines"
"""

from __future__ import annotations
import argparse, logging, pathlib, sys
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from scipy.stats import mannwhitneyu, rankdata
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)


# ------------ effect sizes & CI ------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    n_x, n_y = len(x), len(y)
    ranks = rankdata(np.concatenate([x, y]))
    R_x = ranks[: n_x].sum()
    u = R_x - n_x * (n_x + 1) / 2.0
    return (2 * u) / (n_x * n_y) - 1


rank_biserial = cliffs_delta  # identical


def delta_ci(x: np.ndarray, y: np.ndarray,
             n_boot: int = 2_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_x, n_y = len(x), len(y)
    idx_x, idx_y = np.arange(n_x), np.arange(n_y)
    deltas = np.empty(n_boot)
    for i in tqdm(range(n_boot), desc="Bootstrap δ  ", ncols=80):
        deltas[i] = cliffs_delta(
            x[rng.choice(idx_x, n_x, replace=True)],
            y[rng.choice(idx_y, n_y, replace=True)],
        )
    return np.percentile(np.sort(deltas), [2.5, 97.5])


# ------------ helpers ------------
def corpus_scores(
    texts: List[str],
    model: SentenceTransformer,
    desc: str,
) -> np.ndarray:
    """
    Computes stepwise distance for all documents in a corpus using batched
    sentence encoding for performance. Excludes documents with < 2 sentences.
    Returns an array of scores.
    """
    LOG.info("Tokenizing all documents for %s...", desc)
    docs_sents = [sent_tokenize(txt) for txt in texts]

    # Filter documents and prepare for batching
    indices_to_keep = [i for i, sents in enumerate(docs_sents) if len(sents) >= 2]
    docs_to_process = [docs_sents[i] for i in indices_to_keep]

    if not docs_to_process:
        return np.array([], dtype=np.float32)

    all_sents_flat = [s for doc in docs_to_process for s in doc]

    LOG.info(
        "Encoding %d sentences from %d valid documents (batch-processed)...",
        len(all_sents_flat),
        len(docs_to_process),
    )
    all_embeddings = model.encode(
        all_sents_flat,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    LOG.info("Calculating batched stepwise distances...")
    scores = []
    start_idx = 0
    doc_sent_counts = [len(doc) for doc in docs_to_process]
    for count in tqdm(doc_sent_counts, desc=f"SWD for {desc}", ncols=80):
        end_idx = start_idx + count
        emb = all_embeddings[start_idx:end_idx]
        diffs = np.diff(emb, axis=0)
        swd = np.mean(np.sum(diffs**2, axis=1))
        scores.append(swd)
        start_idx = end_idx

    return np.array(scores, dtype=np.float32)


# ------------ main analysis ------------
def analyse(cfg: SimpleNamespace) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rpt = cfg.output_dir / "stepwise_distance_report.txt"
    plot = cfg.output_dir / "stepwise_distance_comparison.png"

    LOG.info("Reading CSV: %s", cfg.aloft_csv)
    try:
        df = pd.read_csv(cfg.aloft_csv)
    except FileNotFoundError:
        LOG.error("ALOFT data file not found at: %s", cfg.aloft_csv)
        sys.exit(1)


    LOG.info("Initialising SBERT (%s)", cfg.sbert_model)
    model = SentenceTransformer(cfg.sbert_model)

    # Aggregate texts from specified corpora columns
    LOG.info("Aggregating text corpora...")
    test_texts = pd.concat(
        [df[col].dropna() for col in cfg.test_corpora], ignore_index=True
    ).astype(str).tolist()
    base_texts = pd.concat(
        [df[col].dropna() for col in cfg.baseline_corpora], ignore_index=True
    ).astype(str).tolist()

    test_desc = f"Test ({', '.join(cfg.test_corpora)})"
    base_desc = f"Baseline ({', '.join(cfg.baseline_corpora)})"

    xs = corpus_scores(test_texts, model, test_desc)
    ys = corpus_scores(base_texts, model, base_desc)

    if len(xs) == 0 or len(ys) == 0:
        LOG.error("After excluding one-sentence docs, one corpus group is empty.")
        sys.exit(1)

    # statistics
    mw_method = "auto"
    u, p = mannwhitneyu(xs, ys, alternative="two-sided", method=mw_method)
    δ = cliffs_delta(xs, ys)
    r_rb = rank_biserial(xs, ys)
    ci_lo, ci_hi = delta_ci(xs, ys)

    with rpt.open("w") as f:
        f.write("=" * 72 + "\nStepwise Semantic Distance Report\n" + "=" * 72 + "\n\n")
        f.write(f"Test Corpora:     {cfg.test_corpora}\n")
        f.write(f"Baseline Corpora: {cfg.baseline_corpora}\n\n")
        for name, arr in (
            ("Test group", xs),
            ("Baseline group", ys),
        ):
            f.write(
                f"{name} (n={len(arr)}):  mean={arr.mean():.3f}  "
                f"median={np.median(arr):.3f}  sd={arr.std(ddof=1):.3f}\n"
            )
        f.write(
            f"\nMann-Whitney U (method={mw_method}):  U={u:.0f}  p={p:.3g}\n"
            f"Cliff’s δ  = {δ:+.3f}  (95 % CI {ci_lo:+.3f} ... {ci_hi:+.3f})\n"
            f"Rank-biserial r_rb = {r_rb:+.3f}\n"
        )
    LOG.info("Report -> %s", rpt)

    # plot
    sns.set_theme(style="whitegrid")
    df_plot = pd.DataFrame(
        {
            "Stepwise Distance": np.r_[xs, ys],
            "Corpus": [test_desc] * len(xs) + [base_desc] * len(ys),
        }
    )
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df_plot,
        x="Corpus",
        y="Stepwise Distance",
        inner="quartile",
        palette="colorblind",
        cut=0,
    )
    plt.title("Stepwise Semantic Distance (>= 2-sentence docs)")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(plot, dpi=300)
    LOG.info("Plot   -> %s", plot)


# ------------ CLI ------------
def parse() -> SimpleNamespace:
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=__doc__)
    p.add_argument("--test_corpora",     nargs='+', type=str, required=True,
                   help="One or more column names from the CSV for the test group.")
    p.add_argument("--baseline_corpora", nargs='+', type=str, required=True,
                   help="One or more column names from the CSV for the baseline group.")
    p.add_argument("--aloft_csv",        type=pathlib.Path, default="ALOFT.csv")
    p.add_argument("--sbert_model",
                   default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--output_dir",       type=pathlib.Path, required=True,
                   help="Directory to save the report and plot.")
    return SimpleNamespace(**vars(p.parse_args()))


if __name__ == "__main__":
    cfg = parse()
    analyse(cfg)
