#!/usr/bin/env python
"""
Compares the Forward Flow between two aggregated groups of text corpora.

This script implements the "Forward Flow" (FF) metric from Gray et al. (2019),
a measure of semantic novelty based on GloVe embeddings. It performs a pairwise
comparison between a test group and a baseline group, where each group can be
composed of one or more source columns from the main ALOFT CSV file.

The analysis includes:
1. Calculation of Forward Flow scores for all documents in both groups.
2. A Mann-Whitney U test to compare the distributions.
3. Cliff's Delta as a non-parametric effect size.
4. A summary report (.txt) with descriptive and inferential statistics.
5. A violin plot (.png) visualizing the comparison.
6. A list of the Top 20 highest-scoring examples from the test group.

Usage:
    python 04_metrics/analyze_forward_flow.py \\
        --test_corpora "Goodreads Popular Quote" "Goodreads Sample Quote" \\
        --baseline_corpora "Google Books Length Matched Snippet" "Non-Literary Baseline" \\
        --output_dir "data/outputs/runtime/forward_flow/quotes_vs_baselines"
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import logging
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, rankdata
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Add project root to sys.path to allow absolute imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)

# --- NLTK data download ---
def check_nltk_data():
    """Downloads NLTK 'punkt' model if not found."""
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        LOG.info("Downloading NLTK 'punkt' model for word tokenization...")
        import nltk
        nltk.download("punkt", quiet=True)
        LOG.info("NLTK 'punkt' model downloaded.")

check_nltk_data()

# --- GloVe Model Loading ---
def load_glove_model(glove_file: pathlib.Path) -> KeyedVectors:
    """
    Loads a GloVe model from a text file, converting it to a faster binary
    format on the first run.
    """
    model_file = glove_file.with_suffix(".model")
    if model_file.exists():
        LOG.info(f"Loading pre-converted GloVe model from '{model_file}'...")
        return KeyedVectors.load(str(model_file))

    LOG.info(f"Loading GloVe model from text file: '{glove_file}'...")
    LOG.info("This may be slow, but a binary version will be saved for future runs.")
    try:
        model = KeyedVectors.load_word2vec_format(
            str(glove_file), binary=False, no_header=True
        )
        model.save(str(model_file))
        LOG.info(f"GloVe model loaded and saved to '{model_file}'.")
        return model
    except Exception as e:
        LOG.error(f"Could not load GloVe model from '{glove_file}'. Details: {e}", exc_info=True)
        sys.exit(1)


# ------------ effect sizes ------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    n_x, n_y = len(x), len(y)
    ranks = rankdata(np.concatenate([x, y]))
    R_x = ranks[: n_x].sum()
    u = R_x - n_x * (n_x + 1) / 2.0
    return (2 * u) / (n_x * n_y) - 1

rank_biserial = cliffs_delta  # identical


# --- Metric Calculation ---
def calculate_forward_flow(text: str, model: KeyedVectors) -> float | None:
    """
    Calculates the Forward Flow for a single text.
    FF = mean of (mean distance from word i to all preceding words)
    Returns None for texts with < 2 words with embeddings.
    """
    if not isinstance(text, str):
        return None
    tokens = word_tokenize(text.lower())
    word_vectors = [model[word] for word in tokens if word in model]
    n = len(word_vectors)
    if n < 2:
        return None

    stepwise_avg_distances = [
        np.mean(np.linalg.norm(word_vectors[i] - np.array(word_vectors[:i]), axis=1))
        for i in range(1, n)
    ]
    # Handle case where only one distance is calculated (n=2)
    return np.mean(stepwise_avg_distances) if stepwise_avg_distances else None


def get_corpus_ff_scores(
    texts: List[str], model: KeyedVectors, desc: str
) -> pd.DataFrame:
    """
    Calculates Forward Flow scores for a pandas Series of texts.
    Returns a DataFrame with ['text', 'score'] for scorable documents.
    """
    processed_data = []
    for text in tqdm(texts, desc=desc, ncols=80):
        score = calculate_forward_flow(text, model)
        if score is not None:
            processed_data.append({"text": text, "score": score})
    return pd.DataFrame(processed_data)


# --- Main Analysis Logic ---
def analyse(cfg: SimpleNamespace):
    """Main analysis pipeline for pairwise comparison."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rpt = cfg.output_dir / "forward_flow_report.txt"
    plot = cfg.output_dir / "forward_flow_comparison.png"

    LOG.info("--- Initializing Models and Data ---")
    try:
        df = pd.read_csv(cfg.aloft_csv)
    except FileNotFoundError:
        LOG.error(f"ALOFT.csv not found at '{cfg.aloft_csv}'")
        sys.exit(1)

    glove_model = load_glove_model(cfg.glove_path)

    LOG.info("--- Processing Corpora ---")
    test_texts = pd.concat(
        [df[col].dropna() for col in cfg.test_corpora], ignore_index=True
    ).astype(str).tolist()
    base_texts = pd.concat(
        [df[col].dropna() for col in cfg.baseline_corpora], ignore_index=True
    ).astype(str).tolist()

    test_desc = f"Test ({', '.join(cfg.test_corpora)})"
    base_desc = f"Baseline ({', '.join(cfg.baseline_corpora)})"

    test_results_df = get_corpus_ff_scores(
        test_texts, glove_model, test_desc
    )
    base_results_df = get_corpus_ff_scores(
        base_texts, glove_model, base_desc
    )

    xs = test_results_df["score"].to_numpy(dtype=np.float32)
    ys = base_results_df["score"].to_numpy(dtype=np.float32)

    if len(xs) == 0 or len(ys) == 0:
        LOG.error("After filtering, one corpus group is empty. Cannot compare.")
        sys.exit(1)

    # --- Statistics ---
    u, p = mannwhitneyu(xs, ys, alternative="two-sided", method="auto")
    δ = cliffs_delta(xs, ys)
    r_rb = rank_biserial(xs, ys)  # Same as Cliff's delta

    # --- Save Top 20 Examples from Test Corpus ---
    top_20_path = cfg.output_dir / "top_20_forward_flow_test_group.txt"
    LOG.info(f"Saving top 20 examples from test group to {top_20_path}")
    top_20_df = test_results_df.nlargest(20, "score").reset_index(drop=True)

    with top_20_path.open("w", encoding="utf-8") as f:
        f.write(
            f"Top 20 Examples from Test Group ({', '.join(cfg.test_corpora)}) ranked by Forward Flow Score\n"
        )
        f.write("=" * 80 + "\n\n")
        for rank, row in top_20_df.iterrows():
            f.write(f"Rank: {rank + 1}\n")
            f.write(f"Forward Flow Score: {row['score']:.4f}\n")
            f.write(f"Text:\n---\n{row['text']}\n---\n\n")

    # --- Generate Summary Report ---
    LOG.info(f"Generating summary report at {rpt}")
    with rpt.open("w", encoding="utf-8") as f:
        f.write("=" * 72 + "\nForward Flow - Pairwise Comparison Report\n" + "=" * 72 + "\n\n")
        f.write(f"Test Corpora:     {', '.join(cfg.test_corpora)}\n")
        f.write(f"Baseline Corpora: {', '.join(cfg.baseline_corpora)}\n\n")

        for name, arr in (("Test Group", xs), ("Baseline Group", ys)):
            f.write(
                f"{name} (n={len(arr)}):  mean={arr.mean():.4f}  "
                f"median={np.median(arr):.4f}  sd={arr.std(ddof=1):.4f}\n"
            )
        f.write(
            f"\nMann-Whitney U:  U={u:.0f}  p={p:.3g}\n"
            f"Cliff’s δ        = {δ:+.4f}\n"
            f"Rank-biserial r_rb = {r_rb:+.4f}\n"
        )
    LOG.info("Report -> %s", rpt)

    # --- Generate Combined Visualization ---
    LOG.info(f"Generating comparison plot at {plot}")
    df_plot = pd.DataFrame({
        "Forward Flow": np.r_[xs, ys],
        "Corpus": [test_desc] * len(xs) + [base_desc] * len(ys),
    })
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.violinplot(
        data=df_plot,
        x="Corpus",
        y="Forward Flow",
        inner="quartile",
        palette="colorblind",
        cut=0,
    )
    plt.title("Forward Flow Comparison")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(plot, dpi=300)
    LOG.info("Plot   -> %s", plot)

    LOG.info("Analysis complete.")


# --- CLI ---
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=__doc__)
    p.add_argument("--test_corpora", nargs='+', type=str, required=True, help="One or more column names from the CSV for the test group.")
    p.add_argument("--baseline_corpora", nargs='+', type=str, required=True, help="One or more column names from the CSV for the baseline group.")
    p.add_argument("--aloft_csv", type=pathlib.Path, default="ALOFT.csv", help="Path to the main ALOFT data file.")
    p.add_argument("--glove_path", type=pathlib.Path, default=pathlib.Path("data/models/glove.840B.300d.zip"), help="Path to the GloVe model file (txt or zip).")
    p.add_argument("--output_dir", type=pathlib.Path, required=True, help="Directory to save the report and plot.")

    cfg = SimpleNamespace(**vars(p.parse_args()))
    # Check if aloft_csv exists
    if not cfg.aloft_csv.is_file():
        LOG.error(f"ALOFT data file not found at: {cfg.aloft_csv}")
        sys.exit(1)

    analyse(cfg)

if __name__ == "__main__":
    main() 