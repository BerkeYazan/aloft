#!/usr/bin/env python
"""
Calculates computational creativity metrics for the ALOFT dataset.

This script implements two metrics to score the creativity of literary quotes:
1.  **Semantic Volatility (Forward Flow):** Measures the average semantic distance
    between adjacent words in a quote. A higher score suggests more "leaps"
    between concepts, a proxy for creative word choice. This uses GloVe
    word embeddings.
2.  **Semantic Originality (Uniqueness):** Measures how semantically distant a
    quote is from a large, baseline corpus of literary text. A higher score
    suggests the quote is more semantically unique or uncommon. This uses
    SBERT sentence embeddings.

The script loads the "Goodreads Popular Quote" corpus, calculates these scores,
and prints the top 20 most "creative" quotes according to each metric.

To run, simply execute from the terminal:
    python 04_metrics/calculate_creativity_metrics.py
"""
from __future__ import annotations

import pathlib
import sys
from typing import List

import gensim.downloader as api
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances

# --- Download NLTK tokenizer models if not present ---
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading NLTK models for tokenization...", file=sys.stderr)
    import nltk
    nltk.download("punkt", quiet=True)


def load_glove_model(model_name: str = "glove-wiki-gigaword-300") -> KeyedVectors:
    """Downloads and loads a pre-trained GloVe model."""
    print(f"Loading GloVe model: '{model_name}'...")
    print("This may take a few minutes on the first run...")
    try:
        model = api.load(model_name)
        print("GloVe model loaded.")
        return model
    except ValueError as e:
        print(f"ERROR: Could not load GloVe model '{model_name}'.", file=sys.stderr)
        sys.exit(1)


def load_sbert_embeddings(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Loads SBERT vectors and their original indices from a .npz file."""
    if not path.exists():
        print(f"ERROR: SBERT embedding file not found at {path}", file=sys.stderr)
        return None
    with np.load(path) as data:
        return data["vectors"], data["ids"]


def calculate_forward_flow(texts: List[str], model: KeyedVectors) -> List[float]:
    """Calculates the Semantic Volatility (Forward Flow) for a list of texts."""
    scores = []
    print("Calculating Semantic Volatility (Forward Flow)...")
    for text in texts:
        tokens = word_tokenize(text.lower())
        word_vectors = [model[word] for word in tokens if word in model]

        if len(word_vectors) < 2:
            scores.append(0.0)
            continue

        distances = [
            cosine(word_vectors[i], word_vectors[i + 1])
            for i in range(len(word_vectors) - 1)
        ]
        scores.append(np.mean(distances) if distances else 0.0)
    return scores


def calculate_originality(
    target_vectors: np.ndarray, reference_vectors: np.ndarray
) -> np.ndarray:
    """Calculates the Semantic Originality for a set of target vectors."""
    print("Calculating Semantic Originality (Uniqueness)...")
    # Calculate the pairwise distances from each target to all reference vectors
    # Result is a matrix of shape (n_target, n_reference)
    dist_matrix = cosine_distances(target_vectors, reference_vectors)
    
    # The originality score is the mean distance to all reference points
    originality_scores = np.mean(dist_matrix, axis=1)
    return originality_scores


def print_top_20_report(df: pd.DataFrame, metric_name: str, description: str):
    """Formats and prints a top-20 report for a given metric."""
    print("\n" + "=" * 80)
    print(f"TOP 20 QUOTES BY: {metric_name.upper()}")
    print("=" * 80)
    print(description)
    print("-" * 80)

    # Sort and select the top 20
    top_20_df = df.sort_values(by=metric_name, ascending=False).head(20)

    # Set pandas display options to show full text
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 120)

    # Format for printing
    report_df = pd.DataFrame({
        "Rank": range(1, 21),
        "Score": top_20_df[metric_name],
        "Quote": top_20_df["text"]
    })
    
    print(report_df.to_string(index=False))
    pd.reset_option('display.max_colwidth')
    pd.reset_option('display.width')


def main():
    """Main function to run the creativity analysis."""
    # --- Configuration ---
    csv_path = pathlib.Path("data/processed/public/ALOFT.csv")
    sbert_dir = pathlib.Path("data/interim/dynamic_embeddings")
    
    target_col = "Goodreads Popular Quote"
    reference_col = "Goodreads Sample Quote"

    # --- 1. Load Data ---
    if not csv_path.exists():
        print(f"ERROR: Source CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading dataset {csv_path}...")
    df_main = pd.read_csv(csv_path)

    # Prepare target texts
    target_series = df_main[target_col].dropna()
    target_texts = target_series.tolist()
    
    # --- 2. Calculate Semantic Volatility (Forward Flow) ---
    glove_model = load_glove_model()
    forward_flow_scores = calculate_forward_flow(target_texts, glove_model)
    
    # Create a DataFrame to store results
    results_df = pd.DataFrame({
        "text": target_texts,
        "Forward Flow": forward_flow_scores
    })

    # --- 3. Calculate Semantic Originality ---
    print("\nLoading SBERT embeddings for Originality calculation...")
    target_data = load_sbert_embeddings(sbert_dir / "goodreads_popular_quote.npz")
    reference_data = load_sbert_embeddings(sbert_dir / "goodreads_sample_quote.npz")

    if not target_data or not reference_data:
        print("Could not load SBERT embeddings. Aborting Originality calculation.", file=sys.stderr)
    else:
        target_vectors, target_ids = target_data
        reference_vectors, _ = reference_data
        
        originality_scores = calculate_originality(target_vectors, reference_vectors)
        
        # Add to results, aligning by index
        originality_df = pd.DataFrame({
            'originality_score': originality_scores,
            'id': target_ids
        }).set_index('id')
        
        # Join with main dataframe index to align scores with text
        temp_df = pd.DataFrame(index=target_series.index)
        temp_df = temp_df.join(originality_df)
        
        results_df["Originality"] = temp_df['originality_score'].values

    # --- 4. Generate Reports ---
    ff_desc = (
        "These quotes have the largest average semantic 'jumps' between adjacent words.\n"
        "They may contain surprising word pairings or bridge disparate concepts."
    )
    print_top_20_report(results_df, "Forward Flow", ff_desc)

    if "Originality" in results_df.columns:
        orig_desc = (
            "These quotes are the most semantically distant from the 'average' literary quote.\n"
            "They represent unique or uncommon ideas and phrasing."
        )
        print_top_20_report(results_df, "Originality", orig_desc)
        
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main() 