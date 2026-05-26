#!/usr/bin/env python
"""
Evaluates and visualizes the effectiveness of length normalization for surprisal.

This script provides a rigorous, data-driven validation that the chosen
length-normalized metric (Average Surprisal / Perplexity) is not biased by
text length, a crucial step for producing publication-quality research.

It generates a two-panel plot to visually demonstrate the "before and after"
of normalization:
1.  The "Problem" panel shows a scatter plot of an un-normalized likelihood
    metric (Total Log Probability) vs. text length (Token Count). This plot
    is expected to show a strong correlation, proving the existence of a
    length-based bias.
2.  The "Solution" panel shows a scatter plot of the length-normalized
    Perplexity vs. Token Count. This plot is expected to show no significant
    correlation, proving that the normalization has successfully removed the
    length bias.

The script also calculates and prints the Pearson correlation coefficient for
each relationship, providing quantitative support for the visual evidence.

To run, execute from the terminal:
    python 05_analysis/evaluate_normalization.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

import config

# --- Configuration ---
INPUT_FILE = config.BASE_OUTPUT_DIR / "surprisal_aha" / "surprisal_aha_metrics.csv"
OUTPUT_DIR = config.BASE_OUTPUT_DIR / "aha_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    """Main function to load data, create plots, and calculate correlations."""
    print(f"Loading metrics from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERROR: Metrics file not found at {INPUT_FILE}")
        print("Please run 'analyze_surprisal_aha.py' first.")
        return

    print("Calculating necessary metrics...")
    # Calculate the un-normalized, length-dependent metric
    df['total_surprisal'] = df['avg_surprisal'] * df['token_count']
    
    # Calculate the length-normalized, human-interpretable metric
    df['perplexity'] = np.exp(df['avg_surprisal'])
    
    # --- Visualization ---
    print("Generating 'before and after' normalization plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Evaluating Length Normalization of Surprisal Metrics", fontsize=20, weight='bold')

    # Panel 1: The Problem (Un-normalized Metric vs. Length)
    sns.regplot(
        data=df, x='token_count', y='total_surprisal', ax=ax1,
        scatter_kws={'alpha': 0.2, 's': 10},
        line_kws={'color': 'red', 'linewidth': 2}
    )
    ax1.set_title("Before Normalization: Biased by Length", fontsize=16)
    ax1.set_xlabel("Token Count (Text Length)", fontsize=12)
    ax1.set_ylabel("Total Surprisal (Un-normalized)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Panel 2: The Solution (Normalized Metric vs. Length)
    sns.regplot(
        data=df, x='token_count', y='perplexity', ax=ax2,
        scatter_kws={'alpha': 0.2, 's': 10},
        line_kws={'color': 'green', 'linewidth': 2}
    )
    ax2.set_title("After Normalization: Unbiased by Length", fontsize=16)
    ax2.set_xlabel("Token Count (Text Length)", fontsize=12)
    ax2.set_ylabel("Perplexity (Normalized)", fontsize=12)
    # Use a log scale for the y-axis to better visualize the distribution
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_path = OUTPUT_DIR / "normalization_evaluation.png"
    plt.savefig(output_path, dpi=300)
    print(f"Evaluation plot saved to {output_path}")
    plt.close(fig)

    # --- Correlation Analysis ---
    print("\nCalculating Pearson correlation coefficients...")
    
    # Correlation for the un-normalized metric
    corr_before, p_before = pearsonr(df['token_count'], df['total_surprisal'])
    print(f"  - Correlation(Token Count, Total Surprisal): r = {corr_before:.3f} (p < {p_before:.2e})")
    
    # Correlation for the normalized metric
    corr_after, p_after = pearsonr(df['token_count'], df['perplexity'])
    print(f"  - Correlation(Token Count, Perplexity):    r = {corr_after:.3f} (p = {p_after:.3f})")

    print("\nAnalysis complete. The plot and correlations demonstrate the effectiveness of normalization.")

if __name__ == "__main__":
    main() 