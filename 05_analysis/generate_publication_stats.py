import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import numpy as np
import os

# --- Helper function for Cliff's Delta ---
def cliffs_delta(a, b):
    """Calculates Cliff's Delta effect size."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return np.nan
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (m * n)

def generate_publication_stats():
    """
    Performs a structured statistical analysis on pre-calculated metrics
    from the ALOFT dataset, generating a publication-ready CSV of results.
    This script reads existing metrics and does not re-calculate them.
    """
    # --- 1. Load Pre-Calculated Data ---
    data_path = 'data/outputs/master_metrics/ALOFT_master_metrics.csv'
    if not os.path.exists(data_path):
        print(f"Error: Master data file not found at {data_path}")
        print("Please run the ALOFT_traditional_metrics.ipynb notebook first to generate it.")
        return

    print(f"Loading pre-calculated metrics from {data_path}...")
    df = pd.read_csv(data_path)
    print("Data loaded successfully.")

    # --- 2. Define Metrics and Robust Column Mappings ---
    # This mapping directly links human-readable names to the exact column names
    # in the CSV, avoiding string construction errors.
    column_map = {
        'Flesch Reading Ease': {
            'Goodreads Sample Quote': 'sample_flesch', 'Goodreads Popular Quote': 'popular_flesch',
            'Google Books Length Matched Snippet': 'snippet_flesch', 'T50 Quote': 't50_flesch',
            'T50 Quote-Free Context Length Matched': 't50freelength_flesch', 'Non-Literary Baseline': 'nonlit_flesch'
        },
        'Coleman-Liau Index': {
            'Goodreads Sample Quote': 'sample_cl', 'Goodreads Popular Quote': 'popular_cl',
            'Google Books Length Matched Snippet': 'snippet_cl', 'T50 Quote': 't50_cl',
            'T50 Quote-Free Context Length Matched': 't50freelength_cl', 'Non-Literary Baseline': 'nonlit_cl'
        },
        'Lexical Diversity (MTLD)': {
            'Goodreads Sample Quote': 'sample_lex_div', 'Goodreads Popular Quote': 'popular_lex_div',
            'Google Books Length Matched Snippet': 'matched_snippet_lex_div', 'T50 Quote': 't50_quote_lex_div',
            'T50 Quote-Free Context Length Matched': 't50_quote_free_context_length_matched_lex_div',
            'Non-Literary Baseline': 'nonlit_baseline_lex_div'
        },
        'Shannon Entropy': {
            'Goodreads Sample Quote': 'sample_entropy', 'Goodreads Popular Quote': 'popular_entropy',
            'Google Books Length Matched Snippet': 'snippet_entropy', 'T50 Quote': 't50_entropy',
            'T50 Quote-Free Context Length Matched': 't50freelength_entropy', 'Non-Literary Baseline': 'nonlit_entropy'
        },
        'GPT-2 Surprisal': {
            'Goodreads Sample Quote': 'sample_surprisal', 'Goodreads Popular Quote': 'popular_surprisal',
            'Google Books Length Matched Snippet': 'snippet_surprisal', 'T50 Quote': 't50_surprisal',
            'T50 Quote-Free Context Length Matched': 't50freelength_surprisal', 'Non-Literary Baseline': 'nonlit_surprisal'
        },
        'Delta-PMI': {
            'Goodreads Sample Quote': 'sample_pmi', 'Goodreads Popular Quote': 'popular_pmi',
            'Google Books Length Matched Snippet': 'snippet_pmi', 'T50 Quote': 't50_pmi',
            'T50 Quote-Free Context Length Matched': 't50freelength_pmi', 'Non-Literary Baseline': 'nonlit_pmi'
        },
        'Sentiment Polarity': {
            'Goodreads Sample Quote': 'sample_sentiment_polarity', 'Goodreads Popular Quote': 'popular_sentiment_polarity',
            'Google Books Length Matched Snippet': 'snippet_sentiment_polarity', 'T50 Quote': 't50_sentiment_polarity',
            'T50 Quote-Free Context Length Matched': 't50freelength_sentiment_polarity', 'Non-Literary Baseline': 'nonlit_sentiment_polarity'
        }
    }

    # --- 3. Define the Structured Comparison Pairs ---
    comparison_pairs = [
        ('Goodreads Sample Quote', 'Google Books Length Matched Snippet', 'Within-Book'),
        ('T50 Quote', 'T50 Quote-Free Context Length Matched', 'Within-Book'),
        ('Goodreads Sample Quote', 'Non-Literary Baseline', 'Literary vs. Informational'),
        ('Goodreads Popular Quote', 'Non-Literary Baseline', 'Literary vs. Informational'),
        ('T50 Quote', 'Non-Literary Baseline', 'Literary vs. Informational'),
        ('Goodreads Popular Quote', 'Goodreads Sample Quote', 'Popular vs. Sample'),
        ('Goodreads Sample Quote', 'T50 Quote', 'Historical (Modern vs. Classic)'),
    ]

    # --- 4. Perform Statistical Analysis ---
    results = []
    print("\nPerforming structured statistical analysis...")

    # A. Perform Pairwise Comparisons
    print("Step 1: Running pairwise comparisons...")
    for metric_name, text_type_cols in column_map.items():
        for group1_name, group2_name, comparison_type in comparison_pairs:

            col1 = text_type_cols.get(group1_name)
            col2 = text_type_cols.get(group2_name)

            if not col1 or not col2 or col1 not in df.columns or col2 not in df.columns:
                continue

            g1_data = df[col1].dropna()
            g2_data = df[col2].dropna()

            if len(g1_data) == 0 or len(g2_data) == 0:
                continue

            u_stat, p_value = mannwhitneyu(g1_data, g2_data, alternative='two-sided')
            effect_size = cliffs_delta(g1_data, g2_data)

            results.append({
                'Comparison Type': comparison_type, 'Metric': metric_name,
                'Group 1': group1_name, 'Group 2': group2_name,
                'Mean Group 1': g1_data.mean(), 'Mean Group 2': g2_data.mean(),
                'U-statistic': u_stat, 'p-value': p_value,
                'Effect Size (Cliff\'s Delta)': effect_size
            })

    # B. Perform Pooled "Extraordinary vs. Ordinary" Comparison
    print("Step 2: Running pooled 'Extraordinary vs. Ordinary' comparison...")
    extraordinary_types = ['Goodreads Sample Quote', 'Goodreads Popular Quote', 'T50 Quote']
    ordinary_types = ['Google Books Length Matched Snippet', 'T50 Quote-Free Context Length Matched', 'Non-Literary Baseline']

    for metric_name, text_type_cols in column_map.items():
        # Collect all extraordinary data for the current metric
        extraordinary_data_series = [df[text_type_cols[etype]].dropna() for etype in extraordinary_types if etype in text_type_cols and text_type_cols[etype] in df.columns]
        # Collect all ordinary data for the current metric
        ordinary_data_series = [df[text_type_cols[otype]].dropna() for otype in ordinary_types if otype in text_type_cols and text_type_cols[otype] in df.columns]

        if not extraordinary_data_series or not ordinary_data_series:
            continue

        # Concatenate all series into two final datasets for comparison
        pooled_extraordinary = pd.concat(extraordinary_data_series, ignore_index=True)
        pooled_ordinary = pd.concat(ordinary_data_series, ignore_index=True)

        if len(pooled_extraordinary) == 0 or len(pooled_ordinary) == 0:
            continue

        u_stat, p_value = mannwhitneyu(pooled_extraordinary, pooled_ordinary, alternative='two-sided')
        effect_size = cliffs_delta(pooled_extraordinary, pooled_ordinary)

        results.append({
            'Comparison Type': 'Pooled', 'Metric': metric_name,
            'Group 1': 'All Extraordinary', 'Group 2': 'All Ordinary',
            'Mean Group 1': pooled_extraordinary.mean(), 'Mean Group 2': pooled_ordinary.mean(),
            'U-statistic': u_stat, 'p-value': p_value,
            'Effect Size (Cliff\'s Delta)': effect_size
        })


    # --- 5. Multiple Comparison Correction ---
    results_df = pd.DataFrame(results)
    p_values = results_df['p-value'].dropna()

    reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    results_df['p-value_adjusted'] = p_adjusted
    results_df['Significant (alpha=0.05)'] = reject

    # --- 6. Save the Tidy Results Table ---
    output_dir = 'data/outputs/analysis'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'publication_ready_inferential_stats.csv')
    
    results_df = results_df.sort_values(by=['Comparison Type', 'Metric', 'Effect Size (Cliff\'s Delta)'])

    final_columns = [
        'Comparison Type', 'Metric', 'Group 1', 'Group 2',
        'Mean Group 1', 'Mean Group 2', 'Effect Size (Cliff\'s Delta)',
        'p-value', 'p-value_adjusted', 'Significant (alpha=0.05)', 'U-statistic'
    ]
    results_df = results_df[final_columns]

    results_df.to_csv(output_path, index=False, float_format='%.4f')

    print(f"\nAnalysis complete. Tidy results table saved to:\n{output_path}")
    print(f"\nTotal tests conducted: {len(results_df)}")
    print(f"Total significant results (after correction): {results_df['Significant (alpha=0.05)'].sum()}")
    print("\nPreview of the first 5 rows of the output:")
    print(results_df.head().to_string())

if __name__ == '__main__':
    generate_publication_stats() 