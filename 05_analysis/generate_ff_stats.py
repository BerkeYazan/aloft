
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

def generate_ff_stats():
    """
    Performs a structured statistical analysis on pre-calculated Forward Flow
    metrics from the ALOFT dataset, generating a publication-ready CSV.
    """
    # --- 1. Load Pre-Calculated Data ---
    data_path = 'data/outputs/master_metrics/ALOFT_master_metrics.csv'
    if not os.path.exists(data_path):
        print(f"Error: Master data file not found at {data_path}")
        return

    print(f"Loading pre-calculated metrics from {data_path}...")
    df = pd.read_csv(data_path)
    print("Data loaded successfully.")

    # --- 2. Define Metric and Column Mappings for Forward Flow ---
    column_map = {
        'Forward Flow': {
            'Goodreads Sample Quote': 'sample_ff', 'Goodreads Popular Quote': 'popular_ff',
            'Google Books Length Matched Snippet': 'snippet_ff', 'T50 Quote': 't50_ff',
            'T50 Quote-Free Context Length Matched': 't50freelength_ff', 'Non-Literary Baseline': 'nonlit_ff'
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
    print("\nPerforming structured statistical analysis for Forward Flow...")

    # A. Perform Pairwise Comparisons
    print("Step 1: Running pairwise comparisons...")
    for metric_name, text_type_cols in column_map.items():
        for group1_name, group2_name, comparison_type in comparison_pairs:
            col1 = text_type_cols.get(group1_name)
            col2 = text_type_cols.get(group2_name)

            if not col1 or not col2 or col1 not in df.columns or col2 not in df.columns:
                continue
            g1_data, g2_data = df[col1].dropna(), df[col2].dropna()
            if len(g1_data) == 0 or len(g2_data) == 0: continue

            u_stat, p_value = mannwhitneyu(g1_data, g2_data, alternative='two-sided')
            effect_size = cliffs_delta(g1_data, g2_data)
            results.append({
                'Comparison Type': comparison_type, 'Metric': metric_name, 'Group 1': group1_name, 'Group 2': group2_name,
                'Mean Group 1': g1_data.mean(), 'Mean Group 2': g2_data.mean(), 'U-statistic': u_stat, 'p-value': p_value,
                'Effect Size (Cliff\'s Delta)': effect_size
            })

    # B. Perform Pooled "Extraordinary vs. Ordinary" Comparison
    print("Step 2: Running pooled 'Extraordinary vs. Ordinary' comparison...")
    extraordinary_types = ['Goodreads Sample Quote', 'Goodreads Popular Quote', 'T50 Quote']
    ordinary_types = ['Google Books Length Matched Snippet', 'T50 Quote-Free Context Length Matched', 'Non-Literary Baseline']
    metric_name = 'Forward Flow'
    text_type_cols = column_map[metric_name]

    extraordinary_data = [df[text_type_cols[t]].dropna() for t in extraordinary_types if t in text_type_cols and text_type_cols[t] in df.columns]
    ordinary_data = [df[text_type_cols[t]].dropna() for t in ordinary_types if t in text_type_cols and text_type_cols[t] in df.columns]

    if extraordinary_data and ordinary_data:
        pooled_extraordinary, pooled_ordinary = pd.concat(extraordinary_data, ignore_index=True), pd.concat(ordinary_data, ignore_index=True)
        u_stat, p_value = mannwhitneyu(pooled_extraordinary, pooled_ordinary, alternative='two-sided')
        effect_size = cliffs_delta(pooled_extraordinary, pooled_ordinary)
        results.append({
            'Comparison Type': 'Pooled', 'Metric': metric_name, 'Group 1': 'All Extraordinary', 'Group 2': 'All Ordinary',
            'Mean Group 1': pooled_extraordinary.mean(), 'Mean Group 2': pooled_ordinary.mean(), 'U-statistic': u_stat, 'p-value': p_value,
            'Effect Size (Cliff\'s Delta)': effect_size
        })

    # --- 5. Multiple Comparison Correction & Save ---
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("\nERROR: No valid comparison data was generated for Forward Flow.")
        print("This is likely because the '_ff' columns in your master CSV are empty or contain only NaN values.")
        print("Please check the 'ALOFT_master_metrics.csv' file.")
        return

    reject, p_adjusted, _, _ = multipletests(results_df['p-value'].dropna(), alpha=0.05, method='fdr_bh')
    results_df['p-value_adjusted'] = p_adjusted
    results_df['Significant (alpha=0.05)'] = reject
    results_df = results_df.sort_values(by=['Comparison Type', 'Metric', 'Effect Size (Cliff\'s Delta)'])
    
    output_path = 'data/outputs/analysis/forward_flow_publication_stats.csv'
    results_df.to_csv(output_path, index=False, float_format='%.4f')

    print(f"\nAnalysis complete. Tidy results table saved to: {output_path}")
    print(results_df.to_string())

if __name__ == '__main__':
    generate_ff_stats() 