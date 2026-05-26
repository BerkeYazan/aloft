
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import numpy as np
import os
import logging
from typing import List, Dict
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pathlib

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
LOG = logging.getLogger(__name__)

# --- Helper function for Cliff's Delta ---
def cliffs_delta(a, b):
    m, n = len(a), len(b)
    if m == 0 or n == 0: return np.nan
    # Vectorized implementation for speed
    gt = np.sum(np.array(a)[:, None] > np.array(b))
    lt = np.sum(np.array(a)[:, None] < np.array(b))
    return (gt - lt) / (m * n)

# --- Metric Calculation for Stepwise Distance ---
def calculate_stepwise_distance(texts: List[str], model: SentenceTransformer) -> List[float | None]:
    """
    Computes stepwise distance for a list of documents. Returns a list of scores,
    with None for documents with fewer than 2 sentences.
    """
    docs_sents = [sent_tokenize(str(txt)) if pd.notna(txt) else [] for txt in texts]
    valid_indices = [i for i, sents in enumerate(docs_sents) if len(sents) >= 2]
    docs_to_process = [docs_sents[i] for i in valid_indices]

    if not docs_to_process:
        return [None] * len(texts)

    all_sents_flat = [s for doc in docs_to_process for s in doc]
    all_embeddings = model.encode(all_sents_flat, batch_size=128, show_progress_bar=True, convert_to_numpy=True)

    doc_scores = []
    start_idx = 0
    for count in [len(doc) for doc in docs_to_process]:
        end_idx = start_idx + count
        emb = all_embeddings[start_idx:end_idx]
        diffs = np.diff(emb, axis=0)
        swd = np.mean(np.sum(diffs**2, axis=1))
        doc_scores.append(swd)
        start_idx = end_idx

    final_scores = [None] * len(texts)
    for i, score in zip(valid_indices, doc_scores):
        final_scores[i] = score
    return final_scores

def generate_swd_stats():
    """
    Calculates Stepwise Distance (SWD) and performs a structured statistical
    analysis, generating a publication-ready CSV and a list of top examples.
    This script is resumable; it caches calculated scores.
    """
    # --- 1. Setup Paths and Load Data ---
    data_path = 'data/outputs/master_metrics/ALOFT_master_metrics.csv'
    output_dir = pathlib.Path('data/outputs/analysis/stepwise_distance')
    cache_path = output_dir / 'swd_scores_cache.csv'
    output_dir.mkdir(exist_ok=True)
    
    if not os.path.exists(data_path):
        LOG.error(f"Master data file not found at {data_path}")
        return

    LOG.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # --- 2. Define Corpora and Manage Caching ---
    corpora_map = {
        'Goodreads Sample Quote': 'sample_swd', 'Goodreads Popular Quote': 'popular_swd',
        'Google Books Length Matched Snippet': 'snippet_swd', 'T50 Quote': 't50_swd',
        'T50 Quote-Free Context Length Matched': 't50freelength_swd', 'Non-Literary Baseline': 'nonlit_swd'
    }
    
    # Load cached scores if they exist
    if cache_path.exists():
        LOG.info(f"Loading cached SWD scores from {cache_path}")
        swd_data = pd.read_csv(cache_path)
    else:
        LOG.info("No cache file found. Starting SWD calculations from scratch.")
        swd_data = pd.DataFrame(index=df.index)

    # --- 3. Calculate Missing SWD Scores ---
    model_loaded = False
    model = None
    
    for corpus_name, col_name in corpora_map.items():
        score_col = col_name
        text_col = f'{col_name}_text'

        # Check if we already have this data in our cache
        if score_col in swd_data.columns and text_col in swd_data.columns:
            LOG.info(f"Corpus '{corpus_name}' found in cache. Skipping calculation.")
            continue

        # If we need to calculate, ensure model is loaded
        if not model_loaded:
            LOG.info("Loading SBERT model (this may take a while)...")
            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            model_loaded = True
            
        LOG.info(f"Calculating SWD for '{corpus_name}'...")
        if corpus_name in df.columns:
            swd_data[text_col] = df[corpus_name] # Keep text for top examples
            swd_data[score_col] = calculate_stepwise_distance(df[corpus_name].tolist(), model)
            
            LOG.info(f"Saving updated cache to {cache_path}...")
            swd_data.to_csv(cache_path, index=False)
        else:
            LOG.warning(f"  Corpus '{corpus_name}' not found in data. Skipping.")

    # --- 4. Define the Structured Comparison Pairs ---
    comparison_pairs = [
        ('Goodreads Sample Quote', 'Google Books Length Matched Snippet', 'Within-Book'),
        ('T50 Quote', 'T50 Quote-Free Context Length Matched', 'Within-Book'),
        ('Goodreads Sample Quote', 'Non-Literary Baseline', 'Literary vs. Informational'),
        ('Goodreads Popular Quote', 'Non-Literary Baseline', 'Literary vs. Informational'),
        ('T50 Quote', 'Non-Literary Baseline', 'Literary vs. Informational'),
        ('Goodreads Popular Quote', 'Goodreads Sample Quote', 'Popular vs. Sample'),
        ('Goodreads Sample Quote', 'T50 Quote', 'Historical (Modern vs. Classic)'),
    ]
    
    # --- 5. Perform Statistical Analysis ---
    results = []
    LOG.info("\nPerforming structured statistical analysis...")

    # A. Pairwise Comparisons
    for group1_name, group2_name, comparison_type in comparison_pairs:
        col1 = corpora_map.get(group1_name)
        col2 = corpora_map.get(group2_name)
        if not col1 or not col2 or col1 not in swd_data.columns or col2 not in swd_data.columns: continue
        g1_data = swd_data[col1].dropna()
        g2_data = swd_data[col2].dropna()
        if len(g1_data) == 0 or len(g2_data) == 0: continue
        u_stat, p_value = mannwhitneyu(g1_data, g2_data, alternative='two-sided')
        results.append({
            'Comparison Type': comparison_type, 'Metric': 'Stepwise Distance',
            'Group 1': group1_name, 'Group 2': group2_name,
            'Mean Group 1': g1_data.mean(), 'Mean Group 2': g2_data.mean(),
            'U-statistic': u_stat, 'p-value': p_value,
            'Effect Size (Cliff\'s Delta)': cliffs_delta(g1_data, g2_data)
        })

    # B. Pooled Comparison
    extraordinary_types = ['Goodreads Sample Quote', 'Goodreads Popular Quote', 'T50 Quote']
    ordinary_types = ['Google Books Length Matched Snippet', 'T50 Quote-Free Context Length Matched', 'Non-Literary Baseline']
    
    extra_cols = [corpora_map[etype] for etype in extraordinary_types if etype in corpora_map and corpora_map[etype] in swd_data.columns]
    ord_cols = [corpora_map[otype] for otype in ordinary_types if otype in corpora_map and corpora_map[otype] in swd_data.columns]
    
    if not extra_cols or not ord_cols:
        LOG.error("Could not find columns for pooled analysis. Check corpora map and data file.")
    else:
        pooled_extraordinary = pd.concat([swd_data[col].dropna() for col in extra_cols])
        pooled_ordinary = pd.concat([swd_data[col].dropna() for col in ord_cols])

        if len(pooled_extraordinary) > 0 and len(pooled_ordinary) > 0:
            u_stat, p_value = mannwhitneyu(pooled_extraordinary, pooled_ordinary, alternative='two-sided')
            results.append({
                'Comparison Type': 'Pooled', 'Metric': 'Stepwise Distance',
                'Group 1': 'All Extraordinary', 'Group 2': 'All Ordinary',
                'Mean Group 1': pooled_extraordinary.mean(), 'Mean Group 2': pooled_ordinary.mean(),
                'U-statistic': u_stat, 'p-value': p_value,
                'Effect Size (Cliff\'s Delta)': cliffs_delta(pooled_extraordinary, pooled_ordinary)
            })

    # --- 6. Save Top 20 Examples ---
    extra_text_cols = [f'{corpora_map[etype]}_text' for etype in extraordinary_types if etype in corpora_map and f'{corpora_map[etype]}_text' in swd_data.columns]
    if extra_text_cols:
        
        dfs_to_concat = []
        for text_col in extra_text_cols:
            score_col = text_col.replace('_text', '')
            temp_df = swd_data[[text_col, score_col]].copy()
            temp_df.columns = ['text', 'score']
            dfs_to_concat.append(temp_df)

        all_extra_texts_with_scores = pd.concat(dfs_to_concat, ignore_index=True)
        top_20 = all_extra_texts_with_scores.dropna().sort_values(by='score', ascending=False).head(20)

        top_20_path = output_dir / "top_20_swd_examples.txt"
        with open(top_20_path, 'w', encoding='utf-8') as f:
            f.write("--- Top 20 Extraordinary Texts by Stepwise Distance ---\n\n")
            for _, row in top_20.iterrows():
                f.write(f"Score: {row['score']:.4f}\n")
                f.write(f"Text: {row['text']}\n\n")
        LOG.info(f"\nSaved top 20 examples to {top_20_path}")

    # --- 7. Finalize and Save Results Table ---
    if not results:
        LOG.error("No valid comparison data was generated. Exiting.")
        return
        
    results_df = pd.DataFrame(results)
    reject, p_adjusted, _, _ = multipletests(results_df['p-value'].dropna(), alpha=0.05, method='fdr_bh')
    results_df['p-value_adjusted'] = p_adjusted
    results_df['Significant (alpha=0.05)'] = reject

    final_columns = [
        'Comparison Type', 'Metric', 'Group 1', 'Group 2',
        'Mean Group 1', 'Mean Group 2', 'Effect Size (Cliff\'s Delta)',
        'p-value', 'p-value_adjusted', 'Significant (alpha=0.05)', 'U-statistic'
    ]
    results_df = results_df[final_columns].sort_values(by=['Comparison Type', 'Metric'])
    
    output_path = output_dir / 'stepwise_distance_publication_stats.csv'
    results_df.to_csv(output_path, index=False, float_format='%.4f')
    LOG.info(f"Analysis complete. Tidy results table saved to:\n{output_path}")
    print("\nPreview of the first 5 rows of the output:")
    print(results_df.head().to_string())

if __name__ == '__main__':
    generate_swd_stats() 