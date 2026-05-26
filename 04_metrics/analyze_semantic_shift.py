#!/usr/bin/env python
"""
Calculates semantic shift metrics for texts in the ALOFT dataset.

This script identifies "AHA moments" by measuring how the semantic meaning of a
text shifts as it unfolds. It uses a sentence-transformer model to compute
embeddings for cumulatively longer parts of a text and measures the distance
between successive embeddings.

For each text, it calculates:
- Average Shift: The mean cosine distance between successive word additions.
- Maximum Shift: The largest single semantic shift, indicating a potential twist.

The script saves two types of outputs to `data/outputs/analysis/semantic_shift/`:
1. A CSV file (`semantic_shift_metrics.csv`) with the metrics for every text.
2. `.txt` files containing the top 20 texts from each category, ranked by their
   maximum semantic shift score.

To run, execute from the terminal:
    python 04_metrics/analyze_semantic_shift.py

For a quick test on 5% of the data:
    python 04_metrics/analyze_semantic_shift.py --sample_frac 0.05
"""
import warnings
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pathlib
import sys
from sentence_transformers import SentenceTransformer, util
import argparse
import json
import nltk
import os

# Add project root to sys.path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

# --- 0. Setup & Configuration ---
warnings.simplefilter("ignore")

def download_nltk_data_if_needed():
    """Checks for and downloads NLTK's 'punkt' tokenizer if not present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt', quiet=True)
        print("'punkt' downloaded.")

download_nltk_data_if_needed()

DATA_CSV_PATH = config.DATA_CSV_PATH
OUTPUT_DIR = config.BASE_OUTPUT_DIR / "semantic_shift"
MODEL_NAME = config.SBERT_MODEL
SOURCES_TO_ANALYZE = config.SOURCES_TO_ANALYZE

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- 1. Load Model ---
def load_model(device):
    """Loads and prepares the SentenceTransformer model."""
    print(f"Loading SBERT model '{MODEL_NAME}' on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print("Model loaded successfully.")
    return model

# --- 2. Semantic Shift Calculation ---
def get_semantic_shift_journey(text, model):
    """
    Calculates the mean/max semantic shift and the full journey for a text.
    Uses NLTK for robust tokenization to handle punctuation correctly.

    Args:
        text (str): The input text to analyze.
        model: The pre-loaded SentenceTransformer model.

    Returns:
        tuple: (average_shift, max_shift, max_shift_word, max_shift_index,
                max_shift_relative_position, journey).
    """
    if pd.isna(text) or not isinstance(text, str):
        return np.nan, np.nan, None, np.nan, np.nan, []
    
    tokens = nltk.word_tokenize(text)
    if len(tokens) < 2:
        return 0.0, 0.0, None, np.nan, np.nan, [] # No shift is possible

    try:
        # Create cumulative phrases: ["token1", "token1 token2", ...]
        cumulative_phrases = [" ".join(tokens[:i+1]) for i in range(len(tokens))]
        
        embeddings = model.encode(
            cumulative_phrases,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        shifts = []
        for i in range(embeddings.size(0) - 1):
            # Calculate cosine similarity and convert to distance
            sim = util.cos_sim(embeddings[i], embeddings[i+1]).item()
            distance = 1 - sim
            shifts.append(distance)

        if not shifts:
            return 0.0, 0.0, None, np.nan, np.nan, []
            
        shifts_arr = np.array(shifts)
        
        # Identify the max shift and its context
        max_shift_value = np.max(shifts_arr)
        max_shift_idx = np.argmax(shifts_arr)
        
        # The token causing the shift is at `max_shift_idx + 1`
        max_shift_token = tokens[max_shift_idx + 1]
        max_shift_token_index = max_shift_idx + 1
        max_shift_relative_position = (max_shift_token_index / len(tokens)) * 100 if len(tokens) > 0 else 0

        # The journey pairs each token (from the 2nd on) with the shift it caused
        shifts_with_zero_at_start = [0.0] + shifts
        journey = list(zip(tokens, shifts_with_zero_at_start))
        
        return (np.mean(shifts_arr), 
                max_shift_value, 
                max_shift_token, 
                max_shift_token_index, 
                max_shift_relative_position, 
                journey)

    except Exception:
        return np.nan, np.nan, None, np.nan, np.nan, []


# --- 3. Main Processing Logic ---
def main():
    """Main function to run the analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Calculate average and maximum semantic shift for texts using SBERT embeddings."
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=None,
        help="Fraction of the dataset to sample for a test run (e.g., 0.05 for 5%). By default, runs on the full dataset."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)

    # Load dataset
    if not DATA_CSV_PATH.exists():
        print(f"ERROR: Source CSV not found at {DATA_CSV_PATH}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading dataset from {DATA_CSV_PATH}...")
    df = pd.read_csv(DATA_CSV_PATH)
    print("Dataset loaded.")

    # Handle sampling for test runs
    if args.sample_frac:
        if not (0 < args.sample_frac <= 1):
            print("ERROR: --sample_frac must be between 0 and 1.", file=sys.stderr)
            sys.exit(1)
        print(f"Sampling {args.sample_frac * 100:.1f}% of the data for this run...")
        df = df.sample(frac=args.sample_frac, random_state=42).copy()
        print(f"Using {len(df)} rows for analysis.")

    all_metrics_results = []

    for src_col, prefix in SOURCES_TO_ANALYZE.items():
        if src_col not in df.columns:
            print(f"Warning: Column '{src_col}' not found, skipping.", file=sys.stderr)
            continue

        print(f"\nProcessing column: '{src_col}'...")
        
        temp_df = df[[src_col]].dropna().copy()
        temp_df.rename(columns={src_col: 'text'}, inplace=True)

        tqdm.pandas(desc=f"Calculating semantic shift for {prefix}")
        analysis_results = temp_df['text'].progress_apply(lambda x: get_semantic_shift_journey(x, model))
        
        temp_df['avg_shift'] = analysis_results.apply(lambda x: x[0])
        temp_df['max_shift'] = analysis_results.apply(lambda x: x[1])
        temp_df['max_shift_word'] = analysis_results.apply(lambda x: x[2])
        temp_df['max_shift_index'] = analysis_results.apply(lambda x: x[3])
        temp_df['max_shift_relative_position'] = analysis_results.apply(lambda x: x[4])
        temp_df['journey'] = analysis_results.apply(lambda x: x[5])
        temp_df['token_count'] = temp_df['journey'].apply(len)

        temp_df['source'] = src_col
        
        # --- Save top 20 examples ---
        top_20_max = temp_df.nlargest(20, 'max_shift')
        
        output_txt_path = OUTPUT_DIR / f"top_20_max_semantic_shift_{prefix}.txt"
        print(f"Saving top 20 max shift examples to {output_txt_path}...")
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Top 20 Examples from '{src_col}' ranked by Maximum Semantic Shift\n")
            f.write("="*80 + "\n\n")
            for idx, row in top_20_max.iterrows():
                f.write(f"Rank: {top_20_max.index.get_loc(idx) + 1}\n")
                f.write(f"Original Index: {idx}\n")
                f.write(f"Token Count: {row['token_count']}\n")
                f.write(f"Max Semantic Shift: {row['max_shift']:.4f}\n")
                if row['max_shift_word']:
                    f.write(f"  - Word: '{row['max_shift_word']}'\n")
                    f.write(f"  - Position: {int(row['max_shift_index'])} / {row['token_count']} ({row['max_shift_relative_position']:.1f}%)\n")
                f.write(f"Avg Semantic Shift: {row['avg_shift']:.4f}\n")
                f.write(f"Text:\n---\n{row['text']}\n---\n\n")
        
        # --- Save all journeys to a JSON file ---
        journeys_dict = {
            str(idx): data['journey']
            for idx, data in temp_df.iterrows() if data['journey']
        }
        output_json_path = OUTPUT_DIR / f"journeys_shift_{prefix}.json"
        print(f"Saving all {len(journeys_dict)} semantic shift journeys to {output_json_path}...")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(journeys_dict, f, ensure_ascii=False, indent=2)

        all_metrics_results.append(temp_df.drop(columns=['journey']))

    # --- 4. Save Combined Metrics ---
    if all_metrics_results:
        final_df = pd.concat(all_metrics_results)
        output_csv_path = OUTPUT_DIR / "semantic_shift_metrics.csv"
        print(f"\nSaving combined metrics to {output_csv_path}...")
        final_df.to_csv(output_csv_path, index_label='original_index')
        print("All processing complete.")
    else:
        print("No data was processed.")

if __name__ == "__main__":
    main() 