#!/usr/bin/env python
"""
Calculates both average and maximum GPT-2 surprisal for texts in the ALOFT dataset.

This script identifies "AHA moments" in texts by measuring the peak surprisal
value, in addition to the standard average surprisal. It processes all relevant
text columns from the main dataset CSV.

For each column, it calculates:
- Average Surprisal: The mean surprisal across all tokens in a text.
- Maximum Surprisal: The highest single-token surprisal in a text.

The script saves two types of outputs to `data/outputs/analysis/surprisal_aha/`:
1. A CSV file (`surprisal_aha_metrics.csv`) containing the calculated metrics
   for every text.
2. A series of `.txt` files, each containing the top 20 texts from a given
   category, ranked by their maximum surprisal score.

To run, execute from the terminal:
    python 04_metrics/analyze_surprisal_aha.py

For a quick test on 5% of the data:
    python 04_metrics/analyze_surprisal_aha.py --sample_frac 0.05
"""
import warnings
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pathlib
import sys
import json
import argparse
import config

# --- 0. Setup & Configuration ---
# Suppress specific warnings for cleaner output
warnings.simplefilter("ignore")

# Define paths and parameters by importing from the central config file
DATA_CSV_PATH = config.DATA_CSV_PATH
OUTPUT_DIR = config.BASE_OUTPUT_DIR / "surprisal_aha"
MODEL_NAME = config.GPT2_MODEL
SOURCES_TO_ANALYZE = config.SOURCES_TO_ANALYZE

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- 1. Load Model and Tokenizer ---
def load_model(device):
    """Loads and prepares the GPT-2 model and tokenizer."""
    print(f"Loading GPT-2 model '{MODEL_NAME}' on {device}...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

# --- 2. Enhanced Surprisal Calculation ---
def get_surprisal_journey(text, model, tokenizer, device):
    """
    Calculates token-level surprisals and returns the full journey.

    Args:
        text (str): The input text to analyze.
        model: The pre-loaded GPT-2 model.
        tokenizer: The pre-loaded GPT-2 tokenizer.
        device: The device (CPU/GPU) to run inference on.

    Returns:
        tuple: A tuple containing (average_surprisal, max_surprisal, journey),
               where journey is a list of [token, surprisal_score] pairs.
               Returns (np.nan, np.nan, []) for invalid input.
    """
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 1:
        return np.nan, np.nan, []

    try:
        # Encode the text and move to the specified device
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs.input_ids

        # We can't calculate surprisal for the first token as it has no context
        if input_ids.shape[1] < 2:
            return 0.0, 0.0, []

        with torch.no_grad():
            # Get model logits (the raw scores for each token in the vocabulary)
            outputs = model(**inputs)
            # Shift logits and labels for calculating per-token loss (surprisal)
            # The prediction for token i is based on logits at position i-1
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Use CrossEntropyLoss with reduction='none' to get per-token loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        surprisals = loss.cpu().numpy()
        
        # Decode tokens, skipping the first one which has no surprisal score
        tokens = [tokenizer.decode(token_id) for token_id in input_ids[0, 1:]]
        
        # Combine tokens with their surprisal scores
        journey = list(zip(tokens, surprisals.tolist()))
        
        if not journey:
            return np.nan, np.nan, []

        return np.mean(surprisals), np.max(surprisals), journey

    except Exception as e:
        print(f"Error processing text: '{text[:50]}...'. Error: {e}", file=sys.stderr)
        return np.nan, np.nan, []


# --- 3. Main Processing Logic ---
def main():
    """Main function to run the analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Calculate average and maximum GPT-2 surprisal, and the full surprisal journey for texts."
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

    model, tokenizer = load_model(device)

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
        
        # Create a temporary dataframe to avoid fragmentation issues
        temp_df = df[[src_col]].dropna().copy()
        temp_df.rename(columns={src_col: 'text'}, inplace=True)

        # Calculate surprisal metrics and the full journey
        tqdm.pandas(desc=f"Calculating surprisal for {prefix}")
        analysis_results = temp_df['text'].progress_apply(
            lambda x: get_surprisal_journey(x, model, tokenizer, device)
        )
        
        temp_df['avg_surprisal'] = analysis_results.apply(lambda x: x[0])
        temp_df['max_surprisal'] = analysis_results.apply(lambda x: x[1])
        temp_df['journey'] = analysis_results.apply(lambda x: x[2])
        temp_df['token_count'] = temp_df['journey'].apply(len)
        temp_df['source'] = src_col
        
        # --- Save top 20 examples by max surprisal ---
        top_20_max = temp_df.nlargest(20, 'max_surprisal')
        
        output_txt_path = OUTPUT_DIR / f"top_20_max_surprisal_{prefix}.txt"
        print(f"Saving top 20 max surprisal examples to {output_txt_path}...")
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Top 20 Examples from '{src_col}' ranked by Maximum Surprisal\n")
            f.write("="*80 + "\n\n")
            for idx, row in top_20_max.iterrows():
                f.write(f"Rank: {top_20_max.index.get_loc(idx) + 1}\n")
                f.write(f"Original Index: {idx}\n")
                f.write(f"Token Count: {row['token_count']}\n")
                f.write(f"Max Surprisal Score: {row['max_surprisal']:.4f}\n")
                f.write(f"Avg Surprisal Score: {row['avg_surprisal']:.4f}\n")
                f.write(f"Text:\n---\n{row['text']}\n---\n\n")

        # --- Save all journeys to a JSON file ---
        journeys_dict = {
            # Use original index from the main df for easy lookup
            str(idx): data['journey'] 
            for idx, data in temp_df.iterrows() if data['journey']
        }
        output_json_path = OUTPUT_DIR / f"journeys_{prefix}.json"
        print(f"Saving all {len(journeys_dict)} surprisal journeys to {output_json_path}...")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(journeys_dict, f, ensure_ascii=False, indent=2)

        # Drop the journey column before concatenating to keep the final CSV clean
        all_metrics_results.append(temp_df.drop(columns=['journey']))

    # --- 4. Save Combined Metrics ---
    if all_metrics_results:
        final_df = pd.concat(all_metrics_results)
        output_csv_path = OUTPUT_DIR / "surprisal_aha_metrics.csv"
        print(f"\nSaving combined metrics to {output_csv_path}...")
        final_df.to_csv(output_csv_path, index_label='original_index')
        print("All processing complete.")
    else:
        print("No data was processed. Please check column names and file paths.")

if __name__ == "__main__":
    main() 