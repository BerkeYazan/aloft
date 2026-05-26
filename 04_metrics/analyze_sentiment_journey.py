#!/usr/bin/env python
"""
Calculates the sentiment journey and affective shift metrics for texts.

This script identifies "emotional twists" by measuring how the sentiment polarity
of a text changes as it unfolds word by word. It uses a RoBERTa-based sentiment
model to compute polarity scores for cumulatively longer parts of a text.

For each text, it calculates:
- Max Affective Shift: The largest single change in sentiment polarity from one
  word to the next, indicating an emotional turning point.
- Sentiment Volatility: The standard deviation of polarity scores, indicating
  how emotionally turbulent the text is.
- Final Sentiment: The final polarity score of the complete text.

The script saves outputs to `data/outputs/analysis/sentiment_journey/`:
1. A CSV file (`sentiment_journey_metrics.csv`) with the summary metrics.
2. JSON files (`journeys_sentiment_...json`) containing the detailed journey
   (word, polarity, shift) for every text.
3. Text files with the top 20 examples ranked by Max Affective Shift.

To run, execute from the terminal:
    python 04_metrics/analyze_sentiment_journey.py
"""
import argparse
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import pipeline
import re
import config

# --- 0. Setup & Configuration ---
warnings.simplefilter("ignore")

DATA_CSV_PATH = config.DATA_CSV_PATH
OUTPUT_DIR = config.BASE_OUTPUT_DIR / "sentiment_journey"
MODEL_NAME = config.SENTIMENT_MODEL
SOURCES_TO_ANALYZE = config.SOURCES_TO_ANALYZE

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_words_and_punctuation(text):
    """A robust function to split text into words and punctuation."""
    if not isinstance(text, str):
        return []
    # This regex finds sequences of word characters (including apostrophes)
    # or single punctuation characters.
    return re.findall(r"[\w']+|[.,!?;]", text)

# --- 1. Load Model ---
def load_sentiment_pipeline():
    """Loads and prepares the sentiment analysis pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading sentiment analysis model '{MODEL_NAME}' on {device}...")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=0 if device.type == "cuda" else -1,
        top_k=None # Get scores for all labels
    )
    print("Model loaded successfully.")
    return sentiment_pipe

# --- 2. Sentiment Journey Calculation ---
def get_sentiment_journey(text, sentiment_pipe):
    """
    Calculates the full sentiment journey for a text.
    """
    if pd.isna(text) or not isinstance(text, str):
        return np.nan, np.nan, np.nan, []
    
    words = text.split()
    if len(words) < 2:
        return 0.0, 0.0, 0.0, []

    try:
        # Create cumulative phrases and get sentiment for all of them in one batch
        cumulative_phrases = [" ".join(words[:i+1]) for i in range(1, len(words) + 1)]
        results = sentiment_pipe(cumulative_phrases, truncation=True, padding=True)

        polarities = []
        for res in results:
            scores = {d["label"].lower(): d["score"] for d in res}
            polarity = scores.get("positive", 0.0) - scores.get("negative", 0.0)
            polarities.append(polarity)

        if not polarities:
            return np.nan, np.nan, np.nan, []
        
        # Calculate shifts between consecutive polarity scores
        shifts = [0.0] + [abs(polarities[i] - polarities[i-1]) for i in range(1, len(polarities))]
        
        # Calculate final metrics
        max_affective_shift = np.max(shifts)
        sentiment_volatility = np.std(polarities)
        final_sentiment = polarities[-1]
        
        # Journey pairs each word with the polarity *after* it was added, and the shift it *caused*.
        journey = list(zip(words, polarities, shifts))
        
        return max_affective_shift, sentiment_volatility, final_sentiment, journey

    except Exception:
        return np.nan, np.nan, np.nan, []


# --- 3. Main Processing Logic ---
def main():
    """Main function to run the analysis pipeline."""
    parser = argparse.ArgumentParser(description="Calculate sentiment journey metrics for texts.")
    parser.add_argument("--sample_frac", type=float, default=None, help="Fraction of the dataset to sample for a test run.")
    args = parser.parse_args()

    sentiment_pipe = load_sentiment_pipeline()

    print(f"Loading dataset from {DATA_CSV_PATH}...")
    df = pd.read_csv(DATA_CSV_PATH)
    print("Dataset loaded.")

    if args.sample_frac:
        print(f"Sampling {args.sample_frac * 100:.1f}% of the data...")
        df = df.sample(frac=args.sample_frac, random_state=42).copy()
        print(f"Using {len(df)} rows.")

    all_metrics_results = []

    for src_col, prefix in SOURCES_TO_ANALYZE.items():
        if src_col not in df.columns:
            print(f"Warning: Column '{src_col}' not found, skipping.", file=sys.stderr)
            continue

        print(f"\nProcessing column: '{src_col}'...")
        temp_df = df[[src_col]].dropna().copy()
        temp_df.rename(columns={src_col: 'text'}, inplace=True)

        tqdm.pandas(desc=f"Calculating sentiment journey for {prefix}")
        analysis_results = temp_df['text'].progress_apply(lambda x: get_sentiment_journey(x, sentiment_pipe))
        
        temp_df['max_affective_shift'] = analysis_results.apply(lambda x: x[0])
        temp_df['sentiment_volatility'] = analysis_results.apply(lambda x: x[1])
        temp_df['final_sentiment'] = analysis_results.apply(lambda x: x[2])
        temp_df['journey'] = analysis_results.apply(lambda x: x[3])
        temp_df['word_count'] = temp_df['text'].apply(lambda x: len(x.split()))
        temp_df['source'] = src_col
        
        # Save top 20 examples
        top_20_max = temp_df.nlargest(20, 'max_affective_shift')
        output_txt_path = OUTPUT_DIR / f"top_20_max_affective_shift_{prefix}.txt"
        print(f"Saving top 20 examples to {output_txt_path}...")
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Top 20 Examples from '{src_col}' ranked by Maximum Affective Shift\n" + "="*80 + "\n\n")
            for idx, row in top_20_max.iterrows():
                f.write(f"Rank: {top_20_max.index.get_loc(idx) + 1}\n")
                f.write(f"Original Index: {idx}\n")
                f.write(f"Word Count: {row['word_count']}\n")
                f.write(f"Max Affective Shift: {row['max_affective_shift']:.4f}\n")
                f.write(f"Sentiment Volatility: {row['sentiment_volatility']:.4f}\n")
                f.write(f"Final Sentiment: {row['final_sentiment']:.4f}\n")
                f.write(f"Text:\n---\n{row['text']}\n---\n\n")
        
        # Save all journeys to a JSON file
        journeys_dict = {str(idx): data['journey'] for idx, data in temp_df.iterrows() if data['journey']}
        output_json_path = OUTPUT_DIR / f"journeys_sentiment_{prefix}.json"
        print(f"Saving all {len(journeys_dict)} sentiment journeys to {output_json_path}...")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(journeys_dict, f, ensure_ascii=False, indent=2)

        all_metrics_results.append(temp_df.drop(columns=['journey']))

    # Save Combined Metrics
    if all_metrics_results:
        final_df = pd.concat(all_metrics_results)
        output_csv_path = OUTPUT_DIR / "sentiment_journey_metrics.csv"
        print(f"\nSaving combined metrics to {output_csv_path}...")
        final_df.to_csv(output_csv_path, index_label='original_index')
        print("All processing complete.")

if __name__ == "__main__":
    main() 