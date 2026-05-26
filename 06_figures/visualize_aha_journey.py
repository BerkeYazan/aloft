#!/usr/bin/env python
"""
Visualizes the outputs of the "AHA moment" analysis scripts.

This script serves two primary functions, controlled by the `--mode` argument:

1.  `--mode compare`:
    Generates a publication-quality violin plot comparing the distribution of
    maximum surprisal or maximum semantic shift scores across all text
    categories. This provides a high-level overview of which categories
    contain more "AHA moments".

2.  `--mode journey`:
    Generates a detailed line plot showing the "AHA journey" for a single,
    specified quote. This allows for a deep dive into exactly which tokens
    caused the highest surprise or the largest semantic shift.

---

Usage Examples:

# 1. To generate a comparison plot for maximum surprisal:
python 06_figures/visualize_aha_journey.py --metric surprisal --mode compare

# 2. To plot the surprisal journey for a specific quote (e.g., index 42 from "Goodreads Popular Quote"):
python 06_figures/visualize_aha_journey.py --metric surprisal --mode journey --index 42 --column "Goodreads Popular Quote"

# 3. To plot the semantic shift journey for the same quote:
python 06_figures/visualize_aha_journey.py --metric semantic_shift --mode journey --index 42 --column "Goodreads Popular Quote"

"""
import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import config

# --- Configuration ---
# Define base directories for inputs and where to save plots
BASE_INPUT_DIR = config.BASE_OUTPUT_DIR
PLOT_OUTPUT_DIR = BASE_INPUT_DIR / "aha_visualizations"
DATA_CSV_PATH = config.DATA_CSV_PATH
SOURCES_TO_PREFIX = config.SOURCES_TO_ANALYZE

# Ensure the output directory for plots exists
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_comparison(metric_name: str):
    """
    Generates and saves a violin plot comparing max metric scores across categories.
    """
    print(f"Generating comparison plot for max {metric_name}...")

    # Define paths based on the metric
    if metric_name == 'surprisal':
        input_dir = BASE_INPUT_DIR / "surprisal_aha"
        metrics_file = input_dir / "surprisal_aha_metrics.csv"
        metric_col = "max_surprisal"
        title = "Distribution of Maximum Linguistic Surprise (AHA Moments)"
        xlabel = "Maximum GPT-2 Surprisal"
    elif metric_name == 'semantic_shift':
        input_dir = BASE_INPUT_DIR / "semantic_shift"
        metrics_file = input_dir / "semantic_shift_metrics.csv"
        metric_col = "max_shift"
        title = "Distribution of Maximum Semantic Shift (AHA Moments)"
        xlabel = "Maximum Semantic Distance"
    else:
        raise ValueError(f"Invalid metric: {metric_name}")

    if not metrics_file.exists():
        print(f"ERROR: Metrics file not found at {metrics_file}", file=sys.stderr)
        print("Please run the corresponding analysis script first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(metrics_file)
    df.dropna(subset=[metric_col], inplace=True)

    # --- Plotting ---
    median_order = df.groupby('source')[metric_col].median().sort_values().index
    palette = sns.color_palette("viridis", n_colors=len(median_order))

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.violinplot(
        data=df, y="source", x=metric_col, order=median_order, palette=palette,
        inner=None, cut=0, ax=ax
    )
    sns.stripplot(
        data=df, y="source", x=metric_col, order=median_order,
        color="black", alpha=0.2, jitter=0.3, size=2, ax=ax
    )
    sns.boxplot(
        data=df, y="source", x=metric_col, order=median_order, ax=ax,
        showfliers=False, width=0.15,
        boxprops={'facecolor':'white', 'edgecolor':'black', 'zorder': 5},
        medianprops={'color':'black', 'linewidth': 2, 'zorder': 6},
        whiskerprops={'color':'black', 'zorder': 6, 'linewidth': 1.5},
        capprops={'color':'black', 'zorder': 6, 'linewidth': 1.5}
    )

    ax.set_title(title, fontsize=18, weight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("", fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    sns.despine(ax=ax, left=True)
    fig.tight_layout()

    # Save the plot
    output_path = PLOT_OUTPUT_DIR / f"comparison_plot_{metric_name}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to {output_path}")
    plt.close(fig)


def plot_trifecta_journey(index: int, column_name: str):
    """
    Generates a triple-axis plot for a complete linguistic, semantic, and affective journey.
    """
    print(f"Plotting TRIFECTA journey for quote at index {index} from '{column_name}'...")

    prefix = SOURCES_TO_PREFIX.get(column_name)
    if not prefix:
        sys.exit(f"ERROR: Invalid column name: {column_name}")

    # --- Step 1: Load all three journey data files ---
    try:
        with open(BASE_INPUT_DIR / "surprisal_aha" / f"journeys_{prefix}.json", 'r') as f:
            surprisal_journey = json.load(f).get(str(index))
        with open(BASE_INPUT_DIR / "semantic_shift" / f"journeys_shift_{prefix}.json", 'r') as f:
            shift_journey = json.load(f).get(str(index))
        with open(BASE_INPUT_DIR / "sentiment_journey" / f"journeys_sentiment_{prefix}.json", 'r') as f:
            sentiment_journey = json.load(f).get(str(index))
    except FileNotFoundError as e:
        sys.exit(f"ERROR: A required journey file was not found. Please run all analysis scripts. Details: {e}")

    if not all([surprisal_journey, shift_journey, sentiment_journey]):
        sys.exit(f"ERROR: Journey data missing for index {index}. Please ensure all analysis scripts have run.")

    # --- Step 2: Final, robust alignment and data preparation ---
    # The word-level journeys are our "source of truth" for what constitutes a word.
    words_in_quote = [item[0] for item in shift_journey]

    # Create the master data structure that will hold all aligned metrics.
    aligned_data = []
    token_stream = iter(surprisal_journey)

    for i, word in enumerate(words_in_quote):
        word_data = {
            "word": word,
            "tokens": [],
            "token_surprisals": [],
            "shift": shift_journey[i][1],
            "sentiment": sentiment_journey[i][1]
        }
        
        reconstructed_word = ""
        # Greedily consume tokens until the reconstructed, stripped string matches the target word.
        while reconstructed_word.strip() != word:
            try:
                # The token from the JSON already has whitespace handled (e.g., " Grand")
                token, surprisal = next(token_stream)
                word_data["tokens"].append(token)
                word_data["token_surprisals"].append(surprisal)
                reconstructed_word += token
            except StopIteration:
                # This can happen if the last word doesn't perfectly match due to tokenization artifacts.
                break
        
        aligned_data.append(word_data)

    # --- Step 3: Plotting (Definitive Version) ---
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # Create axes ONCE
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))

    # Prepare data arrays for plotting
    all_tokens = []
    all_surprisals = []
    all_shifts = []
    all_sentiments = []
    x_tick_positions = []
    x_tick_labels = []

    current_x = 0
    for word_data in aligned_data:
        num_tokens_in_word = len(word_data["tokens"])
        
        # Surprisal (per-token)
        all_tokens.extend(word_data["tokens"])
        all_surprisals.extend(word_data["token_surprisals"])
        
        # Shift and Sentiment (per-word, stretched across tokens)
        all_shifts.extend([word_data["shift"]] * num_tokens_in_word)
        all_sentiments.extend([word_data["sentiment"]] * num_tokens_in_word)

        # Ticks and labels
        x_tick_positions.append(current_x + (num_tokens_in_word - 1) / 2)
        x_tick_labels.append(word_data["word"])
        if current_x > 0:
            ax1.axvline(x=current_x - 0.5, color='grey', linestyle='--', alpha=0.4)
        
        current_x += num_tokens_in_word

    x_axis = range(len(all_tokens))
    p1, = ax1.plot(x_axis, all_surprisals, color='tab:blue', marker='o', markersize=4, linestyle='-', label='Linguistic Surprisal (per-token)')
    p2, = ax2.plot(x_axis, all_shifts, color='tab:red', linestyle='--', label='Semantic Shift (per-word)')
    p3, = ax3.plot(x_axis, all_sentiments, color='tab:green', linestyle=':', label='Sentiment Polarity (per-word)')
    
    # --- Formatting ---
    ax1.set_ylabel('Linguistic Surprisal (GPT-2)', color='tab:blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylabel('Semantic Shift (SBERT)', color='tab:red', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax3.set_ylabel('Sentiment Polarity (RoBERTa)', color='tab:green', fontsize=14)
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax3.set_ylim(-1.05, 1.05)

    ax1.set_xticks(x_tick_positions)
    ax1.set_xticklabels(x_tick_labels, rotation=90, fontsize=9)
    ax1.set_xlabel('Words in Quote', fontsize=14)
    
    # Final Touches
    main_df = pd.read_csv(DATA_CSV_PATH)
    quote_text = main_df.iloc[index][column_name]
    plt.title(f"Combined Linguistic, Semantic & Affective Journey\nSource: '{column_name}' | Index: {index}", fontsize=18, weight='bold')
    fig.suptitle(f"“{quote_text}”", y=0.98, fontsize=12, style='italic')

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='tab:blue', lw=2, marker='o', label='Linguistic Surprisal (per-token)'),
        Line2D([0], [0], color='tab:red', lw=2, linestyle='--', label='Semantic Shift (per-word)'),
        Line2D([0], [0], color='tab:green', lw=2, linestyle=':', label='Sentiment Polarity (per-word)')
    ]
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    safe_col_name = column_name.replace(" ", "_").lower()
    output_path = PLOT_OUTPUT_DIR / f"trifecta_journey_token_level_{safe_col_name}_idx_{index}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Token-level trifecta journey plot saved to {output_path}")
    plt.close(fig)


def plot_combined_journey(index: int, column_name: str):
    """
    Generates a dual-axis plot showing both surprisal and semantic shift journeys.
    """
    print(f"Plotting COMBINED journey for quote at index {index} from '{column_name}'...")

    # --- Step 1: Load data for BOTH metrics ---
    SOURCES_TO_PREFIX = {
        "Goodreads Sample Quote": "sample",
        "Goodreads Popular Quote": "popular",
        "Google Books Length Matched Snippet": "snippet",
        "T50 Quote": "t50",
        "T50 Quote-Free Context Length Matched": "t50_free",
        "Non-Literary Baseline": "nonlit"
    }
    prefix = SOURCES_TO_PREFIX.get(column_name)
    if not prefix:
        # Simplified error handling
        sys.exit(f"ERROR: Invalid column name: {column_name}")

    # Load Surprisal Journey
    surprisal_journey_file = BASE_INPUT_DIR / "surprisal_aha" / f"journeys_{prefix}.json"
    with open(surprisal_journey_file, 'r', encoding='utf-8') as f:
        surprisal_journey_data = json.load(f).get(str(index))

    # Load Semantic Shift Journey
    shift_journey_file = BASE_INPUT_DIR / "semantic_shift" / f"journeys_shift_{prefix}.json"
    with open(shift_journey_file, 'r', encoding='utf-8') as f:
        shift_journey_data = json.load(f).get(str(index))

    if not surprisal_journey_data or not shift_journey_data:
        sys.exit(f"ERROR: Journey data missing for index {index}. Please ensure both analysis scripts have been run.")

    # --- Step 2: Align the data (map sub-word tokens to whole words) ---
    aligned_data = {}
    word_buffer = ""
    surprisal_buffer = []

    for token, score in surprisal_journey_data:
        word_buffer += token
        surprisal_buffer.append(score)
        if ' ' in token or token == surprisal_journey_data[-1][0]: # Word boundary
            word = word_buffer.strip()
            if word:
                aligned_data[word] = {
                    "surprisal": np.mean(surprisal_buffer), # Use mean surprisal for the word's tokens
                    "shift": 0.0 # Default value
                }
            word_buffer, surprisal_buffer = "", []

    for word, shift_score in shift_journey_data:
        if word in aligned_data:
            aligned_data[word]["shift"] = shift_score

    words = list(aligned_data.keys())
    surprisals = [d["surprisal"] for d in aligned_data.values()]
    shifts = [d["shift"] for d in aligned_data.values()]

    # --- Step 3: Plotting ---
    fig, ax1 = plt.subplots(figsize=(18, 8))

    # Plot Surprisal
    color1 = 'tab:blue'
    ax1.set_xlabel('Words in Quote', fontsize=12)
    ax1.set_ylabel('Linguistic Surprisal (GPT-2)', color=color1, fontsize=12)
    ax1.plot(words, surprisals, color=color1, marker='o', linestyle='-', label='Surprisal')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.tick_params(axis='x', rotation=90, labelsize=8)

    # Create a second y-axis for Semantic Shift
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Semantic Shift (SBERT)', color=color2, fontsize=12)
    ax2.plot(words, shifts, color=color2, marker='s', linestyle='--', label='Semantic Shift')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Final touches
    main_df = pd.read_csv(DATA_CSV_PATH)
    quote_text = main_df.iloc[index][column_name]
    plt.title(f"Combined Linguistic & Semantic Journey\nSource: '{column_name}' | Index: {index}", fontsize=16, weight='bold')
    fig.suptitle(f"“{quote_text}”", y=0.94, fontsize=10, style='italic')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    
    # Save the plot
    safe_col_name = column_name.replace(" ", "_").lower()
    output_path = PLOT_OUTPUT_DIR / f"combined_journey_{safe_col_name}_idx_{index}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Combined journey plot saved to {output_path}")
    plt.close(fig)


def plot_individual_journey(metric_name: str, index: int, column_name: str):
    """
    Generates and saves a line plot for a single quote's "AHA Journey".
    """
    print(f"Plotting {metric_name} journey for quote at index {index} from '{column_name}'...")

    # --- Step 1: Find the correct prefix and file paths ---
    SOURCES_TO_PREFIX = {
        "Goodreads Sample Quote": "sample",
        "Goodreads Popular Quote": "popular",
        "Google Books Length Matched Snippet": "snippet",
        "T50 Quote": "t50",
        "T50 Quote-Free Context Length Matched": "t50_free",
        "Non-Literary Baseline": "nonlit"
    }
    if column_name not in SOURCES_TO_PREFIX:
        print(f"ERROR: Invalid column name provided: '{column_name}'", file=sys.stderr)
        print(f"Valid options are: {list(SOURCES_TO_PREFIX.keys())}", file=sys.stderr)
        sys.exit(1)
    
    prefix = SOURCES_TO_PREFIX[column_name]

    if metric_name == 'surprisal':
        input_dir = BASE_INPUT_DIR / "surprisal_aha"
        journey_file = input_dir / f"journeys_{prefix}.json"
        ylabel = "Surprisal Score"
        title = "Linguistic Surprisal (AHA) Journey"
    elif metric_name == 'semantic_shift':
        # NOTE: semantic_shift journey file is not created by the base script,
        # but this shows how it *would* work.
        print("ERROR: Semantic shift journey visualization is not yet supported. Use --mode combined_journey instead.", file=sys.stderr)
        sys.exit(1)
    else:
        raise ValueError(f"Invalid metric: {metric_name}")

    # --- Step 2: Load the journey data ---
    if not journey_file.exists():
        print(f"ERROR: Journey file not found at {journey_file}", file=sys.stderr)
        sys.exit(1)

    with open(journey_file, 'r', encoding='utf-8') as f:
        all_journeys = json.load(f)

    journey_data = all_journeys.get(str(index))
    if not journey_data:
        print(f"ERROR: No journey data found for index {index} in {journey_file}", file=sys.stderr)
        sys.exit(1)
        
    # --- Step 3: Load the original quote text for context ---
    print(f"Loading main dataset from {DATA_CSV_PATH} to fetch original text...")
    if not DATA_CSV_PATH.exists():
        print(f"ERROR: Main data file not found at {DATA_CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    main_df = pd.read_csv(DATA_CSV_PATH) # Load with default integer index, matching the analysis script

    # Check if index is valid
    if not 0 <= index < len(main_df):
        print(f"ERROR: Provided index {index} is out of bounds for the dataset (size: {len(main_df)}).", file=sys.stderr)
        sys.exit(1)

    quote_text = main_df.iloc[index][column_name]

    # Check if the quote text is valid
    if pd.isna(quote_text):
        print(f"ERROR: The text for index {index} in column '{column_name}' is empty or NaN.", file=sys.stderr)
        sys.exit(1)
    
    # --- Step 4: Prepare data for plotting ---
    tokens = [item[0] for item in journey_data]
    scores = [item[1] for item in journey_data]
    
    # --- Step 5: Plotting ---
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(range(len(scores)), scores, marker='o', linestyle='-', color='b', label=ylabel)
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    
    # Highlight the peak
    max_score_idx = scores.index(max(scores))
    ax.plot(max_score_idx, max(scores), 'r*', markersize=15, label=f'Peak: {max(scores):.2f}')
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Tokens in Quote", fontsize=12)
    ax.set_title(f"{title}\nSource: '{column_name}' | Index: {index}", fontsize=14, weight='bold')
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
    fig.suptitle(f"“{quote_text}”", y=0.92, fontsize=10, style='italic') # Add quote as a sub-header
    
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust layout to make space for suptitle

    # Save the plot
    safe_col_name = column_name.replace(" ", "_").lower()
    output_path = PLOT_OUTPUT_DIR / f"{metric_name}_journey_{safe_col_name}_idx_{index}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Journey plot saved to {output_path}")
    plt.close(fig)


def main():
    """Parses arguments and runs the selected visualization mode."""
    parser = argparse.ArgumentParser(description="Visualize 'AHA moment' analysis results.")
    parser.add_argument(
        "--metric",
        type=str,
        required=False, # Make this optional
        choices=['surprisal', 'semantic_shift'],
        help="The metric to visualize. Required for 'compare' and 'journey' modes."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['compare', 'journey', 'combined_journey', 'trifecta_journey'],
        help="The visualization mode: 'compare', 'journey', 'combined_journey', or 'trifecta_journey' for the full 3-metric plot."
    )
    parser.add_argument(
        "--index",
        type=int,
        help="The row index of the quote to visualize (required for 'journey' mode)."
    )
    parser.add_argument(
        "--column",
        type=str,
        help="The column name of the quote to visualize (required for 'journey' mode)."
    )

    args = parser.parse_args()

    if args.mode == 'compare':
        if not args.metric:
            parser.error("--mode 'compare' requires the --metric argument.")
        plot_comparison(args.metric)
    elif args.mode == 'journey':
        if args.index is None or args.column is None or not args.metric:
            parser.error("--mode 'journey' requires --index, --column, and --metric arguments.")
        plot_individual_journey(args.metric, args.index, args.column)
    elif args.mode == 'combined_journey':
        if args.index is None or args.column is None:
            parser.error("--mode 'combined_journey' requires --index and --column arguments.")
        plot_combined_journey(args.index, args.column)
    elif args.mode == 'trifecta_journey':
        if args.index is None or args.column is None:
            parser.error("--mode 'trifecta_journey' requires --index and --column arguments.")
        plot_trifecta_journey(args.index, args.column)


if __name__ == "__main__":
    main() 