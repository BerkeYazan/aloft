#!/usr/bin/env python
"""
Visualizes the Forward Flow (FF) journey of a specific text.

This script loads the JSON journey output from 'analyze_forward_flow.py' and
the main ALOFT CSV file. It generates an interactive line plot for a selected
text, showing how its semantic novelty evolves word by word.

The plot is saved as an HTML file in 'data/outputs/analysis/forward_flow/visualizations/'.

Usage:
    python 06_figures/visualize_single_ff_journey.py <path_to_journeys.json> <index_to_visualize> --aloft_csv <path_to_aloft.csv> --column_name <column_name>

Example:
    python 06_figures/visualize_single_ff_journey.py data/outputs/analysis/forward_flow/forward_flow_journeys_goodreads_popular_quote.json 3893 --aloft_csv data/processed/public/ALOFT.csv --column_name "Goodreads Popular Quote"
"""
import json
import argparse
import pandas as pd
import plotly.graph_objects as go
from nltk.tokenize import word_tokenize
import pathlib
import sys

# --- NLTK data download ---
def check_nltk_data():
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK 'punkt' model for word tokenization...")
        import nltk
        nltk.download("punkt", quiet=True)
check_nltk_data()


def main():
    """Main function to load data and generate the plot."""
    parser = argparse.ArgumentParser(description="Visualize the Forward Flow journey for a specific text.")
    parser.add_argument("journeys_json_path", type=pathlib.Path, help="Path to the input journeys JSON file.")
    parser.add_argument("text_index", type=str, help="The original DataFrame index of the text to visualize.")
    parser.add_argument("--aloft_csv", type=pathlib.Path, required=True, help="Path to the main ALOFT.csv file.")
    parser.add_argument("--column_name", type=str, required=True, help="The name of the column in ALOFT.csv where the text is located.")
    
    args = parser.parse_args()

    # --- Setup Output Directory ---
    output_dir = args.journeys_json_path.parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Load Data ---
    try:
        with open(args.journeys_json_path, 'r', encoding='utf-8') as f:
            all_journeys = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Journeys file not found at {args.journeys_json_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df_aloft = pd.read_csv(args.aloft_csv, index_col=0) # Assuming first col is index
    except FileNotFoundError:
        print(f"ERROR: ALOFT CSV not found at {args.aloft_csv}", file=sys.stderr)
        sys.exit(1)

    # --- Get Specific Data ---
    journey_to_plot = all_journeys.get(args.text_index)
    if not journey_to_plot:
        print(f"ERROR: Index '{args.text_index}' not found in the JSON file.", file=sys.stderr)
        sys.exit(1)
        
    try:
        text = df_aloft.loc[int(args.text_index), args.column_name]
        tokens = word_tokenize(text.lower())
    except (KeyError, ValueError):
        print(f"ERROR: Could not retrieve text for index '{args.text_index}' from the CSV.", file=sys.stderr)
        sys.exit(1)

    print(f"Found journey for index '{args.text_index}'. Generating visualization...")
    
    # The journey data starts from the 2nd step (word index 2), so we need to align it.
    # The x-axis ticks should correspond to the word that *completed* the context.
    # The first FF score is calculated at word index 1 (the 2nd word).
    x_axis_labels = tokens[1:] # Labels from the second word onwards
    
    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis_labels,
        y=journey_to_plot,
        mode='lines+markers',
        name='Forward Flow Step',
        line=dict(color='mediumpurple', width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=f"Forward Flow Journey for Text: '{args.text_index}'<br><sup>'{' '.join(tokens[:15])}...'</sup>",
        xaxis_title="Word in Sequence",
        yaxis_title="Average Semantic Distance from Preceding Context",
        template='plotly_white',
        hovermode='x unified'
    )
    
    # --- Save Plot ---
    output_path = output_dir / f"ff_journey_line_{args.text_index}.html"
    fig.write_html(output_path)
    print(f"Saved line plot to: {output_path}")

if __name__ == "__main__":
    main() 