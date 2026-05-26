#!/usr/bin/env python
"""
Visualizes aggregate statistics for a collection of Forward Flow (FF) journeys.

This script loads a FF journeys JSON file and generates a "spaghetti plot"
overlaying the semantic novelty journeys of the Top 50 texts with the highest
overall FF scores. This visualizes the typical "shape of novelty" for the
most semantically dynamic texts in the corpus.

The plot is saved as an HTML file in 'data/outputs/analysis/forward_flow/visualizations/'.

Usage:
    python 06_figures/visualize_aggregate_ff_journeys.py <path_to_journeys.json> <dataset_name>

Example:
    python 06_figures/visualize_aggregate_ff_journeys.py data/outputs/analysis/forward_flow/forward_flow_journeys_goodreads_popular_quote.json "Popular Quotes"
"""
import json
import argparse
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pathlib
import sys

def main():
    """Main function to load data and generate the plot."""
    parser = argparse.ArgumentParser(description="Visualize aggregate metrics for a collection of Forward Flow journeys.")
    parser.add_argument("journeys_json_path", type=pathlib.Path, help="Path to the input journeys JSON file.")
    parser.add_argument("dataset_name", type=str, help="A descriptive name for the dataset (e.g., 'Popular Quotes').")
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
        
    print(f"Loaded {len(all_journeys)} journeys. Generating aggregate visualization for '{args.dataset_name}'...")

    # --- Calculate final FF scores to find the top 50 ---
    journeys_with_scores = [
        (idx, journey, np.mean(journey))
        for idx, journey in all_journeys.items() if journey
    ]
    # Sort by score, descending
    journeys_with_scores.sort(key=lambda x: x[2], reverse=True)
    
    top_50_journeys = journeys_with_scores[:50]
    
    # --- Generate Spaghetti Plot ---
    fig = go.Figure()
    
    # The #1 journey, to be highlighted
    top_1_idx, top_1_journey, _ = top_50_journeys[0]

    # Plot the other 49 journeys first with low opacity
    for idx, journey, score in top_50_journeys[1:]:
        # Normalize word position to be 0-100%
        normalized_x = np.linspace(0, 100, len(journey))
        fig.add_trace(go.Scatter(
            x=normalized_x, 
            y=journey, 
            mode='lines', 
            line=dict(color='grey', width=1),
            opacity=0.5,
            hoverinfo='none'
        ))

    # Plot the #1 journey on top, highlighted
    normalized_x_top = np.linspace(0, 100, len(top_1_journey))
    fig.add_trace(go.Scatter(
        x=normalized_x_top, 
        y=top_1_journey, 
        mode='lines', 
        name=f'Highest FF Journey (Index: {top_1_idx})',
        line=dict(color='darkviolet', width=4),
    ))

    fig.update_layout(
        title=f"The Shape of Semantic Novelty: Top 50 Forward Flow Journeys<br><sup>Dataset: {args.dataset_name}</sup>",
        xaxis_title="Normalized Position in Text (%)",
        yaxis_title="Average Semantic Distance from Preceding Context",
        template='plotly_white',
        showlegend=True
    )

    # --- Save Plot ---
    output_path = output_dir / f"ff_aggregate_spaghetti_{args.dataset_name.replace(' ', '_').lower()}.html"
    fig.write_html(output_path)
    print(f"Saved top 50 journeys spaghetti plot to: {output_path}")

if __name__ == "__main__":
    main() 