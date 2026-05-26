#!/usr/bin/env python
"""
Visualizes the semantic shift journey of a specific text.

This script loads the JSON output from 'analyze_semantic_shift.py' and generates
two interactive plots for a selected text using its original DataFrame index:
1.  A line plot showing the shift score at each token, highlighting the maximum shift.
2.  A bar chart "fingerprint" showing the same data in a different format.

The plots are saved as HTML files in 'data/outputs/analysis/semantic_shift/visualizations/'.

Usage:
    python 06_figures/visualize_single_journey.py <path_to_journeys.json> <index_to_visualize>

Example:
    python 06_figures/visualize_single_journey.py data/outputs/analysis/semantic_shift/journeys_shift_goodreads_popular_quote.json 3893
"""
import json
import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add project root to sys.path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

# --- Configuration ---
OUTPUT_DIR = config.BASE_OUTPUT_DIR / "semantic_shift" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_single_journey_line(journey_data, text_identifier):
    """
    Generates an interactive line plot of the semantic journey.
    """
    tokens = [item[0] for item in journey_data]
    shifts = [item[1] for item in journey_data]

    # Find the max shift to highlight it
    max_shift_value = 0
    max_shift_idx = -1
    # Start from index 1 since the first shift is always 0
    if len(shifts) > 1:
        max_shift_value = max(shifts[1:])
        max_shift_idx = shifts.index(max_shift_value)

    # Create figure
    fig = go.Figure()

    # Add the main line plot
    fig.add_trace(go.Scatter(
        x=list(range(len(tokens))),
        y=shifts,
        mode='lines+markers',
        name='Shift Score',
        line=dict(color='cornflowerblue', width=2),
        marker=dict(size=6)
    ))

    # Add a special marker for the max shift
    if max_shift_idx != -1:
        fig.add_trace(go.Scatter(
            x=[max_shift_idx],
            y=[max_shift_value],
            mode='markers',
            name='Max Shift',
            marker=dict(color='crimson', size=16, symbol='star'),
            text=[f"Max Shift: {max_shift_value:.4f}"],
            hoverinfo='text'
        ))

    fig.update_layout(
        title=f"Semantic Journey for Text: '{text_identifier}'",
        xaxis_title="Token Sequence",
        yaxis_title="Semantic Shift Score (Cosine Distance)",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(tokens))),
            ticktext=tokens,
            tickangle=45
        ),
        template='plotly_white',
        legend_title_text='Legend',
        hovermode='x unified'
    )
    
    return fig

def plot_journey_fingerprint_bar(journey_data, text_identifier):
    """
    Generates an interactive bar chart "fingerprint" of the semantic journey.
    """
    tokens = [item[0] for item in journey_data]
    shifts = [item[1] for item in journey_data]

    max_shift_value = 0
    max_shift_idx = -1
    if len(shifts) > 1:
        max_shift_value = max(shifts[1:])
        max_shift_idx = shifts.index(max_shift_value)

    colors = ['cornflowerblue',] * len(tokens)
    if max_shift_idx != -1:
        colors[max_shift_idx] = 'crimson'

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=tokens,
        y=shifts,
        marker_color=colors,
        text=shifts,
        texttemplate='%{text:.3f}',
        textposition='outside'
    ))

    fig.update_layout(
        title=f"Semantic Fingerprint for Text: '{text_identifier}'",
        xaxis_title="Tokens",
        yaxis_title="Semantic Shift Score",
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def main():
    """Main function to load data and generate plots."""
    parser = argparse.ArgumentParser(description="Visualize the semantic shift journey for a specific text.")
    parser.add_argument("journeys_json_path", type=str, help="Path to the input journeys JSON file.")
    parser.add_argument("text_index", type=str, help="The original DataFrame index of the text to visualize.")
    args = parser.parse_args()

    # Load the journey data
    try:
        with open(args.journeys_json_path, 'r', encoding='utf-8') as f:
            all_journeys = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Journeys file not found at {args.journeys_json_path}", file=sys.stderr)
        sys.exit(1)

    # Get the specific journey
    journey_to_plot = all_journeys.get(args.text_index)
    if not journey_to_plot:
        print(f"ERROR: Index '{args.text_index}' not found in the JSON file.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found journey for index '{args.text_index}'. Generating visualizations...")

    # --- Generate and save line plot ---
    fig_line = plot_single_journey_line(journey_to_plot, args.text_index)
    output_path_line = OUTPUT_DIR / f"journey_line_{args.text_index}.html"
    fig_line.write_html(output_path_line)
    print(f"Saved line plot to: {output_path_line}")
    
    # --- Generate and save bar plot ---
    fig_bar = plot_journey_fingerprint_bar(journey_to_plot, args.text_index)
    output_path_bar = OUTPUT_DIR / f"journey_bar_{args.text_index}.html"
    fig_bar.write_html(output_path_bar)
    print(f"Saved bar plot to: {output_path_bar}")

if __name__ == "__main__":
    main() 