#!/usr/bin/env python
"""
Visualizes the "Journey Archetypes" for a collection of Forward Flow journeys.

This script moves beyond simple average scores to classify texts based on the
*shape* of their semantic novelty journey. It measures the novelty at the
beginning versus the end of each text and plots them on a 2D scatter plot.

This reveals four primary archetypes:
1. Crescendo: Starts coherent, ends novel (setup/punchline structure).
2. Opening Gambit: Starts novel, ends coherent (explains a surprising premise).
3. Sustained Novelty: Consistently novel throughout.
4. Sustained Coherence: Consistently coherent throughout.

The interactive plot is saved as an HTML file, allowing for deep exploration of
the dominant creative structures within a corpus.

Usage:
    python 06_figures/visualize_journey_archetypes.py \\
        --journeys_json "data/outputs/analysis/forward_flow/forward_flow_journeys_popular.json" \\
        --aloft_csv "data/processed/public/ALOFT.csv" \\
        --column_name "Goodreads Popular Quote" \\
        --dataset_name "Popular Quotes"
"""
import json
import argparse
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pathlib
import sys

def get_journey_archetype_data(journeys_data, aloft_df, column_name):
    """
    Processes journey data to calculate start, end, and average novelty.
    """
    archetype_data = []
    # Ensure aloft_df index is of a type that can be checked against
    aloft_df.index = aloft_df.index.astype(str)
    
    for idx, journey in journeys_data.items():
        if not journey or len(journey) < 4:  # Need at least 4 steps for quartiles
            continue
        
        n = len(journey)
        quartile_len = max(1, n // 4) # Ensure at least one element is taken
        
        start_novelty = np.mean(journey[:quartile_len])
        end_novelty = np.mean(journey[-quartile_len:])
        avg_novelty = np.mean(journey)
        
        try:
            # The aloft_df index was set by the first column when loaded before, let's assume it's the index
            text = aloft_df.loc[str(idx), column_name]
        except (KeyError, ValueError):
            text = "Text not found"

        archetype_data.append({
            'original_index': idx,
            'start_novelty': start_novelty,
            'end_novelty': end_novelty,
            'avg_novelty': avg_novelty,
            'text': text
        })
    return pd.DataFrame(archetype_data)

def main():
    parser = argparse.ArgumentParser(description="Generate a Forward Flow Journey Archetype plot.")
    parser.add_argument("--journeys_json", type=pathlib.Path, required=True, help="Path to the FF journeys JSON file.")
    parser.add_argument("--aloft_csv", type=pathlib.Path, required=True, help="Path to the main ALOFT.csv file.")
    parser.add_argument("--column_name", type=str, required=True, help="The source column name in ALOFT.csv.")
    parser.add_argument("--dataset_name", type=str, required=True, help="A descriptive name for the plot.")
    args = parser.parse_args()

    # --- Setup Output Directory ---
    output_dir = args.journeys_json.parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Load Data ---
    print("--- Loading and processing data... ---")
    try:
        with open(args.journeys_json, 'r', encoding='utf-8') as f:
            all_journeys = json.load(f)
        # Use the first column as the index for lookup
        df_aloft = pd.read_csv(args.aloft_csv, index_col=0)
    except FileNotFoundError as e:
        print(f"ERROR: Could not find a required file: {e.filename}", file=sys.stderr)
        return
        
    archetype_df = get_journey_archetype_data(all_journeys, df_aloft, args.column_name)
    if archetype_df.empty:
        print("No valid journey data found to plot.")
        return

    # --- Calculate Quadrant Statistics ---
    x_median = archetype_df['start_novelty'].median()
    y_median = archetype_df['end_novelty'].median()

    # Classify each point
    is_crescendo = (archetype_df['start_novelty'] < x_median) & (archetype_df['end_novelty'] > y_median)
    is_gambit = (archetype_df['start_novelty'] > x_median) & (archetype_df['end_novelty'] < y_median)
    is_sustained_novelty = (archetype_df['start_novelty'] > x_median) & (archetype_df['end_novelty'] > y_median)
    is_sustained_coherence = (archetype_df['start_novelty'] < x_median) & (archetype_df['end_novelty'] < y_median)

    # Calculate percentages
    total_points = len(archetype_df)
    pct_crescendo = 100 * is_crescendo.sum() / total_points
    pct_gambit = 100 * is_gambit.sum() / total_points
    pct_novelty = 100 * is_sustained_novelty.sum() / total_points
    pct_coherence = 100 * is_sustained_coherence.sum() / total_points
    
    # Get 5 most extreme examples and add an archetype label for the legend
    crescendo_examples = archetype_df[is_crescendo].sort_values(by=['end_novelty', 'start_novelty'], ascending=[False, True]).head(5).assign(archetype='Coherent Start, Novel End')
    gambit_examples = archetype_df[is_gambit].sort_values(by=['start_novelty', 'end_novelty'], ascending=[False, True]).head(5).assign(archetype='Novel Start, Coherent End')
    novelty_examples = archetype_df[is_sustained_novelty].sort_values(by=['avg_novelty'], ascending=False).head(5).assign(archetype='Sustained High Novelty')
    coherence_examples = archetype_df[is_sustained_coherence].sort_values(by=['avg_novelty'], ascending=True).head(5).assign(archetype='Sustained Low Novelty')
    highlight_df = pd.concat([crescendo_examples, gambit_examples, novelty_examples, coherence_examples])


    # --- Create Visualization ---
    print("--- Generating Enhanced Journey Archetype plot... ---")
    fig = go.Figure()

    # Layer 1: 2D Density Heatmap
    fig.add_trace(go.Histogram2dContour(
        x=archetype_df['start_novelty'],
        y=archetype_df['end_novelty'],
        colorscale='Blues',
        showscale=False,
        name='Density'
    ))

    # Layer 2: Highlighted Examples
    fig.add_trace(go.Scatter(
        x=highlight_df['start_novelty'],
        y=highlight_df['end_novelty'],
        mode='markers',
        marker=dict(color='crimson', size=10, symbol='star'),
        text=highlight_df.apply(lambda row: f"<b>Archetype:</b> {row['archetype']}<br><b>Index:</b> {row['original_index']}<br><b>Overall FF Score:</b> {row['avg_novelty']:.2f}<br><b>Text:</b> {row['text'][:100]}...", axis=1),
        hoverinfo='text',
        name='Top 5 Examples per Archetype'
    ))

    # Quadrant Lines
    fig.add_vline(x=x_median, line_width=2, line_dash="dash", line_color="black")
    fig.add_hline(y=y_median, line_width=2, line_dash="dash", line_color="black")

    # Dynamic, Clear Quadrant Annotations
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", text=f"<b>Coherent Start, Novel End</b><br>({pct_crescendo:.1f}%)", showarrow=False, align="left", font=dict(color="#333", size=14))
    fig.add_annotation(x=0.98, y=0.02, xref="paper", yref="paper", text=f"<b>Novel Start, Coherent End</b><br>({pct_gambit:.1f}%)", showarrow=False, align="right", font=dict(color="#333", size=14))
    fig.add_annotation(x=0.98, y=0.98, xref="paper", yref="paper", text=f"<b>Sustained High Novelty</b><br>({pct_novelty:.1f}%)", showarrow=False, align="right", font=dict(color="#333", size=14))
    fig.add_annotation(x=0.02, y=0.02, xref="paper", yref="paper", text=f"<b>Sustained Low Novelty</b><br>({pct_coherence:.1f}%)", showarrow=False, align="left", font=dict(color="#333", size=14))

    fig.update_layout(
        title=dict(text=f"<b>Journey Archetypes of Semantic Novelty</b><br><sup>Dataset: {args.dataset_name}</sup>", font=dict(size=20)),
        xaxis_title="Average Forward Flow Score (First 25% of Text)",
        yaxis_title="Average Forward Flow Score (Last 25% of Text)",
        template='plotly_white',
        height=800,
        width=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # --- Save Plot ---
    output_path = output_dir / f"ff_archetype_plot_final_{args.dataset_name.replace(' ', '_').lower()}.html"
    fig.write_html(output_path)
    print(f"Saved final archetype plot to: {output_path}")

if __name__ == "__main__":
    main() 