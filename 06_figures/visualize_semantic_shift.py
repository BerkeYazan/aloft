#!/usr/bin/env python
"""
Generates a dynamic "constellation" plot to visualize semantic shift.

This script creates a single, interactive visualization for a sample of 50
quotes. In this plot:
- Each quote is a large red star.
- The dictionary definitions of the nouns within that quote are small blue diamonds.
- Lines connect each quote to its constituent noun definitions.
- HOVERING over a quote or its definitions will highlight its specific connecting
  lines, bringing it into focus.

All points are projected into a shared 2D space, and hovering over any point
reveals its full text content. The plot is zoomable and pannable.

To run, simply execute from the terminal:
    python 06_figures/visualize_semantic_shift.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spacy
from nltk.corpus import wordnet
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances


def calculate_and_save_statistics(
    df: pd.DataFrame,
    column: str,
    quote_vectors: np.ndarray,
    quote_ids: np.ndarray,
    def_text_to_vec: dict,
    nlp,
    out_path: pathlib.Path,
):
    """Calculates semantic shift stats for the entire dataset and saves a report."""
    print(f"\n--- Calculating Statistics for: {column} ---")
    
    all_shifts = []
    all_conceptual_diversities = [] # NEW: List to store diversity scores

    for i in tqdm(range(len(quote_ids)), desc=f"Analyzing {column}"):
        original_df_index = quote_ids[i]
        quote_text = df.loc[original_df_index, column]
        quote_vec = quote_vectors[i]

        doc = nlp(quote_text)
        noun_def_vectors = []
        for token in doc:
            if token.pos_ == "NOUN":
                noun = token.lemma_.lower()
                synsets = wordnet.synsets(noun, pos=wordnet.NOUN)
                if synsets:
                    definition = synsets[0].definition()
                    if definition in def_text_to_vec:
                        noun_def_vectors.append(def_text_to_vec[definition])
        
        if len(noun_def_vectors) > 0:
            def_centroid = np.mean(noun_def_vectors, axis=0)
            shift_distance = cosine_distances(quote_vec.reshape(1, -1), def_centroid.reshape(1, -1))[0, 0]
            all_shifts.append(shift_distance)

            # --- NEW: Calculate Conceptual Diversity ---
            if len(noun_def_vectors) > 1:
                # Calculate the average pairwise distance between all noun definitions
                distance_matrix = cosine_distances(noun_def_vectors)
                # Get the values from the upper triangle of the matrix (to avoid duplicates and self-distances)
                upper_triangle_indices = np.triu_indices_from(distance_matrix, k=1)
                diversity_score = np.mean(distance_matrix[upper_triangle_indices])
                all_conceptual_diversities.append(diversity_score)

    report = (
        f"Semantic Analysis Report for: '{column}'\n"
        f"{'='*40}\n\n"
        f"--- I. Semantic Shift (Magnitude) ---\n"
        f"Method: Cosine distance from quote vector to its noun-definition centroid.\n"
        f"Results (based on {len(all_shifts)} texts with nouns):\n"
        f"  - Mean Shift:   {np.mean(all_shifts):.4f}\n"
        f"  - Median Shift:  {np.median(all_shifts):.4f}\n"
        f"  - Std Dev:      {np.std(all_shifts):.4f}\n\n"
        
        f"--- II. Conceptual Diversity ---\n"
        f"Method: Average pairwise cosine distance between all noun definitions in a text.\n"
        f"Results (based on {len(all_conceptual_diversities)} texts with >1 noun):\n"
        f"  - Mean Diversity:   {np.mean(all_conceptual_diversities):.4f}\n"
        f"  - Median Diversity:  {np.median(all_conceptual_diversities):.4f}\n"
        f"  - Std Dev:        {np.std(all_conceptual_diversities):.4f}\n\n"

        f"Interpretation:\n"
        f"  - High 'Semantic Shift' suggests a large transformation of meaning.\n"
        f"  - High 'Conceptual Diversity' suggests the text combines semantically distant ideas.\n"
    )
    
    print("\n" + report)
    return report


def create_interactive_plot(quote_groups: list[dict], projected_vectors: np.ndarray, out_path: pathlib.Path):
    """
    Generates the advanced interactive Plotly visualization with hover effects.
    """
    print("Generating the interactive constellation plot with hover effects...")

    # Re-associate projections with their data in a more structured way
    vector_counter = 0
    for group in quote_groups:
        group["quote_projection"] = projected_vectors[vector_counter]
        vector_counter += 1
        for definition in group["definitions"]:
            definition["projection"] = projected_vectors[vector_counter]
            vector_counter += 1

    fig = go.Figure()

    # --- Create one trace per quote group for interactive highlighting ---
    for i, group in enumerate(quote_groups):
        quote_proj = group["quote_projection"]
        
        # Combine all points for this group into a single trace
        group_x = [quote_proj[0]]
        group_y = [quote_proj[1]]
        
        # Prepare hover text for the quote point
        def_list_for_hover = "<br>".join([f"- {d['text']}" for d in group['definitions']])
        hover_texts = [f"<b>QUOTE ({group['source']}):</b><br>{group['quote_text']}<br><br><b>NOUN DEFINITIONS:</b><br>{def_list_for_hover}"]
        
        line_paths_x = []
        line_paths_y = []

        for definition in group["definitions"]:
            def_proj = definition["projection"]
            group_x.append(def_proj[0])
            group_y.append(def_proj[1])
            hover_texts.append(definition["text"])
            
            # Add line segments for this group
            line_paths_x.extend([quote_proj[0], def_proj[0], None]) # None creates a break
            line_paths_y.extend([quote_proj[1], def_proj[1], None])

        # Add the connecting lines for this group as a separate, non-hoverable trace
        fig.add_trace(go.Scatter(
            x=line_paths_x,
            y=line_paths_y,
            mode='lines',
            line=dict(color='rgba(200, 200, 200, 0.2)', width=0.5),
            hoverinfo='none',
            showlegend=False
        ))

        # Choose color based on source
        quote_color = '#EF553B' if group['source'] == 'Goodreads Sample Quote' else '#00CC96'
        
        # Add the points (quote + definitions) for this group in a single trace
        fig.add_trace(go.Scatter(
            x=group_x,
            y=group_y,
            mode='markers',
            marker=dict(
                # First point is the quote, rest are definitions
                symbol=['circle'] + ['circle'] * len(group["definitions"]),
                color=[quote_color] + ['#636EFA'] * len(group["definitions"]),
                size=[10] + [6] * len(group["definitions"]),
                line=dict(width=1, color='#FFFFFF')
            ),
            text=hover_texts,
            hoverinfo='text',
            hoverlabel=dict(
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                font=dict(size=12)
            ),
            name=f'Quote {i+1}',
            showlegend=False # We will use a custom legend
        ))

    # --- Custom Legend ---
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Goodreads Quotes',
                             marker=dict(color='#EF553B', size=10, symbol='circle')))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Non-Literary Baseline',
                             marker=dict(color='#00CC96', size=10, symbol='circle')))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='Noun Definitions',
                             marker=dict(color='#636EFA', size=6, symbol='circle')))

    # --- Interactivity with FigureWidget ---
    # Note: This is a more declarative way to handle hover without needing a full FigureWidget
    # We can use layout updates to change trace properties on hover, if needed,
    # but the simplest robust method is to rely on Plotly's built-in highlighting.
    # For true line bolding on hover or changing line colors on click, a more complex Dash app would be required.
    # Currently, lines are static with reduced opacity to avoid visual clutter.
    # Hover effects are limited to point highlighting and hover text display.

    fig.update_layout(
        title=f'Semantic Shift Constellation Plot',
        xaxis=dict(title="Semantic Dimension 1 (PCA)", zeroline=False, showgrid=False),
        yaxis=dict(title="Semantic Dimension 2 (PCA)", zeroline=False, showgrid=False),
        legend_title="Point Type",
        hovermode='closest',
        template='plotly_white',
        plot_bgcolor='rgba(245, 245, 245, 1)'  # Light grey background
    )
    
    output_html_path = pathlib.Path("data/outputs/analysis/semantic_shift_interactive_plot.html")
    fig.write_html(output_html_path)
    print(f"\nInteractive constellation plot saved to: {output_html_path}")


def main():
    """Main execution pipeline."""
    # --- Configuration ---
    csv_path = pathlib.Path("data/processed/public/ALOFT.csv")
    quotes_embedding_path = pathlib.Path("data/interim/embeddings/goodreads_sample_quote.npz")
    non_literary_embedding_path = pathlib.Path("data/interim/embeddings/non-literary_baseline.npz")
    definitions_embedding_path = pathlib.Path("data/interim/embeddings/noun_definitions_embeddings.npz")
    columns_to_analyze = {"Goodreads Sample Quote": quotes_embedding_path, "Non-Literary Baseline": non_literary_embedding_path}
    num_samples_to_visualize = 50  # Total samples, split across both datasets
    output_html_path = pathlib.Path("data/outputs/analysis/semantic_shift_constellation_plot_with_lines.html")
    output_report_path = pathlib.Path("data/outputs/analysis/semantic_shift_full_report.txt")

    # --- 1. Load All Necessary Data ---
    print("Loading data and models...")
    if not all([p.exists() for p in [csv_path, quotes_embedding_path, non_literary_embedding_path, definitions_embedding_path]]):
        print("ERROR: One or more required input files are missing.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    embedding_data = {}
    for column, path in columns_to_analyze.items():
        with np.load(path) as data:
            embedding_data[column] = {"vectors": data["vectors"], "ids": data["ids"]}

    with np.load(definitions_embedding_path) as data:
        def_texts = data["definitions"]
        def_vectors = data["vectors"]

    def_text_to_vec = {text: vec for text, vec in zip(def_texts, def_vectors)}
    nlp = spacy.load("en_core_web_sm")

    # --- NEW: Calculate and save full statistics for both datasets ---
    print("\n--- Calculating Full Semantic Analysis for Both Datasets ---")
    full_report = ""
    for column in columns_to_analyze:
        quote_vectors = embedding_data[column]["vectors"]
        quote_ids = embedding_data[column]["ids"]
        
        # Pass the correct data to the updated function
        report_part = calculate_and_save_statistics(
            df, column, quote_vectors, quote_ids, def_text_to_vec, nlp, output_report_path
        )
        full_report += report_part + "\n" + "="*40 + "\n\n"

    output_report_path.write_text(full_report)
    print(f"Full statistical report saved to: {output_report_path}")

    # --- 2. Select Sample and Collect Data in Groups ---
    print(f"\nSelecting {num_samples_to_visualize} samples across both datasets and collecting their data...")
    samples_per_dataset = num_samples_to_visualize // len(columns_to_analyze)
    quote_groups = []
    all_vectors_for_pca = []
    vector_metadata = []
    group_counter = 0

    for column in columns_to_analyze:
        quote_vectors = embedding_data[column]["vectors"]
        quote_ids = embedding_data[column]["ids"]
        sample_indices = np.random.choice(len(quote_ids), samples_per_dataset, replace=False)
        
        for i, sample_idx in enumerate(tqdm(sample_indices, desc=f"Processing {column} samples")):
            original_df_index = quote_ids[sample_idx]
            quote_text = df.loc[original_df_index, column]
            quote_vec = quote_vectors[sample_idx]

            # Each group contains data for one quote and its definitions
            current_group = {
                "quote_text": quote_text,
                "quote_vector": quote_vec,
                "definitions": [],
                "source": column
            }

            # Add quote vector to the list for PCA
            all_vectors_for_pca.append(quote_vec)
            vector_metadata.append({"group_index": group_counter, "type": "quote", "source": column})

            # Find and add definition data
            doc = nlp(quote_text)
            for token in doc:
                if token.pos_ == "NOUN":
                    noun = token.lemma_.lower()
                    synsets = wordnet.synsets(noun, pos=wordnet.NOUN)
                    if synsets:
                        definition = synsets[0].definition()
                        if definition in def_text_to_vec:
                            def_vec = def_text_to_vec[definition]
                            current_group["definitions"].append({
                                "text": f"<b>{noun.upper()}:</b><br>{definition}",
                                "vector": def_vec
                            })
                            all_vectors_for_pca.append(def_vec)
                            vector_metadata.append({"group_index": group_counter, "type": "definition", "source": column})
            
            quote_groups.append(current_group)
            group_counter += 1

    # --- 3. Project All Vectors into 2D Space ---
    print("\nProjecting all collected vectors into 2D space using PCA...")
    if not all_vectors_for_pca:
        print("ERROR: No data to plot. Exiting.", file=sys.stderr)
        sys.exit(1)

    pca = PCA(n_components=2, random_state=42)
    projected_vectors = pca.fit_transform(np.array(all_vectors_for_pca))

    # --- 4. Re-associate Projections with their Data ---
    vector_counter = 0
    for group in quote_groups:
        group["quote_projection"] = projected_vectors[vector_counter]
        vector_counter += 1
        for definition in group["definitions"]:
            definition["projection"] = projected_vectors[vector_counter]
            vector_counter += 1

    # --- 5. Create the Plot ---
    create_interactive_plot(quote_groups, projected_vectors, output_html_path)


if __name__ == "__main__":
    main() 