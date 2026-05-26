#!/usr/bin/env python
"""
Generates an interactive 3D visualization of SBERT embeddings.

This script specifically loads the embeddings for three key classes:
- Goodreads Sample Quote
- Google Books Length Matched Snippet
- Non-Literary Baseline

It then uses UMAP to reduce their dimensionality to 3D and creates an
interactive scatter plot with Plotly, which is saved as an HTML file.

To run, simply execute from the terminal:
    python 06_figures/visualize_3d_embeddings.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP


def load_embeddings_and_ids(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load vectors and their original indices from a .npz file."""
    try:
        with np.load(path) as data:
            if "vectors" not in data or "ids" not in data:
                print(f"ERROR: .npz file {path} is missing 'vectors' or 'ids' array.", file=sys.stderr)
                return None
            return data["vectors"], data["ids"]
    except FileNotFoundError:
        print(f"ERROR: Embedding file not found at {path}", file=sys.stderr)
        return None


def main():
    """Main function to generate and save the 3D UMAP plot."""
    # --- Hardcoded Parameters ---
    csv_path = pathlib.Path("data/processed/public/ALOFT.csv")
    sbert_dir = pathlib.Path("data/interim/dynamic_embeddings")
    out_dir = pathlib.Path("data/outputs/analysis/dynamic")

    # The three classes to visualize
    embedding_info = {
        "Goodreads Sample Quote": sbert_dir / "goodreads_sample_quote.npz",
        "Google Books Length Matched Snippet": sbert_dir / "google_books_length_matched_snippet.npz",
        "Non-Literary Baseline": sbert_dir / "non-literary_baseline.npz",
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Main DataFrame for Text Labels ---
    if not csv_path.exists():
        print(f"ERROR: Main CSV file not found at {csv_path}. Cannot get text labels.", file=sys.stderr)
        sys.exit(1)
    print(f"Loading main CSV from {csv_path} for text labels...")
    df_main = pd.read_csv(csv_path)

    # --- 2. Load Embeddings and Prepare Data ---
    all_vectors, all_labels, all_texts = [], [], []

    print("Loading embedding data...")
    for class_name, path in embedding_info.items():
        loaded_data = load_embeddings_and_ids(path)
        if loaded_data is None:
            sys.exit(1)  # Error message is printed in the load function
        
        vectors, ids = loaded_data
        
        # Check if the column exists in the dataframe
        if class_name not in df_main.columns:
            print(f"ERROR: Column '{class_name}' not found in {csv_path}", file=sys.stderr)
            sys.exit(1)
            
        texts = df_main.loc[ids, class_name].astype(str).tolist()

        all_vectors.append(vectors)
        all_labels.extend([class_name] * len(vectors))
        all_texts.extend(texts)
        print(f"  - Loaded {len(vectors)} vectors for '{class_name}'")

    X = np.vstack(all_vectors)

    # --- 3. Reduce Dimensionality with UMAP ---
    print("\nRunning UMAP to reduce dimensionality to 3D (this may take a minute)...")
    reducer = UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        transform_queue_size=0.0,
    )
    embedding_3d = reducer.fit_transform(X)

    # --- 4. Create and Save 3D Plot ---
    print("Generating interactive 3D plot...")
    plot_df = pd.DataFrame({
        "UMAP 1": embedding_3d[:, 0],
        "UMAP 2": embedding_3d[:, 1],
        "UMAP 3": embedding_3d[:, 2],
        "Category": all_labels,
        "Text": all_texts,
    })

    fig = px.scatter_3d(
        plot_df,
        x="UMAP 1",
        y="UMAP 2",
        z="UMAP 3",
        color="Category",
        hover_name="Text",
        title="3D UMAP Projection of Literary and Non-Literary Embeddings",
        opacity=0.8,
        color_discrete_map={
            "Goodreads Sample Quote": "royalblue",
            "Google Books Length Matched Snippet": "darkorange",
            "Non-Literary Baseline": "green"
        }
    )

    fig.update_traces(marker=dict(size=2.5))
    
    output_path = out_dir / "3d_embedding_visualization.html"
    fig.write_html(output_path)

    print(f"\nSuccess! Interactive 3D plot saved to:\n{output_path.resolve()}")


if __name__ == "__main__":
    main() 