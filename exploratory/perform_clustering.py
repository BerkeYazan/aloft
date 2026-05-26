#!/usr/bin/env python
"""
Performs clustering analysis on ALOFT sentence embeddings.

This script uses a combination of UMAP for dimensionality reduction and HDBSCAN
for density-based clustering to identify thematic groups within a corpus of
literary quotes. It is designed to work with the SBERT embeddings generated
for the ALOFTS project.

The script loads a set of embeddings, reduces their dimensionality, finds
semantically coherent clusters, and then analyzes the content of these
clusters by extracting their most representative documents.

The output includes a console report detailing the clusters and a high-quality
visualization of the clustered embedding space.

Usage:
    # To run with the default extraordinary corpora:
    python 05_analysis/perform_clustering.py

    # To specify custom corpora:
    python 05_analysis/perform_clustering.py \
        --embedding_files data/interim/dynamic_embeddings/goodreads_popular_quote.npz \
                          data/interim/dynamic_embeddings/goodreads_sample_quote.npz
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import hdbscan
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


def load_data(
    embedding_paths: list[pathlib.Path], aloft_csv_path: pathlib.Path
) -> tuple[np.ndarray, list[str]]:
    """
    Loads embeddings and their corresponding texts from multiple sources.

    Args:
        embedding_paths: A list of paths to .npz embedding files.
        aloft_csv_path: Path to the main ALOFT.csv data file.

    Returns:
        A tuple containing the pooled embeddings array and a list of texts.
    """
    print("--- Loading and Pooling Data ---")
    if not aloft_csv_path.exists():
        print(f"FATAL: Main data file not found at {aloft_csv_path}", file=sys.stderr)
        sys.exit(1)
    df_aloft = pd.read_csv(aloft_csv_path)

    all_vectors, all_texts = [], []
    for path in embedding_paths:
        if not path.exists():
            print(f"WARNING: Embedding file not found: {path}. Skipping.", file=sys.stderr)
            continue
        
        print(f"Loading {path.name}...")
        with np.load(path) as data:
            vectors = data["vectors"]
            ids = data["ids"]
            
            # The column name in the CSV is derived from the file stem
            col_name = path.stem.replace("_", " ").replace("glove-", "").replace("t50-", "t50 ").title()
            
            if col_name not in df_aloft.columns:
                print(f"WARNING: Column '{col_name}' not in ALOFT.csv. Skipping.", file=sys.stderr)
                continue
                
            texts = df_aloft.loc[ids, col_name].astype(str).tolist()
            
            all_vectors.append(vectors)
            all_texts.extend(texts)

    if not all_vectors:
        print("FATAL: No valid embedding data could be loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    return np.vstack(all_vectors), all_texts


def analyze_cluster_topics(df_results: pd.DataFrame) -> dict[int, list[str]]:
    """
    Analyzes cluster topics using a c-TF-IDF implementation.
    This treats all texts in a cluster as a single document for TF,
    and compares against the IDF of the entire corpus.

    Args:
        df_results: DataFrame containing 'text' and 'cluster' columns.

    Returns:
        A dictionary mapping cluster IDs to a list of top 5 topic words.
    """
    print("\n--- Analyzing Cluster Themes using c-TF-IDF ---")
    
    # Exclude noise points from topic analysis
    df_clustered = df_results[df_results['cluster'] != -1]
    docs_per_topic = df_clustered.groupby(['cluster'], as_index=False).agg({'text': ' '.join})
    
    if docs_per_topic.empty:
        print("No clusters found to analyze.")
        return {}

    # Custom c-TF-IDF implementation inspired by BERTopic
    def c_tf_idf(documents: pd.Series, m: int, ngram_range: tuple[int, int]=(1, 1)) -> tuple[np.ndarray, CountVectorizer]:
        count_vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
        t = count_vectorizer.transform(documents).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w, out=np.zeros_like(t.T, dtype=float), where=w!=0)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)
        return tf_idf, count_vectorizer

    try:
        tf_idf, count = c_tf_idf(docs_per_topic['text'], m=len(df_results))
        words = count.get_feature_names_out()
        labels = list(docs_per_topic.cluster)
        
        topic_words = {}
        for i, label in enumerate(labels):
            if label != -1:
                # Get top 5 words for each topic
                top_indices = np.argsort(tf_idf.T[i])[::-1][:5]
                topic_words[label] = [words[index] for index in top_indices]
        
        return topic_words
    except Exception as e:
        print(f"Could not perform c-TF-IDF: {e}", file=sys.stderr)
        return {}


def visualize_clusters(
    umap_embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: pathlib.Path
):
    """Generates and saves a 3D visualization of the clusters as an HTML file."""
    print("\n--- Visualizing Clusters in 3D (Interactive) ---")
    
    df_plot = pd.DataFrame(umap_embeddings, columns=('x', 'y', 'z'))
    df_plot['cluster'] = labels

    df_clusters = df_plot[df_plot['cluster'] != -1]
    df_noise = df_plot[df_plot['cluster'] == -1]

    unique_labels = sorted(df_clusters['cluster'].unique())
    
    # Use a professional, perceptually uniform color scale
    colors = px.colors.sequential.Viridis
    palette = {label: colors[int(i * (len(colors)-1) / len(unique_labels))] for i, label in enumerate(unique_labels)}

    fig = go.Figure()

    # Plot clustered points
    for cluster_id in unique_labels:
        cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]
        fig.add_trace(go.Scatter3d(
            x=cluster_data['x'],
            y=cluster_data['y'],
            z=cluster_data['z'],
            mode='markers',
            marker=dict(
                size=3,
                color=palette.get(cluster_id, 'black'),
                opacity=0.8,
                line=dict(width=0)  # Remove marker outlines
            ),
            name=f'Cluster {cluster_id}'
        ))

    # Plot noise points
    if not df_noise.empty:
        fig.add_trace(go.Scatter3d(
            x=df_noise['x'],
            y=df_noise['y'],
            z=df_noise['z'],
            mode='markers',
            marker=dict(
                size=2,
                color='#888888',  # Darker grey for visibility
                opacity=0.4,
                line=dict(width=0)
            ),
            name='Noise'
        ))

    # Modern, clean aesthetic
    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.5))
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)'
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Saved interactive 3D cluster visualization to {output_path}")


def save_analysis_report(
    output_path: pathlib.Path,
    df_results: pd.DataFrame,
    umap_embeddings: np.ndarray,
    topic_words_map: dict
):
    """Saves a comprehensive text report of the clustering analysis."""
    print(f"\n--- Generating Analysis Report ---")
    
    labels = df_results['cluster'].to_numpy()
    clustered_mask = labels != -1
    
    # --- Calculate Metrics ---
    n_clusters = len(np.unique(labels[clustered_mask]))
    n_noise = np.sum(~clustered_mask)
    
    metrics_text = "Clustering Metrics could not be computed."
    if n_clusters > 1:
        # Use UMAP embeddings for metric calculation as they were used for clustering
        clustered_embeddings = umap_embeddings[clustered_mask]
        clustered_labels = labels[clustered_mask]
        
        try:
            sil_score = silhouette_score(clustered_embeddings, clustered_labels, metric='euclidean')
            ch_score = calinski_harabasz_score(clustered_embeddings, clustered_labels)
            db_score = davies_bouldin_score(clustered_embeddings, clustered_labels)
            metrics_text = (
                f"Silhouette Score: {sil_score:.4f}\n"
                f"Calinski-Harabasz Score: {ch_score:.4f}\n"
                f"Davies-Bouldin Score: {db_score:.4f}"
            )
        except Exception as e:
            metrics_text = f"Error computing metrics: {e}"
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(" " * 25 + "CLUSTERING ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("--- Summary ---\n")
        f.write(f"Total points analyzed: {len(labels)}\n")
        f.write(f"Number of clusters found: {n_clusters}\n")
        f.write(f"Number of noise points: {n_noise} ({n_noise / len(labels) * 100:.2f}%)\n\n")

        f.write("--- Clustering Evaluation Metrics ---\n")
        f.write(metrics_text + "\n\n")

        f.write("="*80 + "\n")
        f.write(" " * 20 + "TOP CLUSTERS & REPRESENTATIVE QUOTES\n")
        f.write("="*80 + "\n\n")

        cluster_sizes = df_results[clustered_mask].groupby('cluster').size().sort_values(ascending=False)
        
        for cluster_id, size in cluster_sizes.head(10).items():
            f.write(f"\n{'='*20} Cluster {cluster_id} (Size: {size}) {'='*20}\n")
            
            # Display thematic words
            if cluster_id in topic_words_map:
                f.write(f"  THEME: {', '.join(topic_words_map[cluster_id])}\n\n")

            cluster_df = df_results[df_results['cluster'] == cluster_id]
            
            # Find 5 documents closest to the centroid in the original embedding space
            cluster_embeddings = np.vstack(cluster_df['embedding'].values)
            centroid = np.mean(cluster_embeddings, axis=0)
            similarities = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1)).flatten()
            top_indices = similarities.argsort()[-5:][::-1]
            
            for i, text in enumerate(cluster_df['text'].iloc[top_indices]):
                f.write(f"  {i+1}. \"{text}\"\n\n")
    
    print(f"Saved analysis report to {output_path}")


def main():
    """Main function to run the clustering pipeline."""
    parser = argparse.ArgumentParser(description="Perform clustering on ALOFT SBERT embeddings.")
    parser.add_argument(
        "--embedding_files",
        nargs="+",
        type=pathlib.Path,
        default=[
            pathlib.Path("data/interim/dynamic_embeddings/goodreads_popular_quote.npz"),
            pathlib.Path("data/interim/dynamic_embeddings/goodreads_sample_quote.npz"),
            pathlib.Path("data/interim/dynamic_embeddings/t50_quote.npz"),
        ],
        help="Paths to the .npz embedding files to cluster.",
    )
    parser.add_argument(
        "--aloft_csv",
        type=pathlib.Path,
        default=pathlib.Path("data/processed/public/ALOFT.csv"),
        help="Path to the main ALOFT.csv file.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("data/outputs/analysis/clustering"),
        help="Directory to save output files.",
    )
    args = parser.parse_args()

    embeddings, texts = load_data(args.embedding_files, args.aloft_csv)

    print("\n--- Performing Dimensionality Reduction and Clustering ---")
    # 1. Reduce dimensionality with UMAP
    print("Step 1: Reducing dimensions with UMAP...")
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    umap_embeddings = umap_model.fit_transform(embeddings)

    # 2. Cluster with HDBSCAN
    print("Step 2: Finding clusters with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom')
    cluster_labels = clusterer.fit_predict(umap_embeddings)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = np.sum(cluster_labels == -1)
    print(f"Found {n_clusters} clusters and {n_noise} noise points.")

    # 3. Analyze and print cluster contents
    df_results = pd.DataFrame({'text': texts, 'cluster': cluster_labels, 'embedding': list(embeddings)})
    
    # Get thematic words for each cluster
    topic_words_map = analyze_cluster_topics(df_results)

    # 4. Visualize the clusters using a 3D UMAP projection
    print("\nStep 3: Generating 3D projection for visualization...")
    umap_3d_model = UMAP(n_neighbors=15, n_components=3, min_dist=0.1, metric='cosine', random_state=42, n_jobs=1)
    umap_3d_embeddings = umap_3d_model.fit_transform(embeddings)
    visualize_clusters(umap_3d_embeddings, cluster_labels, args.output_dir / "cluster_visualization_3d.html")
    
    # 5. Save the full analysis to a text file
    save_analysis_report(
        args.output_dir / "cluster_analysis_report.txt",
        df_results,
        umap_embeddings, # Pass the UMAP embeddings used for clustering
        topic_words_map
    )


if __name__ == "__main__":
    main()