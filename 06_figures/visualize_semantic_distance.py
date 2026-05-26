import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os

def load_embeddings(embedding_dir):
    """Loads all .npz embeddings from a directory and categorizes them."""
    print(f"--- Loading embeddings from: {embedding_dir} ---")
    embeddings_data = {}
    embedding_files = list(Path(embedding_dir).glob("*.npz"))
    if not embedding_files:
        print(f"Error: No .npz files found in {embedding_dir}")
        return None

    for f_path in embedding_files:
        try:
            # Use filename as the key, e.g., 'goodreads_popular_quote'
            key = f_path.stem
            embeddings_data[key] = np.load(f_path)['vectors']
            print(f"  - Loaded {key} ({embeddings_data[key].shape[0]} vectors)")
        except Exception as e:
            print(f"  - Failed to load {f_path}: {e}")
            
    return embeddings_data

def calculate_semantic_distance(embeddings_data):
    """Calculates the 'Ordinary Centroid' and measures distance to it."""
    print("\n--- Calculating Semantic Distances ---")
    
    ordinary_keys = [k for k in embeddings_data.keys() if 'google_books' in k or 't50_quote-free_context_length_matched' in k or 'non-literary' in k]
    extraordinary_keys = [k for k in embeddings_data.keys() if 'goodreads' in k or 't50_quote' == k]

    print(f"Ordinary sources: {ordinary_keys}")
    print(f"Extraordinary sources: {extraordinary_keys}")

    # Stack all ordinary embeddings to calculate the centroid
    ordinary_embeddings = np.vstack([embeddings_data[k] for k in ordinary_keys])
    ordinary_centroid = np.mean(ordinary_embeddings, axis=0, keepdims=True)
    print(f"Calculated 'Ordinary Centroid' from {ordinary_embeddings.shape[0]} vectors.")

    # Calculate the 'Extraordinary Centroid'
    extraordinary_embeddings = np.vstack([embeddings_data[k] for k in extraordinary_keys])
    extraordinary_centroid = np.mean(extraordinary_embeddings, axis=0, keepdims=True)
    print(f"Calculated 'Extraordinary Centroid' from {extraordinary_embeddings.shape[0]} vectors.")

    all_distances = []
    for key, embeds in embeddings_data.items():
        # Cosine similarity is between -1 and 1. Distance is 1 - similarity.
        distances = 1 - cosine_similarity(embeds, ordinary_centroid)
        is_extraordinary = 1 if key in extraordinary_keys else 0
        for i in range(embeds.shape[0]):
            all_distances.append({
                'source': key,
                'is_extraordinary': is_extraordinary,
                'sem_dist': distances[i, 0]
            })
            
    distances_df = pd.DataFrame(all_distances)
    print("\nSemantic Distance Summary Statistics:")
    print(distances_df.groupby('is_extraordinary')['sem_dist'].describe().round(4))
    
    return embeddings_data, ordinary_centroid, extraordinary_centroid, distances_df

def create_3d_visualization(embeddings_data, ordinary_centroid, extraordinary_centroid, distances_df, output_path):
    """Performs PCA and generates an interactive 3D plot."""
    print("\n--- Generating 3D Visualization ---")
    
    # Combine all data for PCA
    all_embeddings = np.vstack(list(embeddings_data.values()))
    source_labels = np.concatenate([[key] * embeds.shape[0] for key, embeds in embeddings_data.items()])
    
    # Perform PCA
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(all_embeddings)
    ordinary_centroid_3d = pca.transform(ordinary_centroid)
    extraordinary_centroid_3d = pca.transform(extraordinary_centroid)
    print("PCA dimensionality reduction complete.")

    # Prepare data for plotting
    plot_df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
    plot_df['source'] = source_labels
    plot_df['is_extraordinary'] = plot_df['source'].apply(lambda x: 1 if x in distances_df[distances_df['is_extraordinary']==1]['source'].unique() else 0)
    plot_df = pd.concat([plot_df, distances_df['sem_dist']], axis=1)

    fig = go.Figure()

    # Add scatter points for each class
    for val, name, color in [(0, "Ordinary", "royalblue"), (1, "Extraordinary", "crimson")]:
        df_sub = plot_df[plot_df['is_extraordinary'] == val]
        fig.add_trace(go.Scatter3d(
            x=df_sub['x'], y=df_sub['y'], z=df_sub['z'],
            mode='markers',
            marker=dict(size=2.5, color=color, opacity=0.7),
            name=name,
            hovertext=df_sub['source']
        ))

    # Add the Ordinary Centroid
    fig.add_trace(go.Scatter3d(
        x=ordinary_centroid_3d[:, 0], y=ordinary_centroid_3d[:, 1], z=ordinary_centroid_3d[:, 2],
        mode='markers',
        marker=dict(size=10, color='lime', symbol='diamond'),
        name='Ordinary Centroid'
    ))
    
    # Add the Extraordinary Centroid
    fig.add_trace(go.Scatter3d(
        x=extraordinary_centroid_3d[:, 0], y=extraordinary_centroid_3d[:, 1], z=extraordinary_centroid_3d[:, 2],
        mode='markers',
        marker=dict(size=10, color='yellow', symbol='diamond'),
        name='Extraordinary Centroid'
    ))

    # Update layout for a cool, modern theme
    fig.update_layout(
        title=dict(text="<b>Semantic Space: Extraordinary vs. Ordinary Texts</b>", x=0.5, font=dict(size=20)),
        template="plotly_dark",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        legend=dict(x=0.05, y=0.95)
    )

    # Save to interactive HTML
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"Interactive 3D visualization saved to '{output_path}'")


if __name__ == "__main__":
    dynamic_embedding_dir = "data/interim/dynamic_embeddings"
    output_viz_path = "data/outputs/master_metrics/explanation_plots/semantic_distance_3d_viz.html"
    
    embeddings = load_embeddings(dynamic_embedding_dir)
    
    if embeddings:
        embeddings, ordinary_centroid, extraordinary_centroid, distances_df = calculate_semantic_distance(embeddings)
        create_3d_visualization(embeddings, ordinary_centroid, extraordinary_centroid, distances_df, output_viz_path) 