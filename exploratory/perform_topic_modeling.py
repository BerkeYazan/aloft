#!/usr/bin/env python
"""
Performs topic modeling on ALOFT sentence embeddings using BERTopic.

This script leverages the high-quality SBERT embeddings generated for the
ALOFT project to identify and analyze dominant themes within the corpus of
literary quotes. BERTopic is a modern topic modeling technique that combines
embeddings with clustering to find dense, semantically coherent groups of
documents, which are then interpreted as topics.

The script loads a set of embeddings and their corresponding texts, trains a
BERTopic model, and then saves the results, including the model itself, the
topic-document assignments, and several publication-quality visualizations.

Prerequisites:
    This script requires the `bertopic` library and its dependencies.
    Install it via pip:
    pip install bertopic "hdbscan>=0.8.29"

Usage:
    # To run with the default extraordinary corpora:
    python 05_analysis/perform_topic_modeling.py

    # To specify custom corpora:
    python 05_analysis/perform_topic_modeling.py \
        --embedding_files data/interim/dynamic_embeddings/goodreads_popular_quote.npz \
                          data/interim/dynamic_embeddings/goodreads_sample_quote.npz
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from tqdm import tqdm


# --- Download NLTK models if not present ---
# This check is moved inside main() to prevent it from running in parallel processes
# during coherence calculation, which causes messy logging.

# Define a custom tokenizer with lemmatization for BERTopic's vectorizer
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def calculate_topic_coherence(topic_model: BERTopic, texts: list[str]) -> pd.DataFrame:
    """
    Calculates the 'c_v' coherence score for each topic.
    """
    print("--- Calculating Topic Coherence (NPMI/c_v) ---")
    try:
        # Pre-tokenize texts for Gensim
        tokenized_docs = [doc.split() for doc in texts]
        dictionary = Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        
        # Extract topics as lists of words
        topics = topic_model.get_topics()
        topic_words = {topic: [word for word, _ in words] for topic, words in topics.items()}
        
        coherence_scores = {}
        # Use tqdm for a progress bar as this is the slowest part
        for topic_id, words in tqdm(topic_words.items(), desc="Calculating Topic Coherence"):
            if topic_id == -1: continue # Skip outlier topic
            
            # Gensim's CoherenceModel needs a list of lists of words
            cm = CoherenceModel(topics=[words],
                                texts=tokenized_docs,
                                corpus=corpus,
                                dictionary=dictionary,
                                coherence='c_v',
                                processes=-1) # Use all available cores
            coherence_scores[topic_id] = cm.get_coherence()
            
        coherence_df = pd.DataFrame.from_dict(coherence_scores, orient='index', columns=['Coherence'])
        return coherence_df

    except Exception as e:
        print(f"Could not calculate topic coherence: {e}", file=sys.stderr)
        return pd.DataFrame(columns=['Coherence'])


def load_data(
    embedding_paths: list[pathlib.Path], aloft_csv_path: pathlib.Path
) -> tuple[np.ndarray, list[str]]:
    """
    Loads embeddings and their corresponding texts from multiple sources.
    (This function is adapted from perform_clustering.py)
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


def visualize_topics_3d(topic_model: BERTopic, output_path: pathlib.Path):
    """
    Generates an interactive 3D visualization of the topics.
    """
    print("--- Generating 3D Topic Visualization ---")
    try:
        # Reduce dimensionality of topic embeddings to 3D
        topics_3d = UMAP(n_neighbors=2, n_components=3, min_dist=0.0, metric='cosine', random_state=42).fit_transform(topic_model.topic_embeddings_)
        
        # Get topic information
        topic_info = topic_model.get_topic_info()
        
        # Filter out the outlier topic (-1) for visualization
        topic_info_filtered = topic_info[topic_info.Topic != -1].copy()
        
        # Align 3D embeddings with the filtered topic info
        # The first embedding corresponds to the -1 topic, so we skip it
        df_plot = pd.DataFrame(topics_3d[1:len(topic_info_filtered) + 1], columns=['x', 'y', 'z'])
        df_plot['Topic'] = topic_info_filtered['Topic']
        df_plot['Name'] = topic_info_filtered['Name']
        df_plot['Size'] = topic_info_filtered['Count']

        # Ensure there are no NaN values that would break plotting
        df_plot.dropna(inplace=True)

        # Create a professional color scale
        colors = px.colors.sequential.Viridis
        
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=df_plot['x'],
            y=df_plot['y'],
            z=df_plot['z'],
            mode='markers+text',
            text=[f"T{row.Topic}" for row in df_plot.itertuples()],
            textposition="top center",
            hoverinfo='text',
            hovertext=df_plot.apply(lambda r: f"<b>{r.Name}</b><br>Size: {r.Size}", axis=1),
            marker=dict(
                size=df_plot['Size'],
                sizemode='area',
                sizeref=2.*max(df_plot['Size'])/(40.**2),
                sizemin=4,
                color=df_plot['Topic'],
                colorscale=colors,
                opacity=0.8,
                line=dict(width=0)
            )
        ))

        # Modern, clean aesthetic
        fig.update_layout(
            title=dict(text="3D Visualization of Thematic Topics", x=0.5),
            showlegend=False,
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
                zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )

        fig.write_html(str(output_path))
        print(f"Saved 3D topic visualization to {output_path}")

    except Exception as e:
        print(f"Could not generate 3D topic visualization: {e}", file=sys.stderr)


def save_topic_report(topic_model: BERTopic, output_path: pathlib.Path, coherence_df: pd.DataFrame):
    """
    Saves a comprehensive text report of the topic modeling analysis.
    """
    print("\n--- Generating Topic Analysis Report ---")
    
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info.Topic != -1])
    n_outliers = topic_info[topic_info.Topic == -1]['Count'].iloc[0]
    total_docs = topic_info['Count'].sum()

    # Merge coherence scores into the topic_info dataframe
    topic_info = topic_info.merge(coherence_df, left_on='Topic', right_index=True, how='left')

    with output_path.open("w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(" " * 27 + "TOPIC MODELING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("--- Model Summary ---\n")
        f.write(f"Number of topics found: {n_topics}\n")
        f.write(f"Number of outlier documents: {n_outliers} ({n_outliers / total_docs * 100:.2f}%)\n\n")

        f.write("="*80 + "\n")
        f.write(" " * 22 + "TOP 15 TOPICS & REPRESENTATIVE QUOTES\n")
        f.write("="*80 + "\n\n")

        for _, row in topic_info.head(16).iterrows(): # Top 15 + outlier topic
            if row.Topic == -1: continue # Skip detailed report for outliers
            
            coherence_str = f"{row['Coherence']:.4f}" if pd.notna(row['Coherence']) else "N/A"
            f.write(f"\n{'='*20} Topic {row.Topic}: {row.Name} (Size: {row.Count} | Coherence: {coherence_str}) {'='*20}\n")
            
            try:
                docs = topic_model.get_representative_docs(row.Topic)
                for i, doc in enumerate(docs):
                    f.write(f"  {i+1}. \"{doc}\"\n\n")
            except Exception:
                f.write("  Could not retrieve representative documents for this topic.\n\n")

    print(f"Saved topic analysis report to {output_path}")


def main():
    """Main function to run the topic modeling pipeline."""
    
    # --- Download NLTK models if not present ---
    # Moved here to prevent noisy logging from multiprocessing workers in gensim
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('wordnet')
        nltk.data.find('stopwords')
    except LookupError:
        print("Downloading required NLTK data (punkt, wordnet, stopwords)...")
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)

    parser = argparse.ArgumentParser(description="Perform topic modeling on ALOFT SBERT embeddings.")
    parser.add_argument(
        "--embedding_files",
        nargs="+",
        type=pathlib.Path,
        default=[
            pathlib.Path("data/interim/dynamic_embeddings/goodreads_popular_quote.npz"),
            pathlib.Path("data/interim/dynamic_embeddings/goodreads_sample_quote.npz"),
            pathlib.Path("data/interim/dynamic_embeddings/t50_quote.npz"),
        ],
        help="Paths to the .npz SBERT embedding files.",
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
        default=pathlib.Path("data/outputs/analysis/topic_modeling"),
        help="Directory to save output files and visualizations.",
    )
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings, texts = load_data(args.embedding_files, args.aloft_csv)
    
    print("\n--- Training BERTopic Model ---")
    print("Note: This may take several minutes depending on the data size.")

    # Create a custom stopword list to remove lemmatization artifacts and common noise
    custom_stopwords = list(ENGLISH_STOP_WORDS.union(['wa', 'ha', 'le', 'u', 've', 'nt', 'don', 'll', 're']))
    
    # We use a CountVectorizer with a lemmatizer and stop words for best practice topic representations
    vectorizer_model = CountVectorizer(
        tokenizer=LemmaTokenizer(),
        stop_words=custom_stopwords, 
        min_df=5, 
        ngram_range=(1, 2)
    )

    # Define the representation model with Maximal Marginal Relevance (MMR)
    representation_model = MaximalMarginalRelevance(diversity=0.3)

    # Explicitly define UMAP and HDBSCAN for more control and sensitivity
    # Lower min_cluster_size and min_samples to capture more niche topics and reduce outliers
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=15, 
        min_samples=5,
        metric='euclidean', 
        cluster_selection_method='eom', 
        prediction_data=True,
        gen_min_span_tree=True # Often helps with cluster selection
    )

    # Initialize BERTopic with all components
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        verbose=True,
        calculate_probabilities=True,
        nr_topics='auto' # Automatically reduce topics after initial clustering
    )
    
    # Fit the model and get topics and probabilities
    topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
    
    print("\n--- Analysis Complete: Top Topics ---")
    # Get and display the most frequent topics
    freq_topics = topic_model.get_topic_info()
    print(freq_topics.head(15))
    
    # --- Save Results ---
    print("\n--- Saving Results and Visualizations ---")
    
    # +++ Calculate Coherence Scores +++
    coherence_df = calculate_topic_coherence(topic_model, texts)

    # 0. Save a detailed text report
    save_topic_report(topic_model, args.output_dir / "topic_analysis_report.txt", coherence_df)
    
    # 1. Save the topic info DataFrame with coherence scores
    freq_topics_with_coherence = freq_topics.merge(coherence_df, left_on='Topic', right_index=True, how='left')
    results_path = args.output_dir / "topics_summary.csv"
    freq_topics_with_coherence.to_csv(results_path, index=False)
    print(f"Saved topic summary to {results_path}")
    
    # 2. Save the trained model
    model_path = args.output_dir / "bertopic_model"
    topic_model.save(str(model_path), serialization="safetensors")
    print(f"Saved BERTopic model to {model_path}")
    
    # --- Generate and Save Visualizations ---
    
    # 3. Visualize Topics in 3D
    visualize_topics_3d(topic_model, args.output_dir / "topics_3d_visualization.html")

    # 4. Intertopic Distance Map
    try:
        fig = topic_model.visualize_topics()
        out_path = args.output_dir / "intertopic_distance_map.html"
        fig.write_html(out_path)
        print(f"Saved intertopic distance map to {out_path}")
    except Exception as e:
        print(f"Could not generate intertopic distance map: {e}", file=sys.stderr)

    # 5. Topic Word Scores Bar Chart
    try:
        fig = topic_model.visualize_barchart(top_n_topics=12)
        out_path = args.output_dir / "topic_word_scores.html"
        fig.write_html(out_path)
        print(f"Saved topic word scores chart to {out_path}")
    except Exception as e:
        print(f"Could not generate word score chart: {e}", file=sys.stderr)

    # 6. Topic Hierarchy Dendrogram
    try:
        fig = topic_model.visualize_hierarchy(top_n_topics=20)
        out_path = args.output_dir / "topic_hierarchy.html"
        fig.write_html(out_path)
        print(f"Saved topic hierarchy dendrogram to {out_path}")
    except Exception as e:
        print(f"Could not generate hierarchy plot: {e}", file=sys.stderr)

    print("\n--- Topic Modeling Complete ---")

if __name__ == "__main__":
    main()