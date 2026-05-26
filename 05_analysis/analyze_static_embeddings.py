#!/usr/bin/env python
"""
Runs a suite of analyses on the ALOFT dataset's static (GloVe) embeddings.

This script is hardcoded for a specific, reproducible analysis pipeline. It runs
a predefined list of comparisons using static (GloVe) embeddings, generating
a comprehensive text report and an interactive UMAP plot for each.

If a report file for a given analysis already exists, it will be skipped.

To run, simply execute from the terminal:
    python 05_analysis/analyze_static_embeddings.py

"""

from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, bootstrap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    pairwise_distances,
    roc_auc_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import StratifiedKFold
from umap import UMAP

pio.templates.default = "plotly_white"


def load_embeddings_and_ids(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vectors and their original indices from a .npz file."""
    try:
        with np.load(path) as data:
            if "vectors" not in data or "ids" not in data:
                raise KeyError("Arrays 'vectors' and 'ids' must be present in .npz file.")
            return data["vectors"], data["ids"]
    except FileNotFoundError:
        print(f"ERROR: Embedding file not found at {path}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Invalid .npz file {path}. {e}", file=sys.stderr)
        sys.exit(1)


def run_classification(X: np.ndarray, y: np.ndarray, class_names: list[str], all_texts: list[str] | None) -> str:
    """
    Train and evaluate a logistic regression classifier using cross-validation.
    Adapts to binary or multi-class scenarios. Returns a formatted report string.
    """
    print("-" * 80)
    print("Running 5-fold cross-validation with Logistic Regression...")

    n_classes = len(np.unique(y))
    is_binary = n_classes == 2

    # Configure model based on number of classes
    model_params = {"class_weight": "balanced", "random_state": 42, "max_iter": 1000}
    if is_binary:
        model_params["solver"] = "liblinear"
    else:
        model_params["multi_class"] = "multinomial"
        model_params["solver"] = "lbfgs"

    model = LogisticRegression(**model_params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_y_test, all_y_pred, all_y_proba = [], [], []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

    # Convert to NumPy arrays for easier manipulation
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    # --- Calculate metrics with confidence intervals using bootstrapping ---
    indices = np.arange(len(all_y_test))
    
    # 1. Accuracy
    acc = accuracy_score(all_y_test, all_y_pred)
    def acc_statistic(resampled_indices):
        return accuracy_score(all_y_test[resampled_indices], all_y_pred[resampled_indices])
    acc_res = bootstrap((indices,), acc_statistic, confidence_level=0.95, random_state=42)
    acc_ci = acc_res.confidence_interval

    # 2. ROC AUC (if binary)
    auc, auc_ci = None, None
    if is_binary:
        y_proba_positive = all_y_proba[:, 1]
        auc = roc_auc_score(all_y_test, y_proba_positive)
        def auc_statistic(resampled_indices):
            return roc_auc_score(all_y_test[resampled_indices], y_proba_positive[resampled_indices])
        auc_res = bootstrap((indices,), auc_statistic, confidence_level=0.95, random_state=42)
        auc_ci = auc_res.confidence_interval

    # --- Generate reports ---
    report_str = "I. EMBEDDING SPACE SEPARABILITY ANALYSIS (GloVe)\n"
    report_str += "----------------------------------------------\n\n"
    report_str += "Hypothesis:\n"
    report_str += "    The semantic embeddings of the classes are linearly separable, allowing a\n"
    report_str += "    classifier to predict class identity better than chance.\n\n"
    report_str += "Method:\n"
    report_str += "    A Logistic Regression classifier was trained using 5-fold stratified\n"
    report_str += "    cross-validation. Performance is measured by overall accuracy and, for\n"
    report_str += "    binary cases, the Area Under the ROC Curve (AUROC). Confidence intervals\n"
    report_str += "    (95% CI) are estimated using bootstrapping (n=9999) to assess stability.\n\n"
    report_str += "Results:\n"
    report_str += f"    - Overall Accuracy: {acc:.3f} (95% CI: [{acc_ci.low:.3f}, {acc_ci.high:.3f}])\n"

    if is_binary and auc is not None:
        report_str += f"    - ROC AUC Score:    {auc:.3f} (95% CI: [{auc_ci.low:.3f}, {auc_ci.high:.3f}])\n"

    report_str += "\n"
    report_str += classification_report(
        all_y_test, all_y_pred, target_names=class_names, zero_division=0
    )

    if not is_binary:
        report_str += "\n--- Confusion Matrix ---\n"
        cm = confusion_matrix(all_y_test, all_y_pred)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        report_str += cm_df.to_string()
        report_str += "\n(Rows = True Labels, Columns = Predicted Labels)\n"

    report_str += "\nInterpretation:\n"
    report_str += "    A high AUROC (>0.7) and accuracy, with tight confidence intervals, indicate\n"
    report_str += "    strong and stable linear separability between the classes. The per-class\n"
    report_str += "    precision/recall shows how well the model identifies individual categories.\n"

    # --- Qualitative Analysis: Add noteworthy examples if texts are available ---
    if all_texts:
        report_str += "\n\nIV. QUALITATIVE ANALYSIS: NOTEWORTHY EXAMPLES (GloVe)\n"
        report_str += "-----------------------------------------------------\n\n"
        report_str += "    The following are examples the model classified with the highest and\n"
        report_str += "    lowest confidence, providing insight into what the model finds easy\n"
        report_str += "    or confusing.\n\n"

        df_results = pd.DataFrame({
            'text': all_texts,
            'true_label': all_y_test,
            'pred_label': all_y_pred,
            'proba': np.max(all_y_proba, axis=1),
            'true_label_name': [class_names[i] for i in all_y_test],
            'correct': all_y_test == all_y_pred
        })

        for i, class_name in enumerate(class_names):
            class_df = df_results[df_results['true_label'] == i]
            
            # Most confident correct predictions
            confident_correct = class_df[class_df['correct']].sort_values('proba', ascending=False).head(3)
            report_str += f"    --- Most Confidently Correct for '{class_name}' ---\n"
            for _, row in confident_correct.iterrows():
                report_str += f"    - [Correct, P={row['proba']:.2f}] \"{row['text'][:100]}...\"\n"

            # Most confident incorrect predictions (if any)
            confident_incorrect = class_df[~class_df['correct']].sort_values('proba', ascending=False).head(3)
            if not confident_incorrect.empty:
                report_str += f"\n    --- Most Confidently Incorrect for '{class_name}' ---\n"
                for _, row in confident_incorrect.iterrows():
                    predicted_class_name = class_names[row['pred_label']]
                    report_str += f"    - [Misclassified as '{predicted_class_name}', P={row['proba']:.2f}] \"{row['text'][:100]}...\"\n"
            report_str += "\n"

    return report_str


def create_umap_plot(
    X: np.ndarray,
    labels: list[str],
    hover_texts: list[str] | None,
    n_classes: int,
    out_path: pathlib.Path,
):
    """
    Generate and save an interactive 2D UMAP visualization.
    """
    print("-" * 80)
    print("Generating UMAP projection (this may take a minute)...")

    reducer = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42,
        transform_queue_size=0.0,  # Recommended for MPS backend
    )
    embedding = reducer.fit_transform(X)

    df = pd.DataFrame({
        "UMAP 1": embedding[:, 0],
        "UMAP 2": embedding[:, 1],
        "Category": labels,
        "Text": hover_texts or [""] * len(X)
    })

    # Define a consistent color map for the two categories
    unique_categories = df["Category"].unique()
    color_map = {
        unique_categories[0]: "#1F77B4",  # Blue for the 'quote' or first class
        unique_categories[1]: "#D62728",  # Red for the 'context/baseline' or second class
    } if len(unique_categories) == 2 else None

    fig = px.scatter(
        df,
        x="UMAP 1",
        y="UMAP 2",
        color="Category",
        color_discrete_map=color_map,
        opacity=0.8, # Increased opacity for better visibility
        hover_name="Text",
        hover_data={"Category": True, "Text": False}
    )
    # Increase marker size and remove all non-data elements for a clean, publication-ready look
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        title_text='',      # Remove title
        showlegend=False,   # Remove legend
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,  # Hide entire x-axis
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,  # Hide entire y-axis
        ),
        plot_bgcolor='rgba(255, 255, 255, 1)', # Explicitly set white background
    )
    fig.write_html(out_path)

    # Also save a static, high-resolution PNG for direct use in the thesis
    png_path = out_path.with_suffix('.png')
    fig.write_image(png_path, scale=3) # Scale=3 for high resolution

    print(f"Saved interactive UMAP plot to {out_path}")
    print(f"Saved static UMAP image to {png_path}")


def create_distance_plot(dists1: np.ndarray, dists2: np.ndarray, label1: str, label2: str, out_path: pathlib.Path):
    """
    Creates and saves a raincloud plot to visualize distance distributions.
    """
    df = pd.DataFrame({
        'Distance': np.concatenate([dists1, dists2]),
        'Distribution': np.concatenate([
            np.full(len(dists1), label1),
            np.full(len(dists2), label2)
        ])
    })

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    ax = sns.violinplot(
        data=df, x='Distance', y='Distribution',
        orient='h', inner='quartile',
        palette='viridis', cut=0
    )
    plt.title(f'GloVe Distance Distribution Comparison\n({label1} vs. {label2})', fontsize=16)
    plt.xlabel('Cosine Distance', fontsize=12)
    plt.ylabel('')
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300)
    print(f"Saved distance distribution plot to {out_path}")
    plt.close()


def analyze_distances(
    vectors_list: list[np.ndarray], labels_list: list[str], out_dir: pathlib.Path, analysis_suffix: str
) -> str:
    """
    Calculates nearest-neighbor distances for every unique pair of classes.
    Returns a formatted report string.
    """
    print("-" * 80)
    print("Analyzing nearest-neighbor distances for all pairs...")
    report = "\n\nII. SEMANTIC DISTANCE & OUTLIER ANALYSIS (GloVe)\n"
    report += "--------------------------------------------------\n\n"
    report += "Hypothesis:\n"
    report += "    For a given pair of classes (A, B), items in class A are semantic\n"
    report += "    outliers relative to class B, meaning they are further from B than\n"
    report += "    they are from other items in their own class.\n\n"
    report += "Method:\n"
    report += "    For each pair of classes, a one-tailed Mann-Whitney U test was used to\n"
    report += "    compare inter-class vs. intra-class nearest-neighbor distances.\n\n"
    report += "Results:\n"

    # Iterate over all unique pairs of classes
    for (i, j) in itertools.combinations(range(len(vectors_list)), 2):
        vectors1, label1 = vectors_list[i], labels_list[i]
        vectors2, label2 = vectors_list[j], labels_list[j]

        n1, n2 = len(vectors1), len(vectors2)

        # 1. Inter-set distances: dist from each item in set 1 to its nearest in set 2
        inter_dist_matrix = cosine_distances(vectors1, vectors2)
        d1_to_2 = np.min(inter_dist_matrix, axis=1)

        # 2. Intra-set distances: dist from each item in set 1 to its nearest *other* in set 1
        intra1_dist_matrix = cosine_distances(vectors1)
        np.fill_diagonal(intra1_dist_matrix, np.inf)
        d1_to_1 = np.min(intra1_dist_matrix, axis=1)

        # 3. Statistical test and Effect Size
        stat, p_value = mannwhitneyu(d1_to_2, d1_to_1, alternative="greater")
        
        # Calculate Cliff's Delta as effect size
        effect_size = 1 - (2 * stat) / (n1 * n2)

        pair_report = (
            f"Comparison: '{label1}' vs. '{label2}'\n"
            f"--------------------------------------------------\n"
            f"  - H₀: The distance from a '{label1}' item to the nearest '{label2}' item is the same\n"
            f"    as the distance from a '{label1}' item to its nearest neighbor within its own class.\n"
            f"  - H₁: '{label1}' items are further from '{label2}' items than they are from each other.\n\n"
            f"  - Mean dist ({label1} -> nearest {label2}): {np.mean(d1_to_2):.4f}\n"
            f"  - Mean dist ({label1} -> nearest {label1}): {np.mean(d1_to_1):.4f}\n\n"
            f"  - Mann-Whitney U test (p-value): {p_value:.4g}\n"
            f"  - Cliff's Delta (Effect Size): {effect_size:.3f}\n\n"
            f"  - Interpretation: A small p-value (< 0.05) suggests that '{label1}' items are\n"
            f"    semantic outliers relative to the '{label2}' class. The effect size indicates\n"
            f"    the magnitude of this distinction (closer to 1 is a larger effect).\n\n"
        )
        report += pair_report

        # --- Create and save the distance distribution plot for this pair ---
        plot_path = out_dir / f"glove_distance_dist{analysis_suffix}_{label1.replace(' ', '_')}_vs_{label2.replace(' ', '_')}.png"
        create_distance_plot(
            d1_to_1, d1_to_2,
            f"Intra-Class: '{label1}' to nearest '{label1}'",
            f"Inter-Class: '{label1}' to nearest '{label2}'",
            plot_path
        )

    report += "\nInterpretation:\n"
    report += "    A small p-value (<0.05) supports the outlier hypothesis for a given pair,\n"
    report += "    suggesting a meaningful semantic distinction. Cliff's Delta\n"
    report += "    quantifies the size of this distinction.\n"

    return report

def run_structural_analysis(X: np.ndarray, y: np.ndarray, vectors_list: list[np.ndarray], labels_list: list[str]) -> str:
    """
    Calculates global structural metrics for the embedding space.
    Returns a formatted report string.
    """
    print("-" * 80)
    print("Calculating global and structural metrics...")
    report = "\n\nIII. GLOBAL & STRUCTURAL ANALYSIS (GloVe)\n"
    report += "-----------------------------------------\n\n"
    report += "Hypothesis:\n"
    report += "    The overall distributions of the classes exhibit different global\n"
    report += "    properties, such as central location and internal diversity.\n\n"
    report += "Method:\n"
    report += "    1. Centroid Distance: Cosine distance between class mean vectors.\n"
    report += "    2. Intra-Class Dispersion: Average pairwise cosine distance within a class.\n"
    report += "    3. Silhouette Score: Measures cluster density and separation.\n\n"
    report += "Results:\n"

    # 1. Centroid Distances
    report += "    - Centroid Distances:\n"
    centroids = [np.mean(v, axis=0, keepdims=True) for v in vectors_list]
    for (i, j) in itertools.combinations(range(len(vectors_list)), 2):
        dist = cosine_distances(centroids[i], centroids[j])[0, 0]
        report += f"        - {labels_list[i]} vs. {labels_list[j]}: {dist:.4f}\n"

    # 2. Intra-Class Dispersion
    report += "\n    - Intra-Class Dispersion (Semantic Diversity):\n"
    for i, vectors in enumerate(vectors_list):
        if len(vectors) > 1:
            # Sample to keep computation fast for large datasets
            sample_size = min(len(vectors), 1000)
            sample_indices = np.random.choice(len(vectors), sample_size, replace=False)
            dispersion = np.mean(pairwise_distances(vectors[sample_indices], metric="cosine"))
            report += f"        - {labels_list[i]}: {dispersion:.4f}\n"

    # 3. Silhouette Score
    if len(vectors_list) > 1 and len(X) > 1:
        # Sample to keep computation fast
        sample_size = min(len(X), 5000)
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        score = silhouette_score(X[sample_indices], y[sample_indices], metric="cosine")
        report += f"\n    - Mean Silhouette Score: {score:.4f}\n"
    
    report += "\nInterpretation:\n"
    report += "    - Centroid Distance: Higher value = greater global separation.\n"
    report += "    - Dispersion: Higher value = more internal semantic diversity.\n"
    report += "    - Silhouette Score: Value closer to 1 indicates dense, well-separated\n"
    report += "      clusters. Value near 0 indicates overlapping clusters.\n"

    return report


def main():
    # --- Hardcoded Parameters for Reproducibility ---

    # The full path to your main data file.
    # IMPORTANT: Update this if you move the project.
    csv_path = pathlib.Path("data/processed/public/ALOFT.csv")
    
    # Base directories for embeddings and outputs.
    static_embedding_dir = pathlib.Path("data/interim/static_embeddings")
    out_dir = pathlib.Path("data/outputs/analysis/static")

    # --- List of Analyses to Run ---
    # Each item is a list of paths for a single comparison using GloVe embeddings.
    analyses_to_run = [
        # 1. Within-Book (Goodreads)
        [
            static_embedding_dir / "glove_goodreads_sample_quote.npz",
            static_embedding_dir / "glove_google_books_length_matched_snippet.npz",
        ],
        # 2. Within-Book (T50)
        [
            static_embedding_dir / "glove_t50_quote.npz",
            static_embedding_dir / "glove_t50_quote-free_context_length_matched.npz",
        ],
        # 3. Literary vs. Informational (Sample)
        [
            static_embedding_dir / "glove_goodreads_sample_quote.npz",
            static_embedding_dir / "glove_non-literary_baseline.npz",
        ],
        # 4. Literary vs. Informational (Popular) - NEW
        [
            static_embedding_dir / "glove_goodreads_popular_quote.npz",
            static_embedding_dir / "glove_non-literary_baseline.npz",
        ],
        # 5. Literary vs. Informational (T50) - NEW
        [
            static_embedding_dir / "glove_t50_quote.npz",
            static_embedding_dir / "glove_non-literary_baseline.npz",
        ],
        # 6. Popularity Analysis
        [
            static_embedding_dir / "glove_goodreads_popular_quote.npz",
            static_embedding_dir / "glove_goodreads_sample_quote.npz",
        ],
        # 7. Historical Analysis - NEW
        [
            static_embedding_dir / "glove_goodreads_sample_quote.npz",
            static_embedding_dir / "glove_t50_quote.npz",
        ],
    ]
    # --- End of Hardcoded Parameters ---

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the main dataframe once
    df = None
    if csv_path.exists():
        print(f"Loading main CSV from {csv_path} for text labels...")
        df = pd.read_csv(csv_path)
    else:
        print(f"WARNING: Main CSV file not found at {csv_path}. Plots will not have text labels.", file=sys.stderr)

    # --- Pooled Analysis Definition ---
    # Define the groups for the special pooled analysis
    pooled_extraordinary_stems = ["glove_goodreads_popular_quote", "glove_goodreads_sample_quote", "glove_t50_quote"]
    pooled_ordinary_stems = ["glove_google_books_length_matched_snippet", "glove_t50_quote-free_context_length_matched", "glove_non-literary_baseline"]

    # --- Main Loop to Run All Analyses ---
    # Add a None to the list to represent the special pooled case
    all_analysis_configs = analyses_to_run + [None] 

    for i, analysis_config in enumerate(all_analysis_configs):
        print("-" * 80)
        print(f"Starting Static Analysis {i+1}/{len(all_analysis_configs)}...")
        
        is_pooled = analysis_config is None
        
        if is_pooled:
            extraordinary_paths = [static_embedding_dir / f"{stem}.npz" for stem in pooled_extraordinary_stems]
            ordinary_paths = [static_embedding_dir / f"{stem}.npz" for stem in pooled_ordinary_stems]
            embedding_paths = extraordinary_paths + ordinary_paths
            
            analysis_name = "Pooled_Extraordinary_vs_Ordinary"
            labels_by_class = ["All Extraordinary", "All Ordinary"]

        else: # Pairwise analysis
            embedding_paths = analysis_config
            class_stems = [path.stem.replace("_", "-") for path in embedding_paths]
            if len(embedding_paths) <= 3:
                analysis_name = "_vs_".join(class_stems)
            else:
                analysis_name = f"{len(embedding_paths)}_way_comparison"
        
        # --- Check for missing input files ---
        if any(not p.exists() for p in embedding_paths):
            print(f"WARNING: One or more input files not found for this analysis. Skipping.")
            for p in embedding_paths:
                if not p.exists():
                    print(f"  - Missing: {p}")
            continue

        # --- Generate descriptive filename and check if report already exists ---
        report_path = out_dir / f"comprehensive_analysis_{analysis_name}.txt"

        if report_path.exists():
            print(f"Report file already exists at {report_path}. Skipping analysis.")
            continue
            
        print(f"Running comparison for: {analysis_name}")
        
        # --- Load data for the current analysis ---
        all_vectors, all_labels_text, all_y_numeric, all_texts = [], [], [], []
        vectors_by_class = [[], []] if is_pooled else []
        texts_available = True

        if is_pooled:
            # Pooled loading logic
            for group_idx, group_paths in enumerate([extraordinary_paths, ordinary_paths]):
                group_vectors = []
                for path in group_paths:
                    vectors, ids = load_embeddings_and_ids(path)
                    group_vectors.append(vectors)
                    
                    clean_name = path.stem.replace("glove_", "").replace("_", " ").title()
                    if df is not None and clean_name in df.columns:
                        texts_for_class = df.loc[ids, clean_name].astype(str).tolist()
                        all_texts.extend(texts_for_class)
                    else:
                        texts_available = False
                
                concatenated_vectors = np.vstack(group_vectors)
                vectors_by_class[group_idx] = concatenated_vectors
                all_vectors.append(concatenated_vectors)
                all_labels_text.extend([labels_by_class[group_idx]] * len(concatenated_vectors))
                all_y_numeric.extend([group_idx] * len(concatenated_vectors))
        
        else: # Pairwise loading logic
            labels_by_class = []
            for j, path in enumerate(embedding_paths):
                vectors, ids = load_embeddings_and_ids(path)
                # Create a clean column name for lookup in the dataframe and for labels
                clean_name = path.stem.replace("glove_", "").replace("_", " ").title()
                
                if df is not None and clean_name in df.columns:
                    texts_for_class = df.loc[ids, clean_name].astype(str).tolist()
                    all_texts.extend(texts_for_class)
                else:
                    texts_available = False

                # Prepend 'GloVe' to the label for clarity in plots
                label = f"GloVe {clean_name}"
                vectors_by_class.append(vectors)
                labels_by_class.append(label)
                all_vectors.append(vectors)
                all_labels_text.extend([label] * len(vectors))
                all_y_numeric.extend([j] * len(vectors))
        
        if not texts_available:
            print("WARNING: Could not load texts for one or more classes. Plot will not have hover labels.")
            all_texts = None

        X = np.vstack(all_vectors)
        y = np.array(all_y_numeric)

        # --- Run Analyses and Generate Outputs ---
        umap_path = out_dir / f"umap_visualization_{analysis_name}.html"
        
        report_parts = [
            f"Comprehensive Analysis Report (GloVe)\n{'='*37}\n",
            f"Classes Compared: {', '.join(labels_by_class)}\n\n",
        ]
        report_parts.append(run_classification(X, y, labels_by_class, all_texts))
        
        if all(len(v) > 1 for v in vectors_by_class):
            report_parts.append(analyze_distances(vectors_by_class, labels_by_class, out_dir, f"_{analysis_name}"))
            report_parts.append(run_structural_analysis(X, y, vectors_by_class, labels_by_class))

        final_report = "".join(report_parts)
        report_path.write_text(final_report)
        print(f"Saved comprehensive analysis report to {report_path}")

        create_umap_plot(X, all_labels_text, all_texts, len(labels_by_class), umap_path)

    print("-" * 80)
    print("All static analyses complete.")


if __name__ == "__main__":
    main() 