import pandas as pd
import numpy as np
import pathlib
import logging
import sys

# Ensure the 'Analysis' directory is in the Python path to import publication_analysis
project_root = pathlib.Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from publication_analysis import PublicationAnalysis, get_col_name

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)


def generate_leaderboard(data_path: pathlib.Path, output_dir: pathlib.Path):
    """
    Generates a model-based universal leaderboard by leveraging the training
    pipeline from the PublicationAnalysis class.

    Args:
        data_path: Path to the ALOFT_master_metrics.csv file.
        output_dir: Directory to save the leaderboard CSV file.
    """
    LOG.info("--- Starting Model-Based Leaderboard Generation ---")

    # 1. Initialize the analysis class and train the primary model.
    # As per the new strategy, we train a "Literary vs. Non-Literary" model.
    analysis = PublicationAnalysis(data_path, output_dir)
    
    # Define the classes based on the user's new specification
    literary_prefixes = ['sample', 'popular', 't50']
    non_literary_prefixes = ['t50freelength', 'snippet', 'nonlit']

    # New experiment: Exclude all sentiment-related features
    sentiment_metrics = ['sentiment_polarity', 'sentiment_pos', 'sentiment_neu', 'sentiment_neg', 'sentiment_label']
    
    model, feature_names = analysis.train_and_get_model(
        pos_prefixes=literary_prefixes,
        neg_prefixes=non_literary_prefixes,
        excluded_metrics=["pmi"] + sentiment_metrics
    )
    LOG.info(f"Model trained successfully on '{literary_prefixes} vs {non_literary_prefixes}' using {len(feature_names)} features (No Sentiment).")

    # 2. Prepare all available sources for scoring
    df = analysis.df  # Use the DataFrame loaded by the analysis class
    
    source_map = {
        'sample': 'Goodreads Sample Quote', 'popular': 'Goodreads Popular Quote',
        'snippet': 'Google Books Length Matched Snippet', 't50': 'T50 Quote',
        't50freelength': 'T50 Quote-Free Context Length Matched',
        'nonlit': 'Non-Literary Baseline'
    }

    all_sources_df_list = []
    LOG.info("Preparing all text sources for scoring...")

    for prefix, text_col_name in source_map.items():
        # Check if the source has the text column and all required feature columns
        required_metric_cols = [get_col_name(prefix, stem) for stem in feature_names]
        if not all(col in df.columns for col in [text_col_name] + required_metric_cols):
            LOG.warning(f"Skipping source '{prefix}' for scoring due to missing columns.")
            continue
        
        # Create a clean dataframe for this source
        source_df = df[[text_col_name] + required_metric_cols].copy()
        source_df.dropna(subset=[text_col_name] + required_metric_cols, inplace=True)
        
        # Rename columns to the generic feature names used in the model
        rename_dict = {get_col_name(prefix, stem): stem for stem in feature_names}
        source_df.rename(columns=rename_dict, inplace=True)
        
        # Add 'text' and 'source' columns for the final leaderboard
        source_df['text'] = source_df.pop(text_col_name)
        source_df['source'] = text_col_name
        all_sources_df_list.append(source_df)

    scoring_df = pd.concat(all_sources_df_list, ignore_index=True)
    LOG.info(f"Prepared {len(scoring_df)} total entries from {len(all_sources_df_list)} sources for scoring.")

    # 3. Generate "Distinction Score" using the trained model
    LOG.info("Generating distinction scores for all entries...")
    X_score = scoring_df[feature_names] # Ensure feature order matches training
    distinction_scores = model.predict_proba(X_score)[:, 1]
    scoring_df['distinction_score'] = distinction_scores

    # 4. Create and save the final leaderboard
    LOG.info("Ranking entries and selecting top 50...")
    leaderboard = scoring_df.sort_values(by='distinction_score', ascending=False).head(50)

    # Reorder columns and round values for presentation
    final_cols = ['distinction_score', 'text', 'source'] + feature_names
    leaderboard = leaderboard[final_cols]
    leaderboard['distinction_score'] = leaderboard['distinction_score'].round(4)
    for metric in feature_names:
        leaderboard[metric] = leaderboard[metric].round(4)
    
    # Save to CSV
    output_path = output_dir / "leaderboard_model_based_top_50_NoSentiment.csv"
    leaderboard.to_csv(output_path, index=False)
    
    LOG.info(f"Successfully generated model-based leaderboard (No Sentiment). Saved to {output_path}")
    print(f"\n--- Top 10 Entries (Model-Based Leaderboard, No Sentiment) ---\n")
    print(leaderboard.head(10).to_string(index=False))


if __name__ == "__main__":
    # Run from the repository root, so these paths resolve. See the project README.
    data_file = pathlib.Path("data/outputs/master_metrics/ALOFT_master_metrics.csv")
    output_dir = pathlib.Path("data/outputs/analysis")

    if not data_file.exists():
        LOG.error(f"FATAL: Could not locate ALOFT_master_metrics.csv.")
        sys.exit(1)
        
    generate_leaderboard(data_file, output_dir) 