
import logging
import pathlib
import sys
import json
import pandas as pd
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
LOG = logging.getLogger(__name__)

# --- GloVe Model Loading ---
def load_glove_model(glove_file: pathlib.Path) -> KeyedVectors:
    model_file = glove_file.with_suffix(".model")
    if model_file.exists():
        LOG.info(f"Loading pre-converted GloVe model from '{model_file}'...")
        return KeyedVectors.load(str(model_file))
    LOG.info(f"Loading GloVe model from text file: '{glove_file}'...")
    try:
        model = KeyedVectors.load_word2vec_format(str(glove_file), binary=False, no_header=True)
        model.save(str(model_file))
        LOG.info(f"GloVe model loaded and saved to '{model_file}'.")
        return model
    except Exception as e:
        LOG.error(f"Could not load GloVe model from '{glove_file}'. Details: {e}", exc_info=True)
        sys.exit(1)

# --- Metric Calculation ---
def calculate_forward_flow(text: str, model: KeyedVectors) -> float | None:
    if not isinstance(text, str): return None
    tokens = word_tokenize(text.lower())
    word_vectors = [model[word] for word in tokens if word in model]
    if len(word_vectors) < 2: return None
    stepwise_avg_distances = [
        np.mean(np.linalg.norm(word_vectors[i] - np.array(word_vectors[:i]), axis=1))
        for i in range(1, len(word_vectors))
    ]
    return np.mean(stepwise_avg_distances) if stepwise_avg_distances else None

def main():
    """
    Calculates Forward Flow scores for all necessary corpora and saves them
    to a single CSV file, which can then be merged into the master data file.
    """
    output_dir = pathlib.Path("data/outputs/analysis/forward_flow")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        df_aloft = pd.read_csv('data/outputs/master_metrics/ALOFT_master_metrics.csv')
    except FileNotFoundError:
        LOG.error("Master metrics file not found. Please ensure it exists.")
        return

    glove_model = load_glove_model(pathlib.Path("data/interim/static_embeddings/glove.840B.300d/glove.840B.300d.txt"))

    corpora_to_process = [
        "Goodreads Sample Quote", "Goodreads Popular Quote",
        "Google Books Length Matched Snippet", "T50 Quote",
        "T50 Quote-Free Context Length Matched", "Non-Literary Baseline"
    ]
    
    all_scores = []
    for col_name in corpora_to_process:
        if col_name not in df_aloft.columns:
            LOG.warning(f"Column '{col_name}' not found in master CSV. Skipping.")
            continue
        
        LOG.info(f"Processing column: '{col_name}'")
        texts_subset = df_aloft[[col_name]].dropna().reset_index() # Use original index
        
        for index, row in tqdm(texts_subset.iterrows(), total=len(texts_subset), desc=f"Calculating FF for {col_name}"):
            text = row[col_name]
            score = calculate_forward_flow(text, glove_model)
            if score is not None:
                all_scores.append({
                    'original_index': row['index'],
                    'source': col_name,
                    'score': score
                })

    if not all_scores:
        LOG.error("No Forward Flow scores were calculated. Exiting.")
        return

    final_df = pd.DataFrame(all_scores)
    metrics_path = output_dir / "forward_flow_metrics.csv"
    LOG.info(f"Saving combined metrics for {len(final_df)} documents to {metrics_path}")
    final_df.to_csv(metrics_path, index=False)
    LOG.info("Forward Flow score generation complete.")

if __name__ == "__main__":
    # Ensure NLTK data is available.
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    import numpy as np # Need to import it here for the metric function
    main() 