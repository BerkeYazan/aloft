#!/usr/bin/env python
"""
Generates SBERT embeddings for dictionary definitions of nouns.

This is the first part of the semantic shift analysis pipeline. It performs
the following data preparation steps:
1.  Loads the clean `ALOFT.csv` dataset.
2.  Extracts all unique nouns from the "Goodreads Sample Quote" column using spaCy.
3.  Fetches the primary WordNet dictionary definition for each unique noun.
4.  Embeds these unique definitions using a pre-trained SBERT model.
5.  Saves the definitions and their corresponding vectors to a single .npz file
    for use in the visualization script.

To run, simply execute from the terminal:
    python 04_metrics/generate_definition_embeddings.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import spacy
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Download NLTK wordnet if not present ---
try:
    import nltk
    wordnet.ensure_loaded()
except LookupError:
    print("Downloading NLTK 'wordnet' model...")
    nltk.download("wordnet", quiet=True)


def main():
    """Main execution pipeline."""
    # --- Configuration ---
    csv_path = pathlib.Path("data/processed/public/ALOFT.csv")
    columns_to_process = ["Goodreads Sample Quote", "Non-Literary Baseline"]
    output_path = pathlib.Path("data/interim/embeddings/noun_definitions_embeddings.npz")
    model_name = "sentence-transformers/all-mpnet-base-v2"

    # --- 1. Load Data and Models ---
    if not csv_path.exists():
        print(f"ERROR: ALOFT.csv not found at {csv_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("Loading SBERT and spaCy models...")
    sbert_model = SentenceTransformer(model_name)
    nlp = spacy.load("en_core_web_sm")

    # --- 2. Extract All Unique Nouns and Definitions ---
    print("\nExtracting all unique nouns and definitions from the corpus...")
    unique_nouns = set()
    for column in columns_to_process:
        for text in tqdm(df[column].dropna(), desc=f"Scanning {column}"):
            doc = nlp(text)
            for token in doc:
                if token.pos_ == "NOUN":
                    unique_nouns.add(token.lemma_.lower())

    noun_to_definition = {}
    for noun in tqdm(sorted(list(unique_nouns)), desc="Fetching definitions"):
        synsets = wordnet.synsets(noun, pos=wordnet.NOUN)
        if synsets:
            noun_to_definition[noun] = synsets[0].definition()
            
    if not noun_to_definition:
        print("ERROR: Could not find any nouns with definitions. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found {len(noun_to_definition)} unique nouns with definitions from both columns.")

    # --- 3. Embed Unique Definitions ---
    # We embed definitions, not nouns, to avoid duplicate work
    unique_definitions = sorted(list(set(noun_to_definition.values())))
    print(f"\nEmbedding {len(unique_definitions)} unique definitions...")
    
    definition_vectors = sbert_model.encode(
        unique_definitions,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # --- 4. Save Definitions and Embeddings ---
    # We save the definitions as a NumPy array of strings for easy loading
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        definitions=np.array(unique_definitions),
        vectors=definition_vectors,
    )
    
    print(f"\nSuccessfully saved {len(unique_definitions)} definitions and their embeddings to:")
    print(output_path)


if __name__ == "__main__":
    main() 