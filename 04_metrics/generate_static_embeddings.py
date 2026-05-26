#!/usr/bin/env python
"""
Generates static (non-contextual) sentence embeddings using a local GloVe model file.

This script creates a sentence vector by averaging the pre-trained GloVe word vectors
for all words in a sentence. It serves as a high-quality baseline for comparison
with contextual models like SBERT.

To use this script:
1. Download the GloVe model (e.g., 'glove.840B.300d.txt') from the official Stanford site.
2. Place it in the designated path specified in the `main` function.
3. Run the script.

Example usage
-------------
$ python 04_metrics/generate_static_embeddings.py

This will produce .npz files in 'data/interim/static_embeddings' which can be
consumed by 'analyze_static_embeddings.py'.

Requirements
------------
    pip install gensim nltk pandas
"""

from __future__ import annotations

import pathlib
import sys
from typing import List

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize

# --- Download NLTK tokenizer models if not present ---
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading NLTK models for tokenization...")
    # The `import nltk` is repeated here to handle the edge case where the
    # `try` block fails before the import is assigned, even though a
    # `LookupError` implies it was successfully imported.
    import nltk
    nltk.download("punkt", quiet=True)


def load_glove_model(glove_file: pathlib.Path) -> KeyedVectors:
    """
    Loads a GloVe model from a text file.

    For efficiency, it converts the GloVe text file to a binary '.model'
    format on the first run. Subsequent runs load the binary model directly,
    which is significantly faster.
    """
    model_file = glove_file.with_suffix(".model")
    if model_file.exists():
        print(f"Loading pre-converted GloVe model from '{model_file}'...")
        return KeyedVectors.load(str(model_file))

    print(f"Loading GloVe model from text file: '{glove_file}'...")
    print(
        "This will take several minutes, but a faster-loading binary "
        "version will be saved for all future runs."
    )

    try:
        # The glove2word2vec conversion script is deprecated and can be buggy.
        # We now load the GloVe text file directly, which is more reliable.
        model = KeyedVectors.load_word2vec_format(
            str(glove_file), binary=False, no_header=True
        )
        model.save(str(model_file))
        print(
            f"GloVe model loaded and saved to '{model_file}' for faster loading next time."
        )
        return model
    except FileNotFoundError:
        print(f"ERROR: GloVe source file not found at '{glove_file}'.", file=sys.stderr)
        print("Please ensure you have downloaded the GloVe model and placed it in the correct directory.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not load GloVe model from '{glove_file}'.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        sys.exit(1)


def sentence_to_vector(
    sentence: str, model: KeyedVectors, tokenizer
) -> np.ndarray:
    """
    Converts a sentence to a single vector by averaging its word vectors.
    Words not in the model's vocabulary are ignored.
    """
    tokens = tokenizer(sentence.lower())
    word_vectors = [model[word] for word in tokens if word in model]

    if not word_vectors:
        # Return a zero vector if no words are in the vocabulary
        return np.zeros(model.vector_size)

    # Average the vectors of the words found
    return np.mean(word_vectors, axis=0)


def _save_vectors(
    out_path: pathlib.Path,
    ids: np.ndarray,
    vectors: np.ndarray,
) -> None:
    """Save indices and vectors to a compressed .npz archive."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ids=ids, vectors=vectors)
    print(f"Saved {vectors.shape[0]} vectors -> {out_path}")


def main() -> None:
    # --- Hardcoded Parameters for Reproducibility ---
    csv_path = pathlib.Path("data/processed/public/ALOFT.csv")
    out_dir = pathlib.Path("data/interim/static_embeddings")
    glove_model_path = out_dir / "glove.840B.300d" / "glove.840B.300d.txt"

    columns_to_embed = [
        "Goodreads Sample Quote",
        "Google Books Length Matched Snippet",
        "Non-Literary Baseline",
        "Goodreads Popular Quote",
        "T50 Quote",
        "T50 Quote-Free Context",
        "T50 Quote-Free Context Length Matched",
    ]
    # --- End of Hardcoded Parameters ---

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    print(f"Loading dataset {csv_path} ...", file=sys.stderr)
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Source CSV file not found at {csv_path}", file=sys.stderr)
        print("Please ensure the ALOFT.csv file is present in the Analysis directory.", file=sys.stderr)
        sys.exit(1)

    # 2. Load GloVe model
    glove_model = load_glove_model(glove_model_path)

    # 3. Encode each column
    for col in columns_to_embed:
        safe_name = f"glove_{col.lower().replace(' ', '_').replace('/', '-')}"
        out_path = out_dir / f"{safe_name}.npz"

        if out_path.exists():
            print(f"Skipping '{col}', {out_path} already exists.", file=sys.stderr)
            continue

        series = df[col].dropna()
        ids = series.index.to_numpy()
        texts = series.astype(str).tolist()

        print(f"-> Encoding column '{col}' ({len(texts)} texts)...", file=sys.stderr)

        vectors = np.array(
            [sentence_to_vector(text, glove_model, word_tokenize) for text in texts]
        )

        # 4. Save
        _save_vectors(out_path, ids, vectors)


if __name__ == "__main__":
    main() 