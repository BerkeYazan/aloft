#!/usr/bin/env python
"""
Generates dynamic (SBERT) embeddings for the ALOFT dataset.

This script reads the `..._before_semantic_deduplication.csv`
file and generate SBERT embeddings for all relevant text columns, saving them
to the `data/interim/dynamic_embeddings/` directory.

It will skip any column for which an embedding file already exists.

To run, simply execute from the terminal:
    python 04_metrics/generate_dynamic_embeddings.py
"""

from __future__ import annotations

# Prevent silent CPU fallback on Apple-Silicon and tweak memory watermark
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import argparse
import pathlib
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def _detect_device() -> str:
    """Return 'mps' on Apple Silicon if available, else 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(model_name: str, device: str) -> SentenceTransformer:
    """Load a SentenceTransformer model on the given device."""
    print(f"Loading SBERT model '{model_name}' on {device} ...", file=sys.stderr)
    return SentenceTransformer(model_name, device=device)


def _embed(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 128,
) -> np.ndarray:
    """Vectorise a list of texts into a 2-D NumPy array."""
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def _save_vectors(
    out_path: pathlib.Path,
    ids: np.ndarray,
    vectors: np.ndarray,
) -> None:
    """Save indices and vectors to a compressed .npz archive."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ids=ids, vectors=vectors)
    print(f"Saved {vectors.shape[0]} vectors -> {out_path}")


def main():
    # --- Hardcoded Parameters for Reproducibility ---
    
    csv_path = pathlib.Path("data/processed/public/ALOFT_before_semantic_deduplication.csv")
    out_dir = pathlib.Path("data/interim/dynamic_embeddings")
    
    columns_to_embed = [
        "Goodreads Sample Quote",
        "Google Books Length Matched Snippet",
        "Non-Literary Baseline",
        "Goodreads Popular Quote",
        "T50 Quote",
        "T50 Quote-Free Context Length Matched",
    ]

    model_name = "sentence-transformers/all-mpnet-base-v2"
    batch_size = 64 # Using a safe default batch size

    # --- End of Hardcoded Parameters ---

    out_dir.mkdir(parents=True, exist_ok=True)
    device = _detect_device()

    # 1. Read dataset
    if not csv_path.exists():
        print(f"ERROR: Source CSV file not found at {csv_path}", file=sys.stderr)
        print("Please run the ALOFT_creation.ipynb notebook first.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading dataset {csv_path} ...", file=sys.stderr)
    df = pd.read_csv(csv_path)

    # Validate columns exist
    missing = [c for c in columns_to_embed if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in CSV: {', '.join(missing)}")

    # 2. Model ---------------------------------------------------------------
    model = _load_model(model_name, device)

    # 3. Encode each column independently -----------------------------------
    for col in columns_to_embed:
        safe_name = col.lower().replace(" ", "_").replace("/", "-")
        out_path = out_dir / f"{safe_name}.npz"

        # Skip work that is already done --------------------------------------------------
        if out_path.exists():
            print(f"Skipping '{col}', {out_path} already exists.", file=sys.stderr)
            continue

        series = df[col].dropna()
        ids = series.index.to_numpy()
        texts = series.astype(str).tolist()

        print(f"-> Encoding column '{col}' ({len(texts)} texts)...", file=sys.stderr)
        vectors = _embed(model, texts, batch_size=batch_size)

        # 4. Save -------------------------------------------------------------
        _save_vectors(out_path, ids, vectors)


if __name__ == "__main__":
    main()
