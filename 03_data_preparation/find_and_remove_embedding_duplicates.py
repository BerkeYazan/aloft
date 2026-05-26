#!/usr/bin/env python
"""
Finalizes the ALOFT dataset by finding and removing near-duplicates.

This script is the final step in the data creation pipeline. It uses pre-generated
SBERT embeddings to find and remove near-duplicate rows from the output of the
`ALOFT_creation.ipynb` notebook.

It produces two key outputs:
1.  `duplication_removal_log.txt`: A detailed log for reproducibility.
2.  `ALOFT.csv` and `ALOFT_with_metadata.csv`: The final, clean datasets.

NOTE: This script ASSUMES that up-to-date embeddings have already been
generated for the '..._before_semantic_deduplication.csv' files. If not,
please run `generate_dynamic_embeddings.py` first.

To run, simply execute from the terminal:
    python 03_data_preparation/find_and_remove_embedding_duplicates.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


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


def find_duplicates_in_file(
    vectors: np.ndarray, original_ids: np.ndarray, threshold: float
) -> list[tuple]:
    """
    Finds pairs of vectors with cosine similarity above the threshold.
    Uses NearestNeighbors for efficient search.
    """
    # The distance is 1 - similarity. So we look for distances < (1 - threshold).
    distance_threshold = 1 - threshold

    # We ask for 2 neighbors because the closest neighbor to any point is itself.
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute")
    nn.fit(vectors)
    distances, indices = nn.kneighbors(vectors)

    found_pairs = set()
    duplicates = []

    for i in range(len(vectors)):
        # The nearest neighbor is at index 1 (index 0 is the point itself)
        neighbor_idx = indices[i, 1]
        neighbor_dist = distances[i, 1]

        if neighbor_dist < distance_threshold:
            # Sort the indices to ensure (A, B) and (B, A) are treated as the same pair
            original_id1 = original_ids[i]
            original_id2 = original_ids[neighbor_idx]
            
            pair = tuple(sorted((original_id1, original_id2)))
            
            if pair not in found_pairs:
                similarity = 1 - neighbor_dist
                duplicates.append((pair[0], pair[1], similarity))
                found_pairs.add(pair)
                
    return duplicates


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
    
    # Source file created by the notebook
    source_metadata_path = pathlib.Path("data/processed/public/ALOFT_with_metadata_before_semantic_deduplication.csv")

    # Final output files
    final_csv_path = pathlib.Path("data/processed/public/ALOFT.csv")
    final_metadata_path = pathlib.Path("data/processed/public/ALOFT_with_metadata.csv")

    embedding_dir = pathlib.Path("data/interim/dynamic_embeddings")
    out_dir = pathlib.Path("data/outputs/analysis")
    
    # Columns to check for duplicates
    columns_to_check = [
        "Goodreads Sample Quote",
        "Google Books Length Matched Snippet",
        "Non-Literary Baseline",
        "Goodreads Popular Quote",
        "T50 Quote",
        "T50 Quote-Free Context",
        "T50 Quote-Free Context Length Matched",
    ]
    
    threshold = 0.95
    log_file = out_dir / "duplication_removal_log.txt"

    # --- End of Hardcoded Parameters ---

    # 1. Load the source dataframe with metadata
    if not source_metadata_path.exists():
        print(f"ERROR: Source metadata file not found at {source_metadata_path}", file=sys.stderr)
        print("Please run the ALOFT_creation.ipynb notebook first.", file=sys.stderr)
        sys.exit(1)
    print(f"Loading source data from {source_metadata_path}...")
    df = pd.read_csv(source_metadata_path)
    
    # 2. Find duplicates across all specified columns
    indices_to_remove = set()
    log_parts = [
        f"Duplication Removal Log\n",
        f"Similarity Threshold: > {threshold}\n",
        f"Input File: {source_metadata_path}\n",
        f"Output Files: {final_csv_path}, {final_metadata_path}\n",
        f"{'='*40}\n\n",
    ]

    print("Starting duplicate analysis...")
    for column_name in columns_to_check:
        embedding_path = embedding_dir / f"{column_name.lower().replace(' ', '_')}.npz"
        
        if not embedding_path.exists():
            print(f"\nERROR: Embedding file not found for column '{column_name}' at: {embedding_path}", file=sys.stderr)
            print("Please run `generate_dynamic_embeddings.py` on the `..._before_semantic_deduplication.csv` file first.", file=sys.stderr)
            sys.exit(1)
            
        print(f"Processing file for duplicates: {embedding_path.name}...")
        vectors, ids = load_embeddings_and_ids(embedding_path)
        
        if column_name not in df.columns:
            print(f"WARNING: Column '{column_name}' not found in CSV. Cannot process {embedding_path.name}", file=sys.stderr)
            continue
            
        duplicate_pairs = find_duplicates_in_file(vectors, ids, threshold)
        
        if duplicate_pairs:
            log_parts.append(f"--- Found {len(duplicate_pairs)} duplicate pairs in: {column_name} ---\n")
            for id1, id2, sim in sorted(duplicate_pairs, key=lambda x: x[0]):
                # Keep the first instance (lower index), remove the second
                index_to_keep, index_to_drop = min(id1, id2), max(id1, id2)
                indices_to_remove.add(index_to_drop)
                
                text_kept = df.loc[index_to_keep, column_name]
                text_dropped = df.loc[index_to_drop, column_name]
            
                log_parts.append(f"Action: Remove index {index_to_drop} (Duplicate of {index_to_keep}, Similarity: {sim:.4f})\n")
                log_parts.append(f"  - Kept ({index_to_keep}): {text_kept}\n")
                log_parts.append(f"  - Removed ({index_to_drop}): {text_dropped}\n\n")

    # 3. Remove duplicates and save final CSV files
    if not indices_to_remove:
        print("\nNo duplicates found. Copying source files to final destination.")
        df.to_csv(final_metadata_path, index=False)
        # Create the text-only version
        text_only_df = df[[col for col in columns_to_check if col in df.columns]]
        text_only_df.to_csv(final_csv_path, index=False)
    else:
        print(f"\nFound a total of {len(indices_to_remove)} unique rows to remove.")
        cleaned_df = df.drop(index=list(indices_to_remove)).reset_index(drop=True)
        
        # Save the version with metadata
        cleaned_df.to_csv(final_metadata_path, index=False)
        print(f"Successfully saved final deduplicated data with metadata to: {final_metadata_path}")
        
        # Save the text-only version
        text_only_df = cleaned_df[[col for col in columns_to_check if col in cleaned_df.columns]]
        text_only_df.to_csv(final_csv_path, index=False)
        print(f"Successfully saved final deduplicated text-only data to: {final_csv_path}")
        
        log_parts.append(f"\nSummary: Removed a total of {len(indices_to_remove)} rows from the original {len(df)} rows.\n")
        log_parts.append(f"New dataset has {len(cleaned_df)} rows.\n")

    # 4. NEW: Create new, cleaned embedding files that match the deduplicated CSV
    print("\n--- Creating Final, Cleaned Embedding Files ---")
    
    # Get the list of indices that were *kept* in the final dataframe
    final_indices_to_keep = set(cleaned_df.index if 'cleaned_df' in locals() else df.index)
    
    for column_name in columns_to_check:
        embedding_path = embedding_dir / f"{column_name.lower().replace(' ', '_')}.npz"
        
        if not embedding_path.exists():
            print(f"WARNING: Original embedding file not found at {embedding_path}. Cannot create cleaned version.", file=sys.stderr)
            continue

        print(f"Cleaning embedding file: {embedding_path.name}...")
        
        # Load the original, full-size embeddings
        original_vectors, original_ids = load_embeddings_and_ids(embedding_path)
        
        # Create a boolean mask to filter the embeddings
        # We keep an embedding if its original ID is in our final set of kept indices
        mask = np.isin(original_ids, list(final_indices_to_keep))
        
        # Apply the mask to get the clean vectors and IDs
        cleaned_vectors = original_vectors[mask]
        cleaned_ids = original_ids[mask]
        
        # Overwrite the original .npz file with the new, smaller, clean version
        _save_vectors(embedding_path, cleaned_ids, cleaned_vectors)
        
        print(f"  - Original size: {len(original_vectors)}. New size: {len(cleaned_vectors)}.")

    # --- Write Final Log File ---
    out_dir.mkdir(parents=True, exist_ok=True)
    final_log = "".join(log_parts)
    log_file.write_text(final_log)
    print(f"Duplication removal log saved to: {log_file}")


if __name__ == "__main__":
    main() 