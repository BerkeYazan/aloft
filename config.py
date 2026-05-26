# -*- coding: utf-8 -*-
"""
Central configuration file for the ALOFT creativity analysis pipeline.

This file centralizes shared parameters, such as file paths and lists of
text sources, to ensure consistency across all analysis scripts and prevent
code duplication. By following the DRY (Don't Repeat Yourself) principle,
it makes the entire workflow more robust and easier to maintain.
"""
import pathlib

# --- Core File Paths ---
# Path to the primary dataset CSV. This is the main input for all analysis.
DATA_CSV_PATH = pathlib.Path("data/processed/public/ALOFT.csv")

# --- Directory Configuration ---
# Base directory where all analysis outputs will be saved.
BASE_OUTPUT_DIR = pathlib.Path("data/outputs/analysis")

# --- Source Column Configuration ---
# A dictionary mapping the column names in the CSV to the short prefixes
# used for file naming. This is the single source of truth for which
# text sources are included in the analysis.
SOURCES_TO_ANALYZE = {
    "Goodreads Sample Quote": "sample",
    "Goodreads Popular Quote": "popular",
    "Google Books Length Matched Snippet": "snippet",
    "T50 Quote": "t50",
    "T50 Quote-Free Context Length Matched": "t50_free",
    "Non-Literary Baseline": "nonlit"
}

# --- Model Configuration ---
# Centralizing model names for easier swapping and reference.
GPT2_MODEL = 'gpt2'
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest" 