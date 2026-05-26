# Pipeline

This file describes the pipeline as a whole, and in particular the steps
that involve manual work and cannot be re-run automatically.

## The five conceptual stages

The repository folders are numbered 00 to 06, but the research itself has
five stages.

1. Source ingestion, folder `01_source_ingestion`. Goodreads quotes are
   collected from two public Kaggle datasets, filtered, and sampled with
   occurrence and like weighted stratified sampling. Publication years and
   language metadata are fetched from the Wikimedia API and Google Books.

2. Google Books extraction, folder `02_google_books_extraction`. For each
   quoted book, a page of non-quoted text is captured from the Google Books
   preview, read with Google Cloud Vision OCR, and corrected.

3. Data preparation, folder `03_data_preparation`. Text is length matched
   within a 70 to 130 percent token tolerance. Fuzzy matching removes
   overlap that would leak between groups. Spelling and punctuation are
   corrected. SBERT based deduplication removes near duplicates.

4. Metrics, folder `04_metrics`. Each text gets a set of metrics: Flesch
   Reading Ease, Coleman-Liau Index, Shannon entropy, lexical diversity,
   GPT-2 surprisal, pointwise mutual information, RoBERTa sentiment, GloVe
   and SBERT embeddings, stepwise distance, and forward flow.

5. Analysis, folder `05_analysis`. Mann-Whitney U tests, Cliff's Delta,
   Benjamini-Hochberg correction, XGBoost classifiers, SHAP explanations,
   and UMAP projections. Figures are produced in folder `06_figures`.

## Steps that involve manual work

These steps cannot be reproduced by running a script. Their recorded outputs
are treated as fixed inputs to the automatic steps that follow.

- Browser capture of Google Books pages. A person opens each book preview in
  a browser and captures the page region. The helper scripts
  `get_book_previews.py`, `define_screenshot_coordinates.py` and
  `view_book_previews.py` support this, but the capture itself is manual.

- OCR correction. After Google Cloud Vision reads the screenshots, the
  output contains OCR errors. `manual_quote_cleanup.py` and
  `manual_spell_checker.py` are interfaces for a person to correct the text
  by hand. The corrections are recorded and merged back by
  `merge_manual_corrections.py`.

- Author completion. `manually_fill_missing_authors.ipynb` is a manual step
  to fill author gaps that the automatic lookups could not resolve.

- Deduplication review. The SBERT based deduplication in
  `find_and_remove_embedding_duplicates.py` includes a manual review of the
  candidate duplicates.

## External services

- Google Books API, used in stage 01 for book metadata.
- Wikimedia API, used in stage 01 for author language.
- Google Cloud Vision, used in stage 02 for OCR. This requires a service
  account and may incur cost.

Because the manual steps and the paid or rate limited services cannot be
re-run freely, verification of the pipeline covers only the deterministic
computational steps. See `docs/REPRODUCIBILITY.md`.
