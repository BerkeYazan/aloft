# 01 Source ingestion

This stage collects the Goodreads quotes that form the starting point of the
dataset, cleans them, and adds metadata such as the author and the
publication year of the book.

## What it needs

- The two public Goodreads quote datasets from Kaggle. These are the raw
  input. Place them where `data/interim/cleaning_data/` expects them.
- A Google Books API key in `.env`, for the metadata lookups.

## What it produces

- A cleaned table of Goodreads quotes with author and language information,
  written into `data/interim/`.

## Files, in run order

- `sqlite_to_csv.py`, converts the raw Kaggle SQLite database to CSV.
- `goodreads_quotes_cleaning.ipynb`, the main cleaning notebook.
- `book_name_merge.py`, merges and normalises book titles.
- `clean_authors_final.py`, final pass on author names.
- `fetch_author_languages.py`, looks up author language using the Wikimedia
  API.
- `fetch_book_publication_dates.py`, looks up publication years using the
  Google Books API and Wikidata.
- `wikipedia+googlebooks.py`, an alternative metadata enrichment script
  that queries Wikidata and the Google Books API for publication years.
- `manually_fill_missing_authors.ipynb`, a manual step to fill gaps that the
  automatic lookups could not resolve.
- `create_random_sample.py`, draws a random sample for inspection.

## Notes

The manual notebook is a human in the loop step. Its output is recorded and
treated as a fixed input to later stages. See `docs/PIPELINE.md`.
