# 03 Data preparation

This stage turns the cleaned quotes and passages into the final ALOFT
dataset. It matches text lengths, removes overlap that would leak answers
between groups, and removes near duplicate entries.

## What it needs

- The cleaned Goodreads quotes from stage 01.
- The Google Books passages from stage 02.

## What it produces

- `data/processed/public/ALOFT.csv` and the related ALOFT tables, the final
  dataset.

## Files, in run order

- `sentence_aware_sampling.py`, samples text in a way that respects sentence
  boundaries, used for length matching.
- `find_fuzzy_matches.py`, finds near identical text across groups.
- `match_quotes_by_title.py`, matches quotes to books by title.
- `remove_matched_quotes.py`, removes the overlap found above so that no
  group leaks into another.
- `find_and_remove_embedding_duplicates.py`, removes near duplicate entries
  using SBERT embeddings, with a manual review step.
- `compare_line_counts.py`, a check that compares row counts before and
  after cleaning.
- `ALOFT_creation.ipynb`, assembles the final ALOFT dataset from all of the
  prepared parts.

## Notes

The length matching keeps passages within a 70 to 130 percent token
tolerance of their paired quote. This tolerance and the matching rules must
not be changed if you want to reproduce the paper. See
`docs/REPRODUCIBILITY.md`.
