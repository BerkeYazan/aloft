# 02 Google Books extraction

This stage builds the GBLM group, the length matched non-quoted passages. For
each quoted book it captures a page of text from the Google Books preview,
reads the text with OCR, and corrects the result.

This is the most manual stage of the project. Several steps cannot be run
automatically. Read `docs/PIPELINE.md` before starting.

## What it needs

- The cleaned Goodreads quotes from stage 01.
- Google Cloud Vision credentials in `.env`, for OCR.
- A person to capture Google Books pages in a browser and to review
  corrections.

## What it produces

- A table of non-quoted passages aligned to the quoted books, written into
  `data/interim/google_books_work/`.

## Files, by step

- `get_random_google_books.ipynb`, selects which books to capture.
- `get_book_previews.py`, helps capture Google Books preview pages.
- `define_screenshot_coordinates.py`, sets the screen regions to capture.
- `view_book_previews.py`, a viewer for the captured pages.
- `process_screenshots.py`, runs Google Cloud Vision OCR on the screenshots.
- `merge_extracted_text.py`, merges the OCR output into one table.
- `manual_quote_cleanup.py`, an interface for hand correcting the text. See
  `manual_quote_cleanup_README.md` in this folder.
- `manual_spell_checker.py`, an interface for spelling correction.
- `merge_manual_corrections.py`, merges the hand corrections back in.
- `remove_incorrect_entries.py`, drops entries that could not be corrected.
- `clean_merged_quotes.py`, final cleaning of the merged text.

## Notes

Captured page images are intermediate artefacts of the extraction stage and
live in `data/restricted/`. See `docs/DATA.md`.
