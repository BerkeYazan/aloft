# Data

This file describes the data groups in ALOFT, where the data lives in the
repository, and the licensing for each source.

## The data groups

ALOFT has 4,569 rows. Each row aligns a literary quote with several
comparison texts. The groups are:

- GSQ, Goodreads sample quotes.
- GPQ, Goodreads popular quotes.
- GBLM, Google Books length matched passages, non-quoted text from the same
  books as the quotes.
- T50Q, public domain quotes from the earlier T50 study.
- T50LM, length matched context for the T50 quotes.
- NLB, a non-literary baseline from Wikipedia and the Brown corpus.

There are also several full context columns that hold the surrounding text.

## Where the data lives

```
data/
  raw/                 original input files used by the pipeline
  interim/             working data produced during the pipeline
  restricted/          source-locked working data such as page screenshots
  processed/
    public/            the released dataset and derived metrics
  outputs/             analysis outputs and reports
  models/              large model files such as GloVe
```

The released dataset and derived metrics sit in `data/processed/public/`.

## Release model

The repository releases the ALOFT pipeline, the analysis code, and the
dataset's released partitions plus the derived metrics. The released CSVs
(`ALOFT_release.csv` and `ALOFT_with_metadata_release.csv` in
`data/processed/public/`) contain every partition except the GBLM (Google
Books length-matched) column, which is in-copyright text from Google Books
pages and cannot be redistributed. The stage 02 pipeline reproduces the
GBLM column from source with Google Books access. With the repository, the
pipeline, and Google Books access, the full ALOFT corpus is reproducible
end to end.

## Per source licensing

- Goodreads quotes: from two public Kaggle datasets, cite the Kaggle source.
- Google Books passages: in-copyright text, not included in the released
  CSV files. The stage 02 pipeline captures Google Books page previews and
  applies Google Cloud Vision OCR to regenerate this partition locally.
- T50 quotes: public domain via Project Gutenberg.
- Wikipedia text: CC BY-SA, requires attribution.
- Brown corpus: distributed with NLTK under its own terms.

The recommended licence for the derived data that can be released is
CC BY 4.0. This is a data licence and is documented here rather than as a
repository LICENSE file. Confirm this choice before publishing.
