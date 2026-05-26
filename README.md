# ALOFT

ALOFT is a dataset and analysis pipeline for studying what makes literary
quotes stand out. It accompanies the paper "What Makes Literary Quotes Stand
Out? A Multidimensional Analysis of Literary Creativity" (ICCC 2026).

This repository contains the full pipeline that builds the dataset and runs
the analysis. It is organised so that someone who has never seen the project
can follow it from start to finish.

## What ALOFT is

ALOFT is a dataset of 4,569 parallel entries. Each row lines up a literary
quote with several comparison texts. The groups of text are:

- GSQ, Goodreads sample quotes.
- GPQ, Goodreads popular quotes.
- GBLM, length matched non-quoted passages from the same books, captured from
  Google Books.
- T50Q, public domain quotes from the earlier T50 study.
- T50LM, length matched context for the T50 quotes.
- NLB, a non-literary baseline drawn from Wikipedia and the Brown corpus.

For the full description of the groups and how the data may be reused, see
`docs/DATA.md`.

## How the repository is laid out

The work runs in seven stages. The folders are numbered so the order is
obvious. Read them in sequence.

```
00_setup/                 install the environment and credentials
01_source_ingestion/      collect and clean the Goodreads quotes
02_google_books_extraction/ capture and OCR the Google Books passages
03_data_preparation/      match lengths, remove leakage, assemble ALOFT
04_metrics/               compute the creativity metrics for every text
05_analysis/              statistics, classifiers, and SHAP explanations
06_figures/               produce the figures used in the paper
notebooks/                exploratory notebooks, not part of the pipeline
config.py                 shared settings, model names and file paths
data/                     all data, see docs/DATA.md
docs/                     detailed documentation
```

Each numbered folder has its own README that explains what the stage does,
what it needs as input, and what it produces.

## Setting up

See `00_setup/README.md` for the full instructions. In short:

1. Create a Python virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env` and fill in your own credentials.

## How to run the code

Every script is meant to be run from the repository root, so that the file
paths inside the scripts resolve correctly, and so that `import config`
finds the shared settings file. Run scripts like this:

```
cd aloft
PYTHONPATH=. python 04_metrics/calculate_creativity_metrics.py
```

Some stages cannot be run automatically because they involve manual work,
for example capturing Google Books pages in a browser and correcting OCR
output by hand. Those stages are documented in `docs/PIPELINE.md`.

## Data and pipeline

This repository releases the ALOFT pipeline, the analysis code, and the
dataset's released partitions (Goodreads quotes, T50, non-literary
baseline) plus the derived metrics for every entry. The GBLM column
(Google Books length-matched passages) is in-copyright text from Google
Books pages and is not redistributed; the stage 02 pipeline regenerates
it from your own Google Books access. With the repository and Google Books
access, the full ALOFT corpus is reproducible end to end. Per-source
details are in `docs/DATA.md`.

## Reproducing the paper

`docs/REPRODUCIBILITY.md` explains the environment, the random seeds, which
stages are deterministic, and how to check that a run matches the paper.

## Citation and licence

A citation template is in `CITATION.cff`. The code is offered under the
licence in `LICENSE`. Data licensing is described in `docs/DATA.md`.
