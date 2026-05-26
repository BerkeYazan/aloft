# Reproducibility

The pipeline is designed to reproduce the numbers reported in the paper.
This file explains the environment, the things that must not change, and
how to check a run.

## Environment

- Python 3.11.
- Packages are listed in `requirements.txt` and should be installed exactly
  as listed. Different package versions can change numeric results.

## Things that must not change

- Model identifiers. The GPT-2, SBERT, RoBERTa and GloVe model names are
  fixed in `config.py`. Different model versions give different numbers.
- Numeric constants and thresholds, for example the length matching
  tolerance, the sampling weights, and the XGBoost settings.
- Random seeds.
- The order of operations where order affects results, for example sampling
  before or after filtering.
- Data filtering and selection criteria.

## Deterministic and manual stages

Manual stages, the browser capture, the OCR correction interfaces, the
author completion notebook, and the deduplication review, cannot be re-run
automatically. Their recorded outputs are fixed inputs.

The deterministic stages are the computational transforms in stages 03, 04
and 05. These are the stages a verification run should cover.

## Verification procedure

To verify a run:

1. Run a deterministic stage on its original inputs.
2. Compare the outputs against the reference outputs. For files, compare
   SHA-256 hashes. For numeric tables, compare values within normal
   floating point tolerance.
3. A match means behaviour is preserved.

Verify one stage at a time so that any difference is easy to locate.
