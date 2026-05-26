# 05 Analysis

This stage runs the statistical analysis on the metrics from stage 04. It
compares the groups, trains classifiers, explains them, and produces the
tables used in the paper.

## What it needs

- The metric tables from stage 04, in particular
  `data/outputs/master_metrics/ALOFT_master_metrics.csv`.

## What it produces

- Statistical results, classifier outputs, tables and reports, written into
  `data/outputs/`.

## Files, by purpose

Statistics:

- `correlation_analysis.py`, correlations between metrics.
- `publication_analysis.py`, the main inferential statistics and the
  XGBoost classifiers with SHAP explanations. Mann-Whitney U tests, Cliff's
  Delta and Benjamini-Hochberg correction. This is the script behind
  Tables 2, 3 and 4 and Figure 4 in the paper.
- `generate_publication_stats.py`, `generate_ff_stats.py`,
  `generate_swd_stats.py`, formatted statistics for the paper.
- `evaluate_normalization.py`, checks on metric normalisation.

Structure of the data:

- `analyze_static_embeddings.py`, `analyze_dynamic_embeddings.py`,
  `compare_static_vs_dynamic_embeddings.py`, embedding analysis.

Tables and leaderboards:

- `create_publication_tables.py`, `generate_leaderboard.py`,
  `generate_latex_leaderboards.py`, `generate_latex_table.py`.

Checks:

- `health_check.py`, a sanity check on the dataset.

## Notes

The classifier settings, the random seeds and the test correction method
must stay as they are to reproduce the paper. See `docs/REPRODUCIBILITY.md`.
