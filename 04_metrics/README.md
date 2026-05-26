# 04 Metrics

This stage computes the creativity and readability metrics for every text in
the dataset. These metrics are the features used by the analysis in stage 05.

## What it needs

- `data/processed/public/ALOFT.csv` from stage 03.
- The GloVe word vectors, placed in `data/models/` and
  `data/interim/static_embeddings/`. GloVe is large and is not stored in the
  repository.
- Model downloads. GPT-2, SBERT and RoBERTa are downloaded on first use. The
  exact model names are fixed in `config.py` and must not be changed.

## What it produces

- Per text metric tables and embedding files, written into `data/interim/`
  and `data/outputs/`.

## Files, by purpose

Readability and surface metrics:

- `calculate_creativity_metrics.py`, the main metric calculator.
- `ALOFT_traditional_metrics.ipynb`, the traditional readability metrics.

Embeddings:

- `generate_static_embeddings.py`, GloVe based static embeddings.
- `generate_dynamic_embeddings.py`, SBERT based contextual embeddings.
- `generate_definition_embeddings.py`, embeddings of word definitions.

Novelty and movement metrics:

- `surprisal_gpt2.ipynb`, GPT-2 surprisal.
- `analyze_surprisal_aha.py`, the surprisal based journey metric.
- `analyze_semantic_shift.py`, semantic shift across a text.
- `analyze_sentiment_journey.py`, the affective journey.
- `analyze_stepwise_distance.py`, stepwise distance.
- `analyze_forward_flow.py` and `calculate_ff_scores.py`, forward flow.
- `delta_pmi_pipeline.py`, pointwise mutual information.

## Notes

The model names are the single most important thing to keep fixed. Different
versions of GPT-2, SBERT or RoBERTa produce different numbers. See
`docs/REPRODUCIBILITY.md`.
