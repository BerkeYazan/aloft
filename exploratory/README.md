# Exploratory

This folder contains scripts that are outside the main pipeline. They are
provided for reference and as a starting point for further work, and are
not run by the numbered stages.

## Contents

- `perform_clustering.py`, clustering of SBERT and GloVe embeddings.
- `perform_topic_modeling.py`, BERTopic topic modelling.
- `analyze_topic_coherence.py`, topic coherence analysis.
- `classification_analysis.py`, a LightGBM classifier for the quote
  versus context task.
- `shap_analysis.py`, a LightGBM regressor with SHAP.
- `onesentencesurprisal.py`, a minimal example computing GPT-2 surprisal
  for a single sentence.
