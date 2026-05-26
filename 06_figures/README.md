# 06 Figures

This stage produces the figures and visualisations used in the paper. It
reads the metrics and analysis outputs from the earlier stages and does not
change any data.

## What it needs

- The metric and analysis outputs from stages 04 and 05.

## What it produces

- Figures and interactive plots, written into `data/outputs/`.

## Files, by purpose

Embedding plots:

- `visualize_3d_embeddings.py`, three dimensional embedding plots.
- `visualize_semantic_distance.py`, semantic distance plots.

Journey plots, the metric trajectories through a text:

- `visualize_semantic_shift.py`
- `visualize_aha_journey.py`
- `visualize_journey_archetypes.py`
- `visualize_single_journey.py` and `visualize_aggregate_journeys.py`
- `visualize_single_ff_journey.py` and `visualize_aggregate_ff_journeys.py`

Other:

- `visualize.PMI.py`, pointwise mutual information plots.
- `restyle_plots.py`, applies a consistent visual style to saved plots.

## Notes

This stage is presentation only. Running it does not affect any number
reported in the paper.
