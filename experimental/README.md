# Experimental variants

Nothing in this folder was used to produce the numbers reported in the MOSAIC paper.
The scripts are kept for completeness and as a starting point for follow-up work.

| Script | What it does |
|---|---|
| `patch_preprocessing.py` | Runs the full pipeline with light image pre-processing (robust percentile normalisation, colour normalisation to a fixed target, autocontrast for very white / low-detail patches, mild contrast gain) applied to every patch before the network. Same CLI as `scripts/train_mosaic.py`. |
| `run_ablation_preprocessing.py` | The K / aggregation ablation of `scripts/run_ablation.py` with the pre-processing variant. Same CLI. |
