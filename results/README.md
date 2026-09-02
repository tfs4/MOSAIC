# `results/`

Default output location of every script. Contents are not versioned.

A run folder written by `scripts/train_mosaic.py` contains:

| File | Description |
|---|---|
| `results.csv` | One-row summary: split sizes, seed, K, aggregation, val/test mean per-gene PCC, val macro-F1, test accuracy / macro-F1 / weighted-F1 |
| `metrics_<TEST>.csv` | Per-class precision / recall / F1 / support (sklearn classification report) |
| `confusion_<TEST>.csv`, `confusion_<TEST>.png` | Confusion matrix (counts and row-normalised heat-map) |
| `expr_model_<TRAIN>.pt` | Phase A checkpoint (ConvNeXt-Tiny + attention aggregation + expression head) |
| `classifier_<TRAIN>.pt`, `classifier_<TRAIN>_norm.npz` | Phase B MLP and the training mean/std used to z-score its input |
| `expr_pred_full_<SLIDE>.npy` | Predicted 460-gene expression for every cell of each slide (skipped with `--no_save_full_predictions`) |
| `val_indices_<VAL>_pct20.npy`, `test_indices_<TEST>_pct10.npy` | Exact cells used for validation and test (re-usable through `--indices_dir`) |
| `classes.json`, `selected_gene_idx.npy`, `gene_selection.json`, `run_config.json` | Class order, gene subset (all 460 by default) and the full argument list of the run |
