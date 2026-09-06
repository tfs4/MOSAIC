# MOSAIC - Morphology-Oriented Spatial Attention for Inferring Cell Expression

Code accompanying the paper

> **MOSAIC: single-cell expression from H&E histology.**
> T. Fernandes, C. Camilo, H. T. I. Nakaya, A. Simizo, M. C. C. Morais, M. F. Vesco, G. R. de-Mira.
> *Scientific Reports* (under review).

MOSAIC predicts a 460-gene Xenium expression profile for every nucleus of an H&E slide from
the cell's own 128x128 patch plus the patches of its **K = 6 spatial nearest neighbours**
(Phase A), and then assigns one of **nine cell lineages** from the predicted expression
alone (Phase B). Training, validation and test cells come from **three different slides**
(UC6 / UC7 / UC1), so no cell of the test slide is ever seen during training.

```
            Phase A                                         Phase B
H&E patch ─┐                                        predicted 460-gene
K=6 neigh. ─┼─ ConvNeXt-Tiny (GAP‖GeM) ─ attention ─ MLP head ─▶ expression ─▶ MLP (256-128) ─▶ lineage
positions  ─┘   shared encoder            (target = query)                    460 → 9
```

Reported results (UC6 train, UC7 20 % val, UC1 10 % test, seed 42):
mean per-gene PCC **0.205** on 20,253 held-out cells; lineage accuracy **0.599**,
macro-F1 **0.514**, weighted-F1 **0.593** (chance 0.111, majority class 0.378).

---

## 1. Repository layout

```
MOSAIC/
├── mosaic/
│   └── pipeline.py              core library: datasets, models, training, evaluation, CLI
├── scripts/                     entry points (run from the repository root)
│   ├── prepare_patches.py       Zarr slide -> per-cell PNG patches + manifest.csv + expr.npy
│   ├── count_images.py          count PNG patches per slide (must total 571,011)
│   ├── verify_dataset.py        sanity checks and split sizes vs the paper
│   ├── train_mosaic.py          main experiment (Phase A + Phase B)          <-- start here
│   ├── run_ablation.py          K / aggregation ablation on identical val/test cells
│   ├── run_slide_rotation.py    rotate which slide is train / val / test
│   ├── run_seed_sweep.py        paper protocol over extra seeds (0, 20; 42 already run)
│   ├── run_k_sweep.py           unattended K + seed sweep (server)
│   ├── predict_expression.py    Phase A inference for a whole slide from a checkpoint
│   ├── expression_metrics.py    Phase A PCC / Spearman / MAE / RMSE on val and test
│   ├── gene_level_analysis.py   per-gene PCC, top/bottom genes, abundance/variance strata
│   ├── classification_tables.py Phase B baselines, per-class metrics, class counts
│   ├── plot_ablation.py         ablation bar charts
│   └── plot_neighborhood_figure.py   target cell + K neighbours figure
├── resources/
│   ├── gene_panel_460.txt       gene symbols in the column order of expr.npy
│   └── marker_genes_top10_per_cell_type.csv   curated marker genes per lineage
├── experimental/                variants NOT used in the paper (patch pre-processing)
├── tools/download_datasets_gdrive.py   internal helper to fetch prepared data on our servers
├── data/                        (git-ignored) prepared slides - see data/README.md
├── results/                     (git-ignored) outputs - see results/README.md
├── .gitignore
├── requirements.txt
├── CITATION.cff
└── README.md
```

## 2. Installation

Tested on Linux with Python >= 3.9, PyTorch 2.x (CUDA 12) and a single NVIDIA GPU.

```bash
git clone <this repository> MOSAIC && cd MOSAIC
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # pick your CUDA
pip install -r requirements.txt
```

CPU-only execution works but is impractical for training (each training step encodes
1 + K = 7 patches per cell for ~224k cells).

## 3. Data

The three slides are ulcerative-colitis colon sections of the **Autoimmune Multimodal
Learning Challenge (AMLC)** dataset (10x Xenium 460-gene panel with paired H&E). The raw
Zarr stores are distributed by the challenge organisers and are **not** included here.
Prepared 128×128 patch archives (one zip per slide) can be downloaded from Google Drive
as below.

Expected layout (details in [`data/README.md`](data/README.md)):

```
data/patches128_2_UC6/{manifest.csv, expr.npy, <lineage>/<cell_id>.png}   # train  223,781 cells
data/patches128_2_UC7/...                                                   # val    144,702 cells (20 % used)
data/patches128_2_UC1/...                                                   # test   202,528 cells (10 % used)
```

### 3.1 Download the prepared archives

From the repository root. If `python` is not on `PATH`, use `python3`. Download **one
slide at a time** (each zip is large) and confirm `manifest.csv` + `expr.npy` before
starting the next:

```bash
pip install -U gdown
mkdir -p data && cd data

python3 -m gdown "https://drive.google.com/uc?id=1mG9DtWTTpNkAJkKd4H9PASnfFyGXIKQ-" -O patches128_2_UC1.zip
unzip -o patches128_2_UC1.zip
ls patches128_2_UC1/manifest.csv patches128_2_UC1/expr.npy

python3 -m gdown "https://drive.google.com/uc?id=1dIQ9PRF2AMd9z_MS4r5vR8Jpb-sR2uAV" -O patches128_2_UC6.zip
unzip -o patches128_2_UC6.zip
ls patches128_2_UC6/manifest.csv patches128_2_UC6/expr.npy

python3 -m gdown "https://drive.google.com/uc?id=15yNvqGzWKbMyRwtF0sSKHMQdWSpm6F7_" -O patches128_2_UC7.zip
unzip -o patches128_2_UC7.zip
ls patches128_2_UC7/manifest.csv patches128_2_UC7/expr.npy

cd ..
rm -f data/patches128_2_UC*.zip
```

Use `python3 -m gdown` rather than the `gdown` binary: some images ship an older CLI
without `--fuzzy`. If a zip unpacks one extra nested folder (e.g.
`data/patches128_2_UC1/patches128_2_UC1/`), move the inner folder up so that
`manifest.csv` sits directly under `data/patches128_2_UC*`.

Alternatively, `tools/download_datasets_gdrive.py` does the same three downloads and
renames the folders automatically (`pip install gdown` first):

```bash
python3 tools/download_datasets_gdrive.py \
    --uc1 "https://drive.google.com/file/d/1mG9DtWTTpNkAJkKd4H9PASnfFyGXIKQ-/view?usp=sharing" \
    --uc6 "https://drive.google.com/file/d/1dIQ9PRF2AMd9z_MS4r5vR8Jpb-sR2uAV/view?usp=sharing" \
    --uc7 "https://drive.google.com/file/d/15yNvqGzWKbMyRwtF0sSKHMQdWSpm6F7_/view?usp=sharing" \
    --data_root data
```

### 3.2 Count the patches

After the three folders are in place, from the repository root:

```bash
python3 scripts/count_images.py --data_root data
```

The counts must match the paper (one PNG per nucleus):

```
slide    folder                                         images
----------------------------------------------------------------
UC1      /workspace/MOSAIC/data/patches128_2_UC1       202,528
UC6      /workspace/MOSAIC/data/patches128_2_UC6       223,781
UC7      /workspace/MOSAIC/data/patches128_2_UC7       144,702
----------------------------------------------------------------
TOTAL    /workspace/MOSAIC/data                        571,011
```

The `folder` column is the absolute path, so it will differ if the repo is not at
`/workspace/MOSAIC`. The three integers and the total **571,011** must match. Then:

```bash
python3 scripts/verify_dataset.py --data_root data
```

which also checks that the seed-42 val/test subset sizes are 28,941 (UC7 20 %) and
20,253 (UC1 10 %).

### 3.3 Build the folders from AMLC Zarr (optional)

If you have the original Zarr stores instead of the zips:

```bash
for UC in UC6 UC7 UC1; do
  python scripts/prepare_patches.py \
      --zarr_path data/raw/${UC}_I.zarr \
      --csv_path  data/raw/${UC}_I_cell_group_positions.csv \
      --out_dir   data/patches128_2_${UC} --img_size 128 --size_crop 1.0
done
python scripts/count_images.py --data_root data
python scripts/verify_dataset.py --data_root data
```

`--size_crop 1.0` doubles the nucleus bounding box before resizing to 128x128 (2x crop).
If you already have the prepared folders elsewhere, point `--data_root` to their parent
or create symbolic links inside `data/`.

## 4. Reproducing the paper

### 4.1 Main experiment (Table 1, Fig. 2-3, Extended Data Tables 1-4, 9)

```bash
python scripts/train_mosaic.py \
    --data_root data \
    --train_image UC6 --val_image UC7 --test_image UC1 \
    --val_fraction 0.2 --test_fraction 0.1 \
    --k_neighbors 6 --aggregation_mode attention \
    --seed 42 --batch_size 64 --num_workers 8 \
    --out_dir results/mosaic_k6_seed42
```

What happens:

1. **Load** the three slides; z-score expression per gene within each slide.
2. **Split**: 100 % of UC6 for training; stratified 20 % of UC7 for validation; stratified
   10 % of UC1 for test (`sklearn.train_test_split`, `random_state = --seed`). The exact
   indices are written to `val_indices_UC7_pct20.npy` / `test_indices_UC1_pct10.npy`.
3. **Phase A**: ConvNeXt-Tiny (ImageNet weights) encodes the target and its K neighbours;
   attention aggregation (target = query); head predicts 460 genes. AdamW, lr 2e-4,
   EMA 0.9995, ReduceLROnPlateau, MSE loss with marker genes weighted x2 (see 4.4),
   early stopping on validation mean per-gene PCC (patience 6, max 30 epochs).
4. **Inference** of the whole UC6 / UC7 / UC1 slides with the best Phase A model
   (`expr_pred_full_*.npy`); mean per-gene PCC on the test subset.
5. **Phase B**: MLP 460-256-128-9 (ReLU, dropout 0.5) trained on the *predicted* expression
   of UC6, early stopping on UC7 macro-F1 (patience 5, max 30 epochs), evaluated on the
   UC1 test subset with the training mean/std.

Outputs are listed in [`results/README.md`](results/README.md). Phase A is the expensive
part: every step encodes 1 + K patches per cell for ~224k cells, i.e. on the order of one
to two hours per epoch on a single modern GPU in our runs; early stopping usually ends
training well before the 30-epoch cap. Phase B takes a few minutes.

Seed robustness (seeds 0 and 20) is in **section 5.2**. Each extra seed redraws the
UC7 20 % / UC1 10 % subsets and retrains; combine those folders with
`results/mosaic_k6_seed42` for the three-seed mean ± s.d.

### 4.2 Tables and gene-level analysis from a finished run

```bash
# Table 1 / ED Table 3-4: chance, majority, MOSAIC, and the ceiling MLP trained on measured expression
python scripts/classification_tables.py --run_dir results/mosaic_k6_seed42

# Phase A: mean per-gene PCC / Spearman, MAE, RMSE (val + test) -> expression_metrics.csv
python scripts/expression_metrics.py --run_dir results/mosaic_k6_seed42

# ED Table 2 and 9: per-gene PCC, top/bottom-20 genes, abundance and variance quartiles
python scripts/gene_level_analysis.py --run_dir results/mosaic_k6_seed42
```

### 4.3 Neighbourhood and aggregation ablations

The runs that actually exist under `results/` (K = 0, K = 4 attention, K = 6 mean,
same cells as seed 42) are documented with copy-paste commands in **section 5**.
`K = 0` is the backbone-only baseline. Every ablation must pass
`--indices_dir results/mosaic_k6_seed42` so the numbers stay on the same cells.

### 4.3.1 Rotate train / val / test slides

Same fractions as the paper (100 % train, 20 % val, 10 % test), but each of the three
slides takes a turn as the training set. The original assignment (UC6 / UC7 / UC1) is
skipped unless `--include_paper_split` is set. Outputs go under `results/slide_rotation/`,
not `results/mosaic_k6_seed42`.

```bash
python3 scripts/run_slide_rotation.py \
    --data_root data --seed 42 --batch_size 64 --num_workers 20 \
    --out_base results/slide_rotation
```

This runs, in order:

1. train UC7 (100 %) | val UC1 (20 %) | test UC6 (10 %) → `results/slide_rotation/trainUC7_valUC1_testUC6_seed42/`
2. train UC1 (100 %) | val UC6 (20 %) | test UC7 (10 %) → `results/slide_rotation/trainUC1_valUC6_testUC7_seed42/`

A one-row-per-run summary is written to `results/slide_rotation/rotation_results.csv`. Resume
after an interruption with `--start_from 1`.

### 4.4 Options that change the protocol

| Flag | Default | Effect |
|---|---|---|
| `--k_neighbors K` | 6 | Number of spatial nearest neighbours (Euclidean on nucleus centroids); 0 = no context |
| `--aggregation_mode` | `attention` | `attention`, `mean` or `max` pooling of neighbour features |
| `--use_graph` | off | Adds a learned embedding of (dx, dy, distance, sin, cos) to each neighbour. **Off in the paper** ("no-graph" configuration). |
| `--log1p_normalize` | off | log(1 + x) before per-gene z-scoring of the expression targets |
| `--markers_csv / --genes_list` | `resources/...` | When both files are valid, marker genes get weight `--marker_loss_weight` (2.0) in the Phase A MSE. Pass `--markers_csv ''` for a plain MSE. |
| `--gene_select_method` | `none` | Optional gene subset for Phase B (`pcc_topk`, `kbest`). Not used in the paper. |
| `--indices_dir DIR` | - | Re-use `val_indices_*` / `test_indices_*` from a previous run |
| `--no_save_full_predictions` | off | Skip the ~260 MB `expr_pred_full_*.npy` files |

Run `python scripts/train_mosaic.py --help` for the complete list.

### 4.5 Inference with a trained model

```bash
python scripts/predict_expression.py \
    --checkpoint results/mosaic_k6_seed42/expr_model_UC6.pt \
    --image UC1 --k_neighbors 6 --aggregation_mode attention
```

### 4.6 Figure of a cell and its neighbours (Extended Data Fig. 1)

```bash
python scripts/plot_neighborhood_figure.py --image_key UC6 --k 6 --seed 42 \
    --auto_select_best_patch --marker_style ring \
    --out_path results/figures/k6_neighbors_UC6.pdf
```

## 5. Tutorial: every experiment currently in `results/`

This section reproduces **only the runs that already exist locally under `results/`**.
It is not a full ablation grid. Commands assume the repository root, the three
prepared slides in `data/` (section 3), and a single GPU. Use `python3` if `python`
is not on `PATH`. On a rented machine, start each command inside `tmux` so an SSH
drop does not kill Phase A (the checkpoint is written only after Phase A ends).

Shared protocol unless a subsection says otherwise:

| Setting | Value |
|---|---|
| Train | UC6, 100 % |
| Validation | UC7, stratified 20 % |
| Test | UC1, stratified 10 % |
| `k_neighbors` | 6 (except the K ablation) |
| `aggregation_mode` | `attention` (except the mean run) |
| `use_graph` / `log1p_normalize` | off |
| `--batch_size` | 64 |
| `--num_workers` | 20 |

Folder names below match the directories on disk, including the typo
`results/abalation_k` and the copied mean folder `k6_aff_mean`.

**Order.** Run 5.1 first. The K and aggregation ablations reuse its val/test
indices through `--indices_dir results/mosaic_k6_seed42`. Seeds 0 and 20 must
**not** reuse those indices: each seed redraws the split.

### 5.1 Paper protocol — `results/mosaic_k6_seed42/`

K = 6, attention, seed 42. This is the reference run. All later ablations
compare against these same 28,941 val / 20,253 test cells.

```bash
python3 scripts/train_mosaic.py \
    --data_root data \
    --train_image UC6 --val_image UC7 --test_image UC1 \
    --val_fraction 0.2 --test_fraction 0.1 \
    --k_neighbors 6 --aggregation_mode attention \
    --seed 42 --batch_size 64 --num_workers 20 \
    --out_dir results/mosaic_k6_seed42
```

When `results/mosaic_k6_seed42/results.csv` exists:

```bash
python3 scripts/expression_metrics.py --run_dir results/mosaic_k6_seed42 --data_root data
python3 scripts/classification_tables.py --run_dir results/mosaic_k6_seed42
python3 scripts/gene_level_analysis.py --run_dir results/mosaic_k6_seed42
```

### 5.2 Extra seeds — `results/seed_sweep/`

Same protocol as 5.1, new split and new weights per seed. Do **not** pass
`--indices_dir`. Seed 42 is already in `results/mosaic_k6_seed42/` and is not
re-run here.

| Folder | Seed |
|---|---|
| `results/seed_sweep/mosaic_k6_seed0/` | 0 |
| `results/seed_sweep/mosaic_k6_seed20/` | 20 |

Both seeds in one go (writes `results/seed_sweep/seed_sweep_results.csv`):

```bash
python3 scripts/run_seed_sweep.py \
    --data_root data \
    --seeds 0,20 \
    --batch_size 64 --num_workers 20 \
    --out_base results/seed_sweep
```

Or one seed at a time:

```bash
python3 scripts/train_mosaic.py \
    --data_root data \
    --train_image UC6 --val_image UC7 --test_image UC1 \
    --val_fraction 0.2 --test_fraction 0.1 \
    --k_neighbors 6 --aggregation_mode attention \
    --seed 0 --batch_size 64 --num_workers 20 \
    --out_dir results/seed_sweep/mosaic_k6_seed0

python3 scripts/train_mosaic.py \
    --data_root data \
    --train_image UC6 --val_image UC7 --test_image UC1 \
    --val_fraction 0.2 --test_fraction 0.1 \
    --k_neighbors 6 --aggregation_mode attention \
    --seed 20 --batch_size 64 --num_workers 20 \
    --out_dir results/seed_sweep/mosaic_k6_seed20
```

Resume the sweep if a seed already finished:

```bash
python3 scripts/run_seed_sweep.py \
    --data_root data --seeds 0,20 --num_workers 20 \
    --out_base results/seed_sweep --skip_existing
```

### 5.3 Neighbourhood size — `results/abalation_k/`

Same cells as seed 42 (`--indices_dir`). Attention for K > 0. The parent
folder on disk is spelled `abalation_k`.

| Folder | K | Aggregation |
|---|---|---|
| `results/abalation_k/k0_agg_n_a/` | 0 (target cell only) | none |
| `results/abalation_k/k4_agg_attention/` | 4 | attention |

K = 6 attention is 5.1, not repeated here. K = 8 / 12 were not run.

```bash
python3 scripts/run_ablation.py \
    --data_root data \
    --ks 0,4 --aggregations attention \
    --seed 42 --batch_size 64 --num_workers 20 \
    --indices_dir results/mosaic_k6_seed42 \
    --ablation_out_dir results/abalation_k
```

A single configuration (for example only K = 4):

```bash
python3 scripts/run_ablation.py \
    --data_root data \
    --ks 4 --aggregations attention \
    --seed 42 --batch_size 64 --num_workers 20 \
    --indices_dir results/mosaic_k6_seed42 \
    --ablation_out_dir results/abalation_k
```

### 5.4 Mean aggregation — `results/ablation_agg/`

K = 6, **mean** pooling of the six neighbours, same cells as seed 42. This is
the attention-vs-mean control. The script writes `k6_agg_mean/`; the local copy
may appear as `k6_aff_mean/` if the folder was renamed after the run.

| Folder | K | Aggregation |
|---|---|---|
| `results/ablation_agg/k6_agg_mean/` (or `k6_aff_mean/`) | 6 | mean |

```bash
python3 scripts/run_ablation.py \
    --data_root data \
    --ks 6 --aggregations mean \
    --seed 42 --batch_size 64 --num_workers 20 \
    --indices_dir results/mosaic_k6_seed42 \
    --ablation_out_dir results/ablation_agg
```

Do not run mean at K = 4. Compare this `results.csv` to
`results/mosaic_k6_seed42/results.csv` (K = 6 attention) and
`results/abalation_k/k0_agg_n_a/results.csv` (no neighbours).

### 5.5 What each finished folder should contain

See [`results/README.md`](results/README.md). The file that means the run is
complete is `results.csv`. Then, for any of the folders above:

```bash
python3 scripts/expression_metrics.py --run_dir <RUN_DIR> --data_root data
```

That writes `expression_metrics.csv` and `expression_metrics_per_gene.csv`
(Pearson, Spearman, MAE, RMSE on the val and test subsets).

## 6. Reproducibility notes

* All random sources are seeded (`--seed`): Python, NumPy, PyTorch (CPU/CUDA), DataLoader
  shuffling and per-worker augmentation; cuDNN runs in deterministic mode and TF32 is
  disabled. Bit-exact repetition still requires the same GPU model, driver and library
  versions; across hardware expect differences in the third decimal of the metrics.
* The validation/test subsets depend only on the manifests and `--seed`; `scripts/verify_dataset.py`
  checks that they reproduce the sizes reported in the paper (28,941 and 20,253 cells).
* `results.csv` and `run_config.json` in each run folder record every setting used.

## 7. Hardware notes

All experiments ran on a single NVIDIA GPU with mixed precision disabled (full fp32 for
determinism). GPU memory scales with `--batch_size x (1 + K)` patches; if you run out of
memory reduce `--batch_size` (the learning rate `--lr_expr` may then need to be lowered
accordingly) and keep `--num_workers` close to the number of CPU cores available for JPEG/PNG
decoding, which is the usual bottleneck.

## 8. Citation

See [`CITATION.cff`](CITATION.cff).

## 9. License

To be defined by the authors before public release.
