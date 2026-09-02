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
│   ├── verify_dataset.py        sanity checks and split sizes vs the paper
│   ├── train_mosaic.py          main experiment (Phase A + Phase B)          <-- start here
│   ├── run_ablation.py          K / aggregation ablation on identical val/test cells
│   ├── run_k_sweep.py           unattended K + seed sweep (server)
│   ├── predict_expression.py    Phase A inference for a whole slide from a checkpoint
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
Learning Challenge (AMLC)** dataset (10x Xenium 460-gene panel with paired H&E). They are
distributed by the challenge organisers and are **not** included here.

Expected layout (details in [`data/README.md`](data/README.md)):

```
data/patches128_2_UC6/{manifest.csv, expr.npy, <lineage>/<cell_id>.png}   # train  223,781 cells
data/patches128_2_UC7/...                                                   # val    144,702 cells (20 % used)
data/patches128_2_UC1/...                                                   # test   202,528 cells (10 % used)
```

Build these folders from the AMLC Zarr stores and the cell-type tables:

```bash
for UC in UC6 UC7 UC1; do
  python scripts/prepare_patches.py \
      --zarr_path data/raw/${UC}_I.zarr \
      --csv_path  data/raw/${UC}_I_cell_group_positions.csv \
      --out_dir   data/patches128_2_${UC} --img_size 128 --size_crop 1.0
done
python scripts/verify_dataset.py --data_root data          # split sizes must match the paper
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

Seed robustness (paper: 0.600 +- 0.001 accuracy over seeds 42, 43, 44):

```bash
for S in 43 44; do
  python scripts/train_mosaic.py --k_neighbors 6 --seed $S --out_dir results/mosaic_k6_seed$S
done
```

### 4.2 Tables and gene-level analysis from a finished run

```bash
# Table 1 / ED Table 3-4: chance, majority, MOSAIC, and the ceiling MLP trained on measured expression
python scripts/classification_tables.py --run_dir results/mosaic_k6_seed42

# ED Table 2 and 9: per-gene PCC, top/bottom-20 genes, abundance and variance quartiles
python scripts/gene_level_analysis.py --run_dir results/mosaic_k6_seed42
```

### 4.3 Neighbourhood ablation (K = 0, 4, 6, 8, 12)

```bash
python scripts/run_ablation.py --ks 0,4,6,8,12 --aggregations attention \
    --seed 42 --batch_size 64 --num_workers 8 \
    --ablation_out_dir results/ablation_k
python scripts/plot_ablation.py --input_dir results/ablation_k
```

`K = 0` is the backbone-only baseline (no neighbourhood). Every configuration re-uses the
validation/test indices written by the first run, so all numbers are computed on exactly
the same cells. To compare aggregation operators at fixed K use, e.g.,
`--ks 6 --aggregations attention,mean,max`. Resume an interrupted grid with
`--start_from N --indices_dir results/ablation_k/k0_agg_n_a`.

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

## 5. Reproducibility notes

* All random sources are seeded (`--seed`): Python, NumPy, PyTorch (CPU/CUDA), DataLoader
  shuffling and per-worker augmentation; cuDNN runs in deterministic mode and TF32 is
  disabled. Bit-exact repetition still requires the same GPU model, driver and library
  versions; across hardware expect differences in the third decimal of the metrics.
* The validation/test subsets depend only on the manifests and `--seed`; `scripts/verify_dataset.py`
  checks that they reproduce the sizes reported in the paper (28,941 and 20,253 cells).
* `results.csv` and `run_config.json` in each run folder record every setting used.

## 6. Hardware notes

All experiments ran on a single NVIDIA GPU with mixed precision disabled (full fp32 for
determinism). GPU memory scales with `--batch_size x (1 + K)` patches; if you run out of
memory reduce `--batch_size` (the learning rate `--lr_expr` may then need to be lowered
accordingly) and keep `--num_workers` close to the number of CPU cores available for JPEG/PNG
decoding, which is the usual bottleneck.

## 7. Citation

See [`CITATION.cff`](CITATION.cff).

## 8. License

To be defined by the authors before public release.
