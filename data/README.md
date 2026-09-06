# `data/` - expected layout

Nothing in this folder is versioned (see `.gitignore`). The pipeline expects one folder
per slide, each produced by `scripts/prepare_patches.py`:

```
data/
├── raw/                                  # optional: AMLC Zarr stores + cell-type CSVs
│   ├── UC6_I.zarr/
│   ├── UC6_I_cell_group_positions.csv
│   ├── UC7_I.zarr/  UC7_I_cell_group_positions.csv
│   └── UC1_I.zarr/  UC1_I_cell_group_positions.csv
├── patches128_2_UC6/                     # training slide (223,781 cells)
│   ├── manifest.csv                      # path, cell_type, cell_id, group, expr_idx, cx, cy
│   ├── expr.npy                          # (n_cells, 460) float32, aligned with manifest rows
│   ├── B_Cells/<cell_id>.png
│   ├── Endothelial_Cells/...
│   └── ...                               # one sub-folder per lineage (9 in total)
├── patches128_2_UC7/                     # validation slide (144,702 cells, 20% used)
└── patches128_2_UC1/                     # test slide (202,528 cells, 10% used)
```

Folder names are hard-wired in `mosaic/pipeline.py::IMAGE_PATHS`; the `path` column of the
manifest is resolved relative to the folder that contains the manifest, so the folders can
live anywhere as long as `--data_root` points to their parent (symbolic links are fine).

## Download the prepared archives

Prepared zip files (one per slide) live on Google Drive. From the **repository root**,
install `gdown` and download **one archive at a time**. Use `python3 -m gdown` if the
`gdown` binary is missing or too old (`--fuzzy` unsupported). Use `python3` if `python`
is not on `PATH`.

```bash
pip install -U gdown
mkdir -p data && cd data

# UC1 — 202,528 patches
python3 -m gdown "https://drive.google.com/uc?id=1mG9DtWTTpNkAJkKd4H9PASnfFyGXIKQ-" -O patches128_2_UC1.zip
unzip -o patches128_2_UC1.zip
ls patches128_2_UC1/manifest.csv patches128_2_UC1/expr.npy

# UC6 — 223,781 patches
python3 -m gdown "https://drive.google.com/uc?id=1dIQ9PRF2AMd9z_MS4r5vR8Jpb-sR2uAV" -O patches128_2_UC6.zip
unzip -o patches128_2_UC6.zip
ls patches128_2_UC6/manifest.csv patches128_2_UC6/expr.npy

# UC7 — 144,702 patches
python3 -m gdown "https://drive.google.com/uc?id=15yNvqGzWKbMyRwtF0sSKHMQdWSpm6F7_" -O patches128_2_UC7.zip
unzip -o patches128_2_UC7.zip
ls patches128_2_UC7/manifest.csv patches128_2_UC7/expr.npy

cd ..
rm -f data/patches128_2_UC*.zip
```

`manifest.csv` and `expr.npy` must sit directly under `data/patches128_2_UC*/`. If unzip
creates an extra nested folder, move the inner directory up.

To fetch all three in one go:

```bash
python3 tools/download_datasets_gdrive.py \
    --uc1 "https://drive.google.com/file/d/1mG9DtWTTpNkAJkKd4H9PASnfFyGXIKQ-/view?usp=sharing" \
    --uc6 "https://drive.google.com/file/d/1dIQ9PRF2AMd9z_MS4r5vR8Jpb-sR2uAV/view?usp=sharing" \
    --uc7 "https://drive.google.com/file/d/15yNvqGzWKbMyRwtF0sSKHMQdWSpm6F7_/view?usp=sharing" \
    --data_root data
```

## Count the patches

```bash
python3 scripts/count_images.py --data_root data
```

Expected result (the `folder` path follows `--data_root`; the counts must be exact):

```
slide    folder                                         images
----------------------------------------------------------------
UC1      /workspace/MOSAIC/data/patches128_2_UC1       202,528
UC6      /workspace/MOSAIC/data/patches128_2_UC6       223,781
UC7      /workspace/MOSAIC/data/patches128_2_UC7       144,702
----------------------------------------------------------------
TOTAL    /workspace/MOSAIC/data                        571,011
```

Then check manifests, `expr.npy` and the paper split sizes:

```bash
python3 scripts/verify_dataset.py --data_root data
```

## Where the data come from

The three slides (UC1, UC6, UC7) are ulcerative-colitis colon sections from the
**Autoimmune Multimodal Learning Challenge (AMLC)** dataset (Xenium 460-gene panel with
paired H&E). The raw Zarr stores are distributed by the challenge organisers. After
obtaining them, extract the patches with:

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

`*_cell_group_positions.csv` holds one row per nucleus with `cell_id`, the nucleus
bounding box (`bbox-0` .. `bbox-3`) and the curated `cell_type` label (nine lineages
obtained from Leiden clustering of the Xenium profiles and marker-based annotation; see
the paper, Methods).
