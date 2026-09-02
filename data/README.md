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

## Where the data come from

The three slides (UC1, UC6, UC7) are ulcerative-colitis colon sections from the
**Autoimmune Multimodal Learning Challenge (AMLC)** dataset (Xenium 460-gene panel with
paired H&E). They are distributed by the challenge organisers and cannot be redistributed
here. After obtaining the Zarr stores, extract the patches with:

```bash
for UC in UC6 UC7 UC1; do
  python scripts/prepare_patches.py \
      --zarr_path data/raw/${UC}_I.zarr \
      --csv_path  data/raw/${UC}_I_cell_group_positions.csv \
      --out_dir   data/patches128_2_${UC} --img_size 128 --size_crop 1.0
done
python scripts/verify_dataset.py --data_root data
```

`*_cell_group_positions.csv` holds one row per nucleus with `cell_id`, the nucleus
bounding box (`bbox-0` .. `bbox-3`) and the curated `cell_type` label (nine lineages
obtained from Leiden clustering of the Xenium profiles and marker-based annotation; see
the paper, Methods).
