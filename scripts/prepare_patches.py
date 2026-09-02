#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract 128x128 H&E patches centred on each nucleus from an AMLC Zarr store.

Outputs, in --out_dir:
  <cell_type>/<cell_id>.png   one RGB patch per cell, grouped by lineage
  manifest.csv                path, cell_type, cell_id, group, expr_idx, cx, cy
  expr.npy                    (n_cells, n_genes) float32, aligned with manifest rows

Run once per slide (UC6, UC7, UC1):

  python scripts/prepare_patches.py \
      --zarr_path data/raw/UC6_I.zarr \
      --csv_path  data/raw/UC6_I_cell_group_positions.csv \
      --out_dir   data/patches128_2_UC6 --img_size 128 --size_crop 1.0

--size_crop f enlarges the nucleus bounding box by a factor (1 + f) before cropping
(1.0 = 2x, the setting used for the paper).

The cell-type CSV (cell_id, cell_type, bbox-0..bbox-3, group) is produced in house
(Leiden clustering + curator mapping to nine lineages; see the paper, Methods 3.4).
"""
import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import zarr
from PIL import Image
from tqdm import tqdm


def sanitize_cell_type(s: str) -> str:
    return str(s).strip().replace(" ", "_").replace("/", "_")


def to_uint8(img):
    img = np.asarray(img).astype(np.float32)
    lo, hi = np.nanpercentile(img, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(img), np.nanmax(img)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(img, dtype=np.uint8)
    img = np.clip((img - lo) / (hi - lo), 0, 1)
    return (img * 255.0 + 0.5).astype(np.uint8)


def arr_to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        g = to_uint8(arr)
        return np.stack([g, g, g], -1)
    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):  # HWC
            if arr.shape[-1] == 1:
                g = to_uint8(arr[..., 0])
                return np.stack([g, g, g], -1)
            return to_uint8(arr[..., :3])
        if arr.shape[0] in (1, 3, 4):  # CHW
            if arr.shape[0] == 1:
                g = to_uint8(arr[0])
                return np.stack([g, g, g], -1)
            return to_uint8(arr[:3].transpose(1, 2, 0))
    raise RuntimeError(f"Unsupported image layout: {arr.shape}")


def main():
    ap = argparse.ArgumentParser(description="Extract per-cell H&E patches from an AMLC Zarr store")
    ap.add_argument("--zarr_path", required=True, help="e.g. data/raw/UC6_I.zarr")
    ap.add_argument("--csv_path", required=True, help="e.g. data/raw/UC6_I_cell_group_positions.csv")
    ap.add_argument("--out_dir", required=True, help="e.g. data/patches128_2_UC6")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--size_crop", type=float, default=1.0,
                    help="Linear enlargement of the nucleus bbox: side * (1 + size_crop); 1.0 = 2x")
    ap.add_argument("--max_cells", type=int, default=None, help="Smoke test: process only N cells")
    args = ap.parse_args()

    if not os.path.exists(args.zarr_path):
        raise FileNotFoundError(f"Zarr not found: {args.zarr_path}")
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path, low_memory=False)
    df = df[df["cell_type"].notna()].copy()
    df["cell_type"] = df["cell_type"].apply(sanitize_cell_type)

    z = zarr.open_group(args.zarr_path, mode="r")
    img_arr = z["images/HE_original/0"]  # (C, H, W)
    expr = z["tables/anucleus/X"]  # (N, genes)
    obs_cell_ids = z["tables/anucleus/obs/cell_id"][:]
    id_to_idx: Dict[int, int] = {int(cid): i for i, cid in enumerate(obs_cell_ids.tolist())}

    records: List[Dict] = []
    missing = 0
    for _, row in df.iterrows():
        cid = int(row["cell_id"])
        if cid not in id_to_idx:
            missing += 1
            continue
        records.append(dict(
            cell_id=cid,
            cell_type=row["cell_type"],
            group=row.get("group", "train"),
            # CSV columns are (bbox-0, bbox-1, bbox-2, bbox-3) = (y0, x0, y1, x1)
            bbox=(int(row["bbox-1"]), int(row["bbox-0"]), int(row["bbox-2"]), int(row["bbox-3"])),
            expr_idx=id_to_idx[cid],
        ))
    if missing:
        print(f"[INFO] Skipped {missing} cells from the CSV without expression in the Zarr store.")

    if args.max_cells:
        records = records[: args.max_cells]

    classes = sorted({r["cell_type"] for r in records})
    for c in classes:
        os.makedirs(os.path.join(args.out_dir, c), exist_ok=True)

    n = len(records)
    genes = expr.shape[1]
    expr_path = os.path.join(args.out_dir, "expr.npy")
    expr_mm = np.lib.format.open_memmap(expr_path, mode="w+", dtype=np.float32, shape=(n, genes))

    manifest_rows = []
    H, W = img_arr.shape[1], img_arr.shape[2]
    skipped_invalid_bbox = 0
    skipped_empty_patch = 0
    out_dir_name = os.path.basename(os.path.normpath(args.out_dir))

    for i, rec in enumerate(tqdm(records, desc="Saving patches")):
        x0, y0, y1, x1 = rec["bbox"]
        if args.size_crop != 0.0:
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            w = (x1 - x0) * (1.0 + args.size_crop)
            h = (y1 - y0) * (1.0 + args.size_crop)
            x0 = int(round(cx - w / 2.0))
            x1 = int(round(cx + w / 2.0))
            y0 = int(round(cy - h / 2.0))
            y1 = int(round(cy + h / 2.0))

        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, y0), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            skipped_invalid_bbox += 1
            continue

        patch = img_arr[:, y0:y1, x0:x1]
        if patch.size == 0:
            skipped_empty_patch += 1
            continue

        pil = Image.fromarray(arr_to_rgb(np.moveaxis(patch, 0, -1)))
        pil = pil.resize((args.img_size, args.img_size), Image.Resampling.LANCZOS)
        rel_path = os.path.join(out_dir_name, rec["cell_type"], f"{rec['cell_id']}.png")
        pil.save(os.path.join(args.out_dir, rec["cell_type"], f"{rec['cell_id']}.png"))

        expr_mm[i] = expr[rec["expr_idx"]]

        manifest_rows.append(dict(
            path=rel_path,
            cell_type=rec["cell_type"],
            cell_id=rec["cell_id"],
            group=rec["group"],
            expr_idx=i,  # row of expr.npy
            cx=(x0 + x1) / 2.0,
            cy=(y0 + y1) / 2.0,
        ))

    expr_mm.flush()
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    if len(manifest_rows) == 0:
        raise ValueError("No patch was written; check the input files.")
    if len(manifest_rows) != n:
        print(f"[WARN] Expected {n} patches but only {len(manifest_rows)} were written "
              f"({skipped_invalid_bbox} invalid bboxes, {skipped_empty_patch} empty patches). "
              f"expr.npy has {n} rows: rows without a patch must be dropped before training.")

    print(f"[OK] Manifest saved to {manifest_path}")
    print(f"[OK] Aligned expression saved to {expr_path}")
    print(f"[INFO] Patches: {len(manifest_rows)} | genes: {genes} | cell types: {len(classes)}")


if __name__ == "__main__":
    main()
