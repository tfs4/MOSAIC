#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase A regression metrics for a finished MOSAIC run.

On the validation and test subsets, compares predicted vs measured expression
(same per-gene z-score as training) and writes:

  <run_dir>/expression_metrics.csv           mean per-gene PCC / Spearman, MAE, RMSE
  <run_dir>/expression_metrics_per_gene.csv  per-gene PCC / Spearman / MAE / RMSE (test)

Assumes Phase A completed and saved expr_pred_full_<SLIDE>.npy plus the split indices.

  python scripts/expression_metrics.py --run_dir results/mosaic_k6_seed42
"""
import _repo  # noqa: F401

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mosaic.pipeline import DEFAULT_DATA_ROOT, IMAGE_PATHS, normalize_expr


def _read_run_config(run_dir: str) -> dict:
    path = os.path.join(run_dir, "run_config.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    if kind == "pearson":
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(x, y)[0, 1]
    elif kind == "spearman":
        with np.errstate(invalid="ignore", divide="ignore"):
            c = spearmanr(x, y).correlation
    else:
        raise ValueError(kind)
    return float(c) if c is not None and np.isfinite(c) else float("nan")


def _per_gene_vector(pred: np.ndarray, gt: np.ndarray, kind: str) -> np.ndarray:
    return np.array([_safe_corr(pred[:, g], gt[:, g], kind) for g in range(gt.shape[1])],
                    dtype=np.float64)


def _mean_finite(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.mean(values[valid])) if valid.any() else float("nan")


def _split_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    err = pred - gt
    abs_err = np.abs(err)
    sq_err = err ** 2
    pcc = _per_gene_vector(pred, gt, "pearson")
    spearman = _per_gene_vector(pred, gt, "spearman")
    per_gene_mae = abs_err.mean(axis=0)
    per_gene_rmse = np.sqrt(sq_err.mean(axis=0))
    return {
        "n_cells": int(pred.shape[0]),
        "n_genes": int(pred.shape[1]),
        "mean_per_gene_pcc": _mean_finite(pcc),
        "median_per_gene_pcc": float(np.nanmedian(pcc)),
        "mean_per_gene_spearman": _mean_finite(spearman),
        "median_per_gene_spearman": float(np.nanmedian(spearman)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(sq_err.mean())),
        "mean_per_gene_mae": float(per_gene_mae.mean()),
        "mean_per_gene_rmse": float(per_gene_rmse.mean()),
        "_pcc": pcc,
        "_spearman": spearman,
        "_mae": per_gene_mae,
        "_rmse": per_gene_rmse,
    }


def _load_split(run_dir: str, data_root: str, image: str, kind: str,
                fraction: float, log1p: bool):
    pred_path = os.path.join(run_dir, f"expr_pred_full_{image}.npy")
    idx_path = os.path.join(run_dir, f"{kind}_indices_{image}_pct{int(round(fraction * 100))}.npy")
    gt_path = os.path.join(data_root, IMAGE_PATHS[image]["expr"])
    for path in (pred_path, idx_path, gt_path):
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(
                f"Missing or empty {path}. Phase A must have saved full predictions "
                f"(do not pass --no_save_full_predictions)."
            )
    pred = np.load(pred_path)
    gt_raw = np.load(gt_path).astype(np.float32)
    gt, _, _ = normalize_expr(gt_raw, log1p=log1p)
    if pred.shape != gt.shape:
        raise ValueError(f"{image}: pred {pred.shape} vs measured {gt.shape}")
    idx = np.load(idx_path).astype(np.int64)
    return pred[idx], gt[idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCC, Spearman, MAE and RMSE for Phase A predictions")
    p.add_argument("--run_dir", required=True, help="Output folder of scripts/train_mosaic.py")
    p.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--val_image", default=None)
    p.add_argument("--test_image", default=None)
    p.add_argument("--val_fraction", type=float, default=None)
    p.add_argument("--test_fraction", type=float, default=None)
    p.add_argument("--log1p", dest="log1p", action="store_true", default=None,
                   help="log1p before z-scoring measured expression (default: run_config.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    cfg = _read_run_config(run_dir)

    val_image = (args.val_image or cfg.get("val_image") or "UC7").upper()
    test_image = (args.test_image or cfg.get("test_image") or "UC1").upper()
    val_frac = float(args.val_fraction if args.val_fraction is not None else cfg.get("val_fraction", 0.2))
    test_frac = float(args.test_fraction if args.test_fraction is not None else cfg.get("test_fraction", 0.1))
    log1p = args.log1p
    if log1p is None:
        log1p = bool(cfg.get("log1p_normalize", False))

    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] measured expression: {'log1p + ' if log1p else ''}per-gene z-score")

    rows = []
    test_detail = None
    for kind, image, frac in (("val", val_image, val_frac), ("test", test_image, test_frac)):
        pred, gt = _load_split(run_dir, args.data_root, image, kind, frac, log1p)
        metrics = _split_metrics(pred, gt)
        detail = {k: metrics.pop(k) for k in ("_pcc", "_spearman", "_mae", "_rmse")}
        row = {"split": kind, "image": image, "fraction": frac, **metrics}
        rows.append(row)
        print(
            f"[OK] {kind} {image} {int(round(frac * 100))}%  n={metrics['n_cells']:,}  "
            f"PCC={metrics['mean_per_gene_pcc']:.4f}  "
            f"Spearman={metrics['mean_per_gene_spearman']:.4f}  "
            f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}"
        )
        if kind == "test":
            test_detail = (detail, metrics["n_genes"])

    summary_path = os.path.join(run_dir, "expression_metrics.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"[OK] wrote {summary_path}")

    if test_detail is not None:
        detail, n_genes = test_detail
        per_gene = pd.DataFrame({
            "gene_idx": np.arange(n_genes),
            "pcc": detail["_pcc"],
            "spearman": detail["_spearman"],
            "mae": detail["_mae"],
            "rmse": detail["_rmse"],
        })
        per_gene_path = os.path.join(run_dir, "expression_metrics_per_gene.csv")
        per_gene.to_csv(per_gene_path, index=False)
        print(f"[OK] wrote {per_gene_path}")


if __name__ == "__main__":
    main()
