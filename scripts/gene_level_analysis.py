#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gene-level analysis of a finished MOSAIC run (Extended Data Tables 2 and 9).

Computes, on the test subset only:
  1) per-gene PCC between predicted and measured expression (mean / median / std)
  2) top-K and bottom-K genes by PCC
  3) PCC stratified by gene abundance and variance quartiles

Uses the predictions saved by the pipeline; nothing is retrained.

  python scripts/gene_level_analysis.py --run_dir results/mosaic_k6_seed42
"""
import _repo  # noqa: F401

import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np

from mosaic.pipeline import DEFAULT_DATA_ROOT, DEFAULT_GENES_LIST, IMAGE_PATHS, normalize_expr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gene-level PCC analysis on the test subset")
    p.add_argument("--run_dir", required=True, help="Output folder of scripts/train_mosaic.py")
    p.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--test_image", default="UC1")
    p.add_argument("--test_fraction", type=float, default=0.1)
    p.add_argument("--genes_list", default=DEFAULT_GENES_LIST, help="One gene symbol per line")
    p.add_argument("--log1p", dest="log1p", action="store_true", default=None,
                   help="Apply log1p to the measured expression before z-scoring. Default: read the "
                        "value used for training from <run_dir>/run_config.json (False if absent)")
    p.add_argument("--out_dir", default=None, help="Default: <run_dir>/gene_level_analysis")
    p.add_argument("--top_k", type=int, default=20)
    return p.parse_args()


def _safe_pcc(x: np.ndarray, y: np.ndarray) -> float:
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def _quantile_bins(values: np.ndarray) -> np.ndarray:
    q1, q2, q3 = np.quantile(values, [0.25, 0.50, 0.75])
    bins = np.zeros_like(values, dtype=np.int64)
    bins[values <= q1] = 1
    bins[(values > q1) & (values <= q2)] = 2
    bins[(values > q2) & (values <= q3)] = 3
    bins[values > q3] = 4
    return bins


def _read_genes(path: str, n_genes: int) -> List[str]:
    if not os.path.exists(path):
        return [f"gene_{i}" for i in range(n_genes)]
    with open(path, "r", encoding="utf-8") as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes if len(genes) == n_genes else [f"gene_{i}" for i in range(n_genes)]


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _plots(gene_rows: List[Dict], out_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pcc = np.array([float(r["pcc"]) for r in gene_rows])
    mean_gt = np.array([float(r["mean_gt"]) for r in gene_rows])
    var_gt = np.array([float(r["var_gt"]) for r in gene_rows])
    created = []

    plt.figure(figsize=(7, 4))
    plt.hist(pcc[np.isfinite(pcc)], bins=30, color="#4c72b0", edgecolor="white")
    plt.title("Per-gene PCC distribution (test subset)")
    plt.xlabel("PCC")
    plt.ylabel("Number of genes")
    plt.tight_layout()
    out = os.path.join(out_dir, "hist_pcc.pdf")
    plt.savefig(out)
    plt.close()
    created.append(out)

    for x, name, color, xlabel in [(mean_gt, "pcc_vs_mean", "#2a9d8f", "Mean measured expression"),
                                   (var_gt, "pcc_vs_variance", "#e76f51", "Variance of measured expression")]:
        plt.figure(figsize=(6, 5))
        plt.scatter(x, pcc, s=16, alpha=0.65, color=color)
        plt.xlabel(xlabel)
        plt.ylabel("PCC")
        plt.tight_layout()
        out = os.path.join(out_dir, f"{name}.pdf")
        plt.savefig(out)
        plt.close()
        created.append(out)
    return created


def main() -> None:
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    out_dir = args.out_dir or os.path.join(run_dir, "gene_level_analysis")
    os.makedirs(out_dir, exist_ok=True)
    test_image = args.test_image.upper()

    pred_path = os.path.join(run_dir, f"expr_pred_full_{test_image}.npy")
    idx_path = os.path.join(run_dir, f"test_indices_{test_image}_pct{int(round(args.test_fraction * 100))}.npy")
    gt_path = os.path.join(args.data_root, IMAGE_PATHS[test_image]["expr"])
    for pth in (pred_path, idx_path, gt_path):
        if not os.path.isfile(pth):
            raise FileNotFoundError(pth)

    log1p = args.log1p
    cfg_path = os.path.join(run_dir, "run_config.json")
    if log1p is None:
        log1p = False
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                log1p = bool(json.load(f).get("log1p_normalize", False))
    print(f"[INFO] measured expression normalisation: {'log1p + ' if log1p else ''}per-gene z-score")

    pred = np.load(pred_path)
    gt_raw = np.load(gt_path).astype(np.float32)
    # same target normalisation as training (per-gene z-score of the whole slide)
    gt, _, _ = normalize_expr(gt_raw, log1p=log1p)
    test_idx = np.load(idx_path).astype(np.int64)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")

    pred_t = pred[test_idx]
    gt_t = gt[test_idx]
    n_cells, n_genes = gt_t.shape
    genes = _read_genes(args.genes_list, n_genes)

    pcc = np.array([_safe_pcc(pred_t[:, g], gt_t[:, g]) for g in range(n_genes)])
    mean_gt = gt_raw[test_idx].mean(axis=0)
    var_gt = gt_raw[test_idx].var(axis=0)
    abundance_q = _quantile_bins(mean_gt)
    variance_q = _quantile_bins(var_gt)

    gene_rows = [{
        "gene_idx": i, "gene": genes[i], "pcc": float(pcc[i]),
        "mean_gt": float(mean_gt[i]), "var_gt": float(var_gt[i]),
        "abundance_quartile": int(abundance_q[i]), "variance_quartile": int(variance_q[i]),
    } for i in range(n_genes)]
    fields = list(gene_rows[0].keys())

    valid = np.isfinite(pcc)
    mean_pcc = float(np.mean(pcc[valid]))
    median_pcc = float(np.median(pcc[valid]))
    std_pcc = float(np.std(pcc[valid]))

    order = np.argsort(np.where(valid, pcc, -1e9))
    top_rows = [gene_rows[i] for i in order[-args.top_k:][::-1]]
    bot_rows = [gene_rows[i] for i in order[:args.top_k]]

    strat_rows = []
    for name, q_arr in [("abundance", abundance_q), ("variance", variance_q)]:
        for q in (1, 2, 3, 4):
            vals = pcc[(q_arr == q) & valid]
            strat_rows.append({"stratification": name, "quartile": q, "n_genes": int(np.sum(q_arr == q)),
                               "mean_pcc": float(np.mean(vals)) if len(vals) else float("nan"),
                               "median_pcc": float(np.median(vals)) if len(vals) else float("nan")})

    _write_csv(os.path.join(out_dir, "gene_pcc_all.csv"), gene_rows, fields)
    _write_csv(os.path.join(out_dir, f"gene_top{args.top_k}.csv"), top_rows, fields)
    _write_csv(os.path.join(out_dir, f"gene_bottom{args.top_k}.csv"), bot_rows, fields)
    _write_csv(os.path.join(out_dir, "gene_strata_abundance_variance.csv"), strat_rows,
               ["stratification", "quartile", "n_genes", "mean_pcc", "median_pcc"])
    figures = _plots(gene_rows, out_dir)

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"n_cells_test={n_cells}\nn_genes={n_genes}\n")
        f.write(f"mean_pcc={mean_pcc:.6f}\nmedian_pcc={median_pcc:.6f}\nstd_pcc={std_pcc:.6f}\n")
        f.write(f"top_k={args.top_k}\nfigures=\n" + "".join(f"- {p}\n" for p in figures))

    print(f"[OK] Output: {out_dir}")
    print(f"[OK] Test cells: {n_cells} | genes: {n_genes}")
    print(f"[OK] Mean PCC {mean_pcc:.4f} | median {median_pcc:.4f} | std {std_pcc:.4f}")
    print("[OK] Top genes: " + ", ".join(f"{r['gene']} ({r['pcc']:.3f})" for r in top_rows[:10]))


if __name__ == "__main__":
    main()
