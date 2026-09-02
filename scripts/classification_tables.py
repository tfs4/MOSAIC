#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B tables for a finished MOSAIC run (Table 1, Extended Data Tables 3 and 4).

Writes, in <run_dir>:
  classification_baselines_<TEST>.csv  chance, majority class, MOSAIC (predicted expression)
                                       and the ceiling control trained on measured expression
  classification_per_class_<TEST>.csv  precision / recall / F1 / support per lineage
  classification_class_counts.csv      class counts in the training slide and the test subset

  python scripts/classification_tables.py --run_dir results/mosaic_k6_seed42
"""
import _repo  # noqa: F401

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from mosaic.pipeline import (
    DEFAULT_DATA_ROOT,
    ExprClassifier,
    evaluate_classifier,
    load_image_data,
    normalize_expr,
    set_seed,
    stratified_subset_indices,
    train_classifier,
)


def _metrics_row(method: str, y_true, y_pred) -> dict:
    return {
        "method": method,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase B baselines and per-class tables")
    ap.add_argument("--run_dir", required=True, help="Output folder of scripts/train_mosaic.py")
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--train_image", default="UC6")
    ap.add_argument("--val_image", default="UC7")
    ap.add_argument("--test_image", default="UC1")
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--test_fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs_cls", type=int, default=30)
    ap.add_argument("--early_stop_cls", type=int, default=5)
    ap.add_argument("--log1p", action="store_true", default=None,
                    help="Normalisation of measured expression (default: from <run_dir>/run_config.json)")
    ap.add_argument("--skip_ceiling", action="store_true",
                    help="Do not train the control classifier on measured expression")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    train_image, val_image, test_image = args.train_image.upper(), args.val_image.upper(), args.test_image.upper()

    log1p = args.log1p
    cfg_path = os.path.join(run_dir, "run_config.json")
    if log1p is None:
        log1p = False
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                log1p = bool(json.load(f).get("log1p_normalize", False))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_train, expr_train, _ = load_image_data(train_image, args.data_root)
    manifest_val, expr_val, _ = load_image_data(val_image, args.data_root)
    manifest_test, expr_test, _ = load_image_data(test_image, args.data_root)
    expr_norm_train, _, _ = normalize_expr(expr_train, log1p=log1p)
    expr_norm_val, _, _ = normalize_expr(expr_val, log1p=log1p)
    expr_norm_test, _, _ = normalize_expr(expr_test, log1p=log1p)

    classes = sorted(set(manifest_train["cell_type"]) | set(manifest_val["cell_type"]) | set(manifest_test["cell_type"]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    train_labels = np.array([class_to_idx[c] for c in manifest_train["cell_type"]])
    labels_val = np.array([class_to_idx[c] for c in manifest_val["cell_type"]])
    labels_test = np.array([class_to_idx[c] for c in manifest_test["cell_type"]])

    def _indices(kind, image, frac, labels):
        path = os.path.join(run_dir, f"{kind}_indices_{image}_pct{int(round(frac * 100))}.npy")
        if os.path.isfile(path):
            return np.load(path).astype(np.int64)
        print(f"[INFO] {path} not found; recomputing the split from seed {args.seed}")
        return stratified_subset_indices(labels, frac, args.seed)

    val_idx = _indices("val", val_image, args.val_fraction, labels_val)
    test_idx = _indices("test", test_image, args.test_fraction, labels_test)
    val_labels, test_labels = labels_val[val_idx], labels_test[test_idx]
    val_expr_gt, test_expr_gt = expr_norm_val[val_idx], expr_norm_test[test_idx]

    test_pred = np.load(os.path.join(run_dir, f"expr_pred_full_{test_image}.npy"))[test_idx]

    rows = []
    chance = 1.0 / n_classes
    rows.append({"method": f"Uniform chance (1/{n_classes})", "accuracy": chance, "macro_f1": chance,
                 "weighted_f1": chance, "balanced_accuracy": chance})

    majority_idx = int(np.bincount(train_labels, minlength=n_classes).argmax())
    rows.append(_metrics_row(f"Majority class ({classes[majority_idx].replace('_', ' ')})",
                             test_labels, np.full_like(test_labels, majority_idx)))

    # --- MOSAIC Phase B: re-evaluate the saved classifier with the saved training statistics ---
    norm_path = os.path.join(run_dir, f"classifier_{train_image}_norm.npz")
    train_pred_path = os.path.join(run_dir, f"expr_pred_full_{train_image}.npy")
    if os.path.isfile(norm_path):
        nz = np.load(norm_path)
        cls_mean, cls_std = nz["mean"], nz["std"]
    elif os.path.isfile(train_pred_path):
        train_pred = np.load(train_pred_path)
        cls_mean = train_pred.mean(axis=0, keepdims=True)
        cls_std = train_pred.std(axis=0, keepdims=True) + 1e-6
    else:
        raise FileNotFoundError(f"Neither {norm_path} nor {train_pred_path} found: cannot reproduce the "
                                "classifier input normalisation")

    sel_path = os.path.join(run_dir, "selected_gene_idx.npy")
    sel_idx = np.load(sel_path).astype(np.int64) if os.path.isfile(sel_path) else np.arange(test_pred.shape[1])
    test_pred_sel = test_pred[:, sel_idx]

    clf = ExprClassifier(test_pred_sel.shape[1], n_classes, [256, 128], dropout=0.5).to(device)
    clf.load_state_dict(torch.load(os.path.join(run_dir, f"classifier_{train_image}.pt"), map_location=device))
    res = evaluate_classifier(clf, test_pred_sel, test_labels, device, cls_mean, cls_std)
    mosaic_row = _metrics_row("MOSAIC Phase B (predicted expression)", test_labels, res["predictions"])

    results_csv = os.path.join(run_dir, "results.csv")
    if os.path.isfile(results_csv):
        official = pd.read_csv(results_csv).iloc[0]
        if abs(mosaic_row["accuracy"] - official["test_accuracy"]) > 0.005:
            print(f"[WARN] Re-evaluated accuracy {mosaic_row['accuracy']:.4f} differs from results.csv "
                  f"{official['test_accuracy']:.4f}; check that the run folder is complete.")
    rows.append(mosaic_row)

    # --- Ceiling control: same MLP trained on measured (Xenium) expression ---
    if not args.skip_ceiling:
        print("[INFO] Training the control classifier on measured expression...")

        class _A:
            seed = args.seed
            epochs_cls = args.epochs_cls
            early_stop_cls = args.early_stop_cls
            num_workers = 0

        gt_clf, _, gt_mean, gt_std = train_classifier(
            expr_norm_train, train_labels, val_expr_gt, val_labels, n_classes, _A(), device,
        )
        res_gt = evaluate_classifier(gt_clf, test_expr_gt, test_labels, device, gt_mean, gt_std)
        rows.append(_metrics_row("Measured Xenium expression (same MLP)", test_labels, res_gt["predictions"]))

    baselines_df = pd.DataFrame(rows)
    out_csv = os.path.join(run_dir, f"classification_baselines_{test_image}.csv")
    baselines_df.to_csv(out_csv, index=False)
    print(baselines_df.to_string(index=False))
    print(f"[OK] {out_csv}")

    # --- Per-class table from the pipeline's classification report ---
    m = pd.read_csv(os.path.join(run_dir, f"metrics_{test_image}.csv"), index_col=0)
    per_class = [{
        "cell_type": cn.replace("_", " "),
        "precision": float(m.loc[cn, "precision"]),
        "recall": float(m.loc[cn, "recall"]),
        "f1": float(m.loc[cn, "f1-score"]),
        "support": int(float(m.loc[cn, "support"])),
    } for cn in classes if cn in m.index]
    pd.DataFrame(per_class).to_csv(os.path.join(run_dir, f"classification_per_class_{test_image}.csv"), index=False)

    counts_test = pd.Series(test_labels).map(lambda i: classes[i]).value_counts()
    counts_train = pd.Series(train_labels).map(lambda i: classes[i]).value_counts()
    counts_df = pd.DataFrame({
        "cell_type": [c.replace("_", " ") for c in counts_test.index],
        f"n_train_{train_image}": [int(counts_train.get(c, 0)) for c in counts_test.index],
        f"n_test_{test_image}": counts_test.values.astype(int),
    })
    counts_df.to_csv(os.path.join(run_dir, "classification_class_counts.csv"), index=False)
    print("[OK] per-class and class-count tables written")


if __name__ == "__main__":
    main()
