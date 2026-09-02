#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bar charts comparing the configurations of an ablation run.

Input: the --ablation_out_dir of scripts/run_ablation.py, i.e. either a consolidated
ablation_results.csv or sub-folders (k0_agg_n_a, k6_agg_attention, ...) with results.csv.

  python scripts/plot_ablation.py --input_dir results/ablation_k --output_dir results/ablation_k/figures
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.0

BLUES = ["#7EB6D9", "#4A90C2", "#2E5F8A", "#1A3A5C"]
AGG_ORDER = {"n/a": 0, "attention": 1, "mean": 2, "max": 3}


def infer_config_from_folder(folder_name: str):
    """(k, aggregation label) from folder names such as k6_agg_attention or k0_agg_n_a."""
    m = re.match(r"(?:sem_grafo_|com_grafo_)?k(\d+)_agg_(.+)", folder_name.strip(), re.IGNORECASE)
    if m:
        k = int(m.group(1))
        agg = m.group(2).replace("_", " ").lower()
        return k, ("n/a" if agg in ("n/a", "na", "n a") else agg)
    m = re.match(r"k_?(\d+)$", folder_name.strip(), re.IGNORECASE)
    if m:
        k = int(m.group(1))
        return k, ("n/a" if k == 0 else "attention")
    return None, None


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["k_neighbors"] = df["k_neighbors"].astype(int)
    if "aggregation_mode" not in df.columns:
        df["aggregation_mode"] = "attention"
    df["aggregation_mode"] = df["aggregation_mode"].fillna("n/a").astype(str).str.strip().str.lower()
    df.loc[df["k_neighbors"] == 0, "aggregation_mode"] = "n/a"
    df["_order"] = df["aggregation_mode"].map(AGG_ORDER).fillna(1)
    df = df.sort_values(["k_neighbors", "_order"]).drop(columns="_order").reset_index(drop=True)
    return df


def load_ablation_df(input_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(input_dir, "ablation_results.csv")
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
        if "k_neighbors" in df.columns:
            return _normalise(df)
    rows = []
    for name in sorted(os.listdir(input_dir)):
        res_path = os.path.join(input_dir, name, "results.csv")
        if not os.path.isfile(res_path):
            continue
        df_one = pd.read_csv(res_path)
        if df_one.empty:
            continue
        row = df_one.iloc[0].to_dict()
        if "k_neighbors" not in row:
            k, agg = infer_config_from_folder(name)
            if k is None:
                continue
            row["k_neighbors"], row["aggregation_mode"] = k, agg
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No results.csv found under {input_dir}")
    return _normalise(pd.DataFrame(rows))


def config_labels(df: pd.DataFrame) -> list:
    return [f"K={int(r.k_neighbors)}\n{r.aggregation_mode}" for r in df.itertuples()]


def _col(df, name):
    return df[name].values.astype(float) if name in df.columns else np.full(len(df), np.nan)


def _add_bar_labels(ax, bars, fmt=".2f"):
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:{fmt}}", ha="center", va="bottom", fontsize=9)


def _save(fig, output_dir, stem):
    for ext, dpi in ((".pdf", 150), (".png", 300)):
        path = os.path.join(output_dir, f"{stem}{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  {path}")
    plt.close(fig)


def plot_publication(df: pd.DataFrame, output_dir: str) -> None:
    """PCC (test), accuracy, macro-F1 and weighted-F1 side by side."""
    labels = config_labels(df)
    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
    series = [("PCC (test)", _col(df, "expr_test_pearson")), ("Accuracy", _col(df, "test_accuracy")),
              ("F1 macro", _col(df, "test_f1_macro")), ("F1 weighted", _col(df, "test_f1_weighted"))]
    for i, (name, vals) in enumerate(series):
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=name, color=BLUES[i], edgecolor="white", linewidth=0.5)
        _add_bar_labels(ax, bars)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save(fig, output_dir, "ablation_publication")


def plot_pearson(df: pd.DataFrame, output_dir: str) -> None:
    labels = config_labels(df)
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    b1 = ax.bar(x - w / 2, _col(df, "expr_val_pearson"), w, label="Val PCC", color=BLUES[0], edgecolor="white")
    b2 = ax.bar(x + w / 2, _col(df, "expr_test_pearson"), w, label="Test PCC", color=BLUES[2], edgecolor="white")
    _add_bar_labels(ax, b1)
    _add_bar_labels(ax, b2)
    ax.set_ylabel("Mean per-gene Pearson correlation")
    ax.set_title("Ablation: expression correlation (validation and test)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save(fig, output_dir, "ablation_pearson")


def plot_classification(df: pd.DataFrame, output_dir: str) -> None:
    labels = config_labels(df)
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    for i, (name, col) in enumerate([("Accuracy", "test_accuracy"), ("F1 macro", "test_f1_macro"),
                                     ("F1 weighted", "test_f1_weighted")]):
        bars = ax.bar(x + (i - 1) * w, _col(df, col), w, label=name, color=BLUES[i + 1], edgecolor="white")
        _add_bar_labels(ax, bars)
    ax.set_ylabel("Score")
    ax.set_title("Ablation: lineage classification on the test subset")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save(fig, output_dir, "ablation_classification")


def main():
    p = argparse.ArgumentParser(description="Ablation comparison charts")
    p.add_argument("--input_dir", required=True, help="Folder with ablation_results.csv or per-config sub-folders")
    p.add_argument("--output_dir", default=None, help="Default: <input_dir>/figures")
    args = p.parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(input_dir, "figures"))
    os.makedirs(output_dir, exist_ok=True)
    df = load_ablation_df(input_dir)
    print(f"Input:  {input_dir}\nOutput: {output_dir}\nConfigurations: {len(df)}")
    plot_publication(df, output_dir)
    plot_pearson(df, output_dir)
    plot_classification(df, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
