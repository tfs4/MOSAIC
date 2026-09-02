#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neighbourhood ablation: run the MOSAIC pipeline for several K / aggregation settings
on the *same* validation and test cells and consolidate the metrics in one CSV.

Default grid: K = 0 (backbone only), 4, 6, 8, 12 with attention aggregation.

  python scripts/run_ablation.py --ks 0,4,6,8,12 --aggregations attention \
      --ablation_out_dir results/ablation_k --seed 42 --log1p_normalize --num_workers 8

  # aggregation ablation at fixed K
  python scripts/run_ablation.py --ks 6 --aggregations attention,mean,max \
      --ablation_out_dir results/ablation_agg

Resuming: pass --start_from N together with --indices_dir <first run folder> so every
configuration is evaluated on exactly the same cells.
"""
import _repo  # noqa: F401

import argparse
import copy
import os

import pandas as pd

from mosaic.pipeline import REPO_ROOT, build_argparser, run


def _parse_int_list(raw: str):
    return [int(x) for x in raw.split(",") if x.strip()]


def _parse_str_list(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_configs(ks, aggregations):
    configs = []
    for k in ks:
        if k == 0:
            configs.append({"k_neighbors": 0, "aggregation_mode": "attention", "label": "n_a"})
            continue
        for agg in aggregations:
            configs.append({"k_neighbors": k, "aggregation_mode": agg, "label": agg})
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="MOSAIC ablation over K and neighbour aggregation",
        parents=[build_argparser()], conflict_handler="resolve",
    )
    parser.add_argument("--ablation_out_dir", default=os.path.join(REPO_ROOT, "results", "ablation"),
                        help="Folder for ablation_results.csv and one sub-folder per configuration")
    parser.add_argument("--ablation_csv", default="ablation_results.csv")
    parser.add_argument("--ks", default="0,4,6,8,12", help="Comma-separated K values (0 = backbone only)")
    parser.add_argument("--aggregations", default="attention",
                        help="Comma-separated aggregation modes for K > 0: attention,mean,max")
    parser.add_argument("--start_from", type=int, default=0, metavar="N",
                        help="Skip the first N configurations (requires --indices_dir)")
    args = parser.parse_args()

    if not (0 < args.val_fraction < 1) or not (0 < args.test_fraction < 1):
        raise ValueError("--val_fraction and --test_fraction must be in (0, 1)")

    configs = build_configs(_parse_int_list(args.ks), _parse_str_list(args.aggregations))
    start_from = max(0, args.start_from)
    if start_from > 0:
        if not args.indices_dir or not os.path.isdir(args.indices_dir):
            raise ValueError(
                "--start_from > 0 requires --indices_dir pointing to the first run folder "
                "(e.g. results/ablation/k0_agg_n_a) so that all runs share the same val/test cells."
            )
        configs = configs[start_from:]
        print(f"[INFO] Resuming from configuration {start_from + 1}: {len(configs)} left")

    os.makedirs(args.ablation_out_dir, exist_ok=True)
    all_rows = []
    first_run_dir = None

    for i, cfg in enumerate(configs):
        k = cfg["k_neighbors"]
        run_dir = os.path.abspath(os.path.join(args.ablation_out_dir, f"k{k}_agg_{cfg['label']}"))

        run_args = copy.copy(args)
        run_args.out_dir = run_dir
        run_args.k_neighbors = k
        run_args.aggregation_mode = cfg["aggregation_mode"]
        if not args.indices_dir and first_run_dir is not None:
            run_args.indices_dir = first_run_dir

        print(f"\n{'=' * 60}")
        print(f" Ablation {start_from + i + 1}/{start_from + len(configs)}: K={k}, aggregation={cfg['label']}")
        print(f" out_dir = {run_dir}")
        if run_args.indices_dir:
            print(f" indices_dir = {run_args.indices_dir} (same val/test cells)")
        print(f"{'=' * 60}\n")

        run(run_args)
        if first_run_dir is None:
            first_run_dir = run_dir

        results_path = os.path.join(run_dir, "results.csv")
        if not os.path.isfile(results_path):
            print(f"[WARN] {results_path} not found; skipping in the consolidated CSV.")
            continue
        df = pd.read_csv(results_path)
        df.insert(0, "config", f"k{k}_agg_{cfg['label']}")
        all_rows.append(df)

    if not all_rows:
        print("[WARN] No results.csv found; consolidated CSV not written.")
        return

    out_csv = os.path.join(args.ablation_out_dir, args.ablation_csv)
    pd.concat(all_rows, ignore_index=True).to_csv(out_csv, index=False)
    print(f"\n[OK] Ablation results saved to {out_csv}")


if __name__ == "__main__":
    main()
