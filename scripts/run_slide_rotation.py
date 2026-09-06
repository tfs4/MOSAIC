#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rotate which slide is train (100%), val (20%) and test (10%).

Paper assignment:   train UC6 | val UC7 | test UC1
Cyclic rotations:   train UC7 | val UC1 | test UC6
                    train UC1 | val UC6 | test UC7

Each assignment is a full Phase A + Phase B run in its own folder under --out_base.
The paper split is skipped by default (already in results/mosaic_k6_seed42).

  python3 scripts/run_slide_rotation.py --data_root data --num_workers 20 \
      --out_base results/slide_rotation --seed 42

Resume from the second rotation with --start_from 1.
"""
import _repo  # noqa: F401

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from mosaic.pipeline import REPO_ROOT

TRAIN_SCRIPT = os.path.join(REPO_ROOT, "scripts", "train_mosaic.py")
METRICS_SCRIPT = os.path.join(REPO_ROOT, "scripts", "expression_metrics.py")

# (train, val, test) — cyclic permutation of the paper order
PAPER_SPLIT = ("UC6", "UC7", "UC1")
CYCLIC_SPLITS: List[Tuple[str, str, str]] = [
    ("UC6", "UC7", "UC1"),
    ("UC7", "UC1", "UC6"),
    ("UC1", "UC6", "UC7"),
]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run_dir(out_base: str, train: str, val: str, test: str, seed: int) -> str:
    return os.path.join(out_base, f"train{train}_val{val}_test{test}_seed{seed}")


def _run_cmd(cmd: List[str], log_path: str) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"[{_now()}] CMD: {' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        proc.wait()
        logf.write(f"\n[{_now()}] EXIT_CODE={proc.returncode}\n")
        return int(proc.returncode)


def _read_results_csv(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rotate train / val / test slides (same fractions)")
    p.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data"))
    p.add_argument("--python_exe", default=sys.executable)
    p.add_argument("--out_base", default=os.path.join(REPO_ROOT, "results", "slide_rotation"),
                   help="Parent folder; one sub-folder per assignment")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k_neighbors", type=int, default=6)
    p.add_argument("--aggregation_mode", default="attention")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--test_fraction", type=float, default=0.1)
    p.add_argument("--include_paper_split", action="store_true",
                   help="Also run train UC6 / val UC7 / test UC1")
    p.add_argument("--start_from", type=int, default=0, metavar="N",
                   help="Skip the first N assignments (0-based) to resume")
    p.add_argument("--skip_expression_metrics", action="store_true")
    return p.parse_args()


def _assignments(include_paper: bool) -> List[Tuple[str, str, str]]:
    splits = list(CYCLIC_SPLITS)
    if not include_paper:
        splits = [s for s in splits if s != PAPER_SPLIT]
    return splits


def main() -> int:
    args = parse_args()
    if not (0 < args.val_fraction < 1) or not (0 < args.test_fraction < 1):
        raise ValueError("--val_fraction and --test_fraction must be in (0, 1)")

    out_base = os.path.abspath(args.out_base)
    logs_dir = os.path.join(out_base, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    splits = _assignments(args.include_paper_split)[args.start_from:]
    if not splits:
        print("[WARN] Nothing to run (check --start_from / --include_paper_split).")
        return 1

    print(f"[INFO] out_base={out_base}")
    print(f"[INFO] fractions: train 100% | val {args.val_fraction:.0%} | test {args.test_fraction:.0%}")
    print("[INFO] assignments:")
    for train, val, test in splits:
        print(f"       train {train}  val {val}  test {test}  ->  {_run_dir(out_base, train, val, test, args.seed)}")

    summary_rows: List[Dict] = []
    for train, val, test in splits:
        out_dir = _run_dir(out_base, train, val, test, args.seed)
        log_path = os.path.join(logs_dir, f"train{train}_val{val}_test{test}_seed{args.seed}.log")
        cmd = [
            args.python_exe, TRAIN_SCRIPT,
            "--data_root", args.data_root,
            "--train_image", train,
            "--val_image", val,
            "--test_image", test,
            "--val_fraction", str(args.val_fraction),
            "--test_fraction", str(args.test_fraction),
            "--k_neighbors", str(args.k_neighbors),
            "--aggregation_mode", args.aggregation_mode,
            "--seed", str(args.seed),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--out_dir", out_dir,
        ]
        print(f"\n{'=' * 64}")
        print(f" ROTATION  train={train} (100%)  val={val} ({args.val_fraction:.0%})  "
              f"test={test} ({args.test_fraction:.0%})")
        print(f" out_dir = {out_dir}")
        print(f"{'=' * 64}\n")
        exit_code = _run_cmd(cmd, log_path)

        if exit_code == 0 and not args.skip_expression_metrics:
            _run_cmd(
                [args.python_exe, METRICS_SCRIPT, "--run_dir", out_dir, "--data_root", args.data_root],
                log_path,
            )

        metrics = _read_results_csv(os.path.join(out_dir, "results.csv")) or {}
        row = {
            "train_image": train,
            "val_image": val,
            "test_image": test,
            "seed": args.seed,
            "out_dir": out_dir,
            "exit_code": exit_code,
            "status": "ok" if exit_code == 0 and metrics else "fail",
            "expr_test_pearson": metrics.get("expr_test_pearson", ""),
            "test_accuracy": metrics.get("test_accuracy", ""),
            "test_f1_macro": metrics.get("test_f1_macro", ""),
            "test_f1_weighted": metrics.get("test_f1_weighted", ""),
        }
        summary_rows.append(row)

    summary_path = os.path.join(out_base, "rotation_results.csv")
    keys = list(summary_rows[0].keys())
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n[OK] Summary: {summary_path}")
    print(f"[OK] Logs: {logs_dir}")
    return 0 if all(r["status"] == "ok" for r in summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
