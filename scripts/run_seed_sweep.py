#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reviewer 2 item 3: robustness over random seeds (same paper protocol).

Each seed redraws the stratified 20% UC7 / 10% UC1 split AND retrains Phase A+B.
Do not pass --indices_dir: that would freeze the paper split.

Default seeds: 0, 20 (seed 42 is the paper run, already done). Each run is stored
under --out_base, not results/mosaic_k6_seed42.

  python3 scripts/run_seed_sweep.py --data_root data --num_workers 20 \
      --seeds 0,20 --out_base results/seed_sweep

Skip a seed that already has results.csv with --skip_existing.
Resume with --start_from 1 (skip the first seed in the list).
"""
import _repo  # noqa: F401

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional

from mosaic.pipeline import REPO_ROOT

TRAIN_SCRIPT = os.path.join(REPO_ROOT, "scripts", "train_mosaic.py")
METRICS_SCRIPT = os.path.join(REPO_ROOT, "scripts", "expression_metrics.py")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_int_list(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("--seeds is empty")
    return vals


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
    p = argparse.ArgumentParser(description="Paper protocol repeated over several seeds")
    p.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data"))
    p.add_argument("--python_exe", default=sys.executable)
    p.add_argument("--out_base", default=os.path.join(REPO_ROOT, "results", "seed_sweep"),
                   help="Parent folder; one sub-folder per seed")
    p.add_argument("--seeds", default="0,20", help="Comma-separated integer seeds (42 already run)")
    p.add_argument("--train_image", default="UC6")
    p.add_argument("--val_image", default="UC7")
    p.add_argument("--test_image", default="UC1")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--test_fraction", type=float, default=0.1)
    p.add_argument("--k_neighbors", type=int, default=6)
    p.add_argument("--aggregation_mode", default="attention")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip a seed if <out_dir>/results.csv already exists")
    p.add_argument("--start_from", type=int, default=0, metavar="N",
                   help="Skip the first N seeds in --seeds (0-based)")
    p.add_argument("--skip_expression_metrics", action="store_true")
    return p.parse_args()


def _write_summary(rows: List[Dict], path: str) -> None:
    keys = [
        "seed", "status", "exit_code", "out_dir",
        "expr_val_pearson", "expr_test_pearson",
        "test_accuracy", "test_f1_macro", "test_f1_weighted", "cls_val_f1",
        "train_size", "val_size", "test_size",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main() -> int:
    args = parse_args()
    out_base = os.path.abspath(args.out_base)
    logs_dir = os.path.join(out_base, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    seeds = _parse_int_list(args.seeds)[args.start_from:]
    print(f"[INFO] out_base={out_base}")
    print(f"[INFO] protocol: train {args.train_image} 100% | val {args.val_image} "
          f"{args.val_fraction:.0%} | test {args.test_image} {args.test_fraction:.0%}")
    print(f"[INFO] seeds={seeds}  (each seed redraws val/test cells; no --indices_dir)")

    rows: List[Dict] = []
    for seed in seeds:
        out_dir = os.path.join(out_base, f"mosaic_k6_seed{seed}")
        results_csv = os.path.join(out_dir, "results.csv")
        log_path = os.path.join(logs_dir, f"seed{seed}.log")

        if args.skip_existing and os.path.isfile(results_csv):
            print(f"\n[SKIP] seed {seed}: found {results_csv}")
            metrics = _read_results_csv(results_csv) or {}
            row = dict(metrics)
            row.update({"seed": seed, "status": "skipped_existing", "exit_code": 0, "out_dir": out_dir})
            rows.append(row)
            continue

        cmd = [
            args.python_exe, TRAIN_SCRIPT,
            "--data_root", args.data_root,
            "--train_image", args.train_image,
            "--val_image", args.val_image,
            "--test_image", args.test_image,
            "--val_fraction", str(args.val_fraction),
            "--test_fraction", str(args.test_fraction),
            "--k_neighbors", str(args.k_neighbors),
            "--aggregation_mode", args.aggregation_mode,
            "--seed", str(seed),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--out_dir", out_dir,
        ]
        print(f"\n{'=' * 64}")
        print(f" SEED {seed}  out_dir={out_dir}")
        print(f"{'=' * 64}\n")
        exit_code = _run_cmd(cmd, log_path)

        if exit_code == 0 and not args.skip_expression_metrics:
            _run_cmd(
                [args.python_exe, METRICS_SCRIPT, "--run_dir", out_dir, "--data_root", args.data_root],
                log_path,
            )

        metrics = _read_results_csv(results_csv) or {}
        row = dict(metrics)
        row.update({
            "seed": seed,
            "out_dir": out_dir,
            "exit_code": exit_code,
            "status": "ok" if exit_code == 0 and metrics else "fail",
        })
        rows.append(row)

        summary_path = os.path.join(out_base, "seed_sweep_results.csv")
        _write_summary(rows, summary_path)
        print(f"[OK] summary updated: {summary_path}")

    summary_path = os.path.join(out_base, "seed_sweep_results.csv")
    _write_summary(rows, summary_path)
    print(f"\n[OK] Summary: {summary_path}")
    print(f"[OK] Logs: {logs_dir}")
    ok = all(r.get("status") in ("ok", "skipped_existing") for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
