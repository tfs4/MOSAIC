#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unattended sweep over K (and seeds) for a server run.

1) Runs the pipeline for each K in --ks with --base_seed.
2) Picks the best K by test macro-F1 (tie-break: accuracy).
3) Re-runs the best K with each seed in --extra_seeds.
4) Writes summary_results.csv / summary_results.txt in --out_base.

Every run is launched as a separate process (scripts/train_mosaic.py) and its console
output is stored in <out_base>/logs/.

  python scripts/run_k_sweep.py --ks 6,8,12 --base_seed 42 --extra_seeds 43,44 \
      --out_base results/k_sweep --batch_size 64 --num_workers 8 --log1p_normalize
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


def _parse_int_list(raw: str) -> List[int]:
    vals = [int(x) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty list.")
    return vals


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run_cmd(cmd: List[str], log_path: str) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
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


def _read_results_csv(path: str) -> Optional[Dict[str, float]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out = {}
    for k, v in rows[0].items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def _build_cmd(args, k: int, seed: int, out_dir: str) -> List[str]:
    cmd = [
        args.python_exe, TRAIN_SCRIPT,
        "--data_root", args.data_root,
        "--k_neighbors", str(k),
        "--aggregation_mode", args.aggregation_mode,
        "--seed", str(seed),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--lr_expr", str(args.lr_expr),
        "--early_stop_expr", str(args.early_stop_expr),
        "--epochs_expr", str(args.epochs_expr),
        "--epochs_cls", str(args.epochs_cls),
        "--out_dir", out_dir,
    ]
    if args.use_graph:
        cmd.append("--use_graph")
    if args.log1p_normalize:
        cmd.append("--log1p_normalize")
    if args.no_save_full_predictions:
        cmd.append("--no_save_full_predictions")
    return cmd


def _choose_best(rows: List[Dict]) -> Dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        raise RuntimeError("No successful run to choose the best K from.")
    ok.sort(key=lambda r: (float(r.get("test_f1_macro", -1.0)), float(r.get("test_accuracy", -1.0))), reverse=True)
    return ok[0]


def _write_summary(rows: List[Dict], out_csv: str, out_txt: str):
    keys = ["phase", "k_neighbors", "seed", "status", "out_dir", "results_csv", "test_accuracy",
            "test_f1_macro", "test_f1_weighted", "expr_val_pearson", "expr_test_pearson", "cls_val_f1", "exit_code"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    lines = [f"Summary generated at: {_now()}", ""]
    for r in rows:
        lines.append(
            f"- phase={r.get('phase')} | k={r.get('k_neighbors')} | seed={r.get('seed')} "
            f"| status={r.get('status')} | acc={r.get('test_accuracy')} | f1m={r.get('test_f1_macro')}"
        )
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Sequential K / seed sweep")
    p.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data"))
    p.add_argument("--python_exe", default=sys.executable)
    p.add_argument("--out_base", default=os.path.join(REPO_ROOT, "results", "k_sweep"))
    p.add_argument("--ks", default="6,8,12")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--extra_seeds", default="43", help="Comma-separated; use '' to skip")
    p.add_argument("--aggregation_mode", default="attention")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr_expr", type=float, default=2e-4)
    p.add_argument("--early_stop_expr", type=int, default=6)
    p.add_argument("--epochs_expr", type=int, default=30)
    p.add_argument("--epochs_cls", type=int, default=30)
    p.add_argument("--use_graph", action="store_true")
    p.add_argument("--log1p_normalize", action="store_true")
    p.add_argument("--no_save_full_predictions", action="store_true")
    return p.parse_args()


def _one_run(args, phase: str, k: int, seed: int, out_base: str, logs_dir: str) -> Dict:
    out_dir = os.path.join(out_base, f"k{k}_seed{seed}")
    results_csv = os.path.join(out_dir, "results.csv")
    log_path = os.path.join(logs_dir, f"k{k}_seed{seed}.log")
    print(f"\n[RUN] phase={phase} | k={k} | seed={seed}")
    exit_code = _run_cmd(_build_cmd(args, k=k, seed=seed, out_dir=out_dir), log_path)
    metrics = _read_results_csv(results_csv)
    row = dict(metrics or {})
    row.update({"phase": phase, "k_neighbors": k, "seed": seed,
                "status": "ok" if exit_code == 0 and metrics else "fail",
                "out_dir": out_dir, "results_csv": results_csv, "exit_code": exit_code})
    return row


def main() -> int:
    args = parse_args()
    out_base = os.path.abspath(args.out_base)
    logs_dir = os.path.join(out_base, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    ks = _parse_int_list(args.ks)
    extra_seeds = _parse_int_list(args.extra_seeds) if args.extra_seeds.strip() else []
    print(f"[INFO] out_base={out_base}")
    print(f"[INFO] ks={ks} | base_seed={args.base_seed} | extra_seeds={extra_seeds}")

    rows: List[Dict] = [_one_run(args, "sweep_k", k, args.base_seed, out_base, logs_dir) for k in ks]

    best = _choose_best(rows)
    best_k = int(best["k_neighbors"])
    print(f"\n[INFO] Best K: {best_k} (macro-F1={best.get('test_f1_macro')}, acc={best.get('test_accuracy')})")

    for s in extra_seeds:
        rows.append(_one_run(args, "best_k_extra_seed", best_k, s, out_base, logs_dir))

    summary_csv = os.path.join(out_base, "summary_results.csv")
    summary_txt = os.path.join(out_base, "summary_results.txt")
    _write_summary(rows, summary_csv, summary_txt)
    print(f"\n[OK] Summary CSV: {summary_csv}")
    print(f"[OK] Summary TXT: {summary_txt}")
    print(f"[OK] Logs: {logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
