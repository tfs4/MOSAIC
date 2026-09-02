#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sanity-check the prepared UC1 / UC6 / UC7 folders before training.

Default checks
  - manifest.csv and expr.npy exist and have the same number of rows
  - the split sizes implied by the paper protocol are reproduced:
      UC6 100% train = 223,781 cells
      UC7 20%  val   =  28,941 cells (stratified, seed 42)
      UC1 10%  test  =  20,253 cells (stratified, seed 42)
  - optionally (--check_files) every patch referenced by the manifest exists on disk

Strict mode (--expected_profile profile.json [--strict])
  - additionally compares SHA-256 hashes and metadata of manifest.csv / expr.npy with a
    reference profile produced with --save_profile on the machine that ran the paper.

  python scripts/verify_dataset.py --data_root data
  python scripts/verify_dataset.py --data_root data --save_profile results/dataset_profile.json
  python scripts/verify_dataset.py --data_root data --expected_profile results/dataset_profile.json --strict
"""
import _repo  # noqa: F401

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split

from mosaic.pipeline import DEFAULT_DATA_ROOT

# Values reported in the paper (Extended Data Table 1) for the primary split, seed 42.
PAPER_SPLIT = {
    "UC6": {"role": "train", "fraction": 1.0, "expected": 223781},
    "UC7": {"role": "val", "fraction": 0.2, "expected": 28941},
    "UC1": {"role": "test", "fraction": 0.1, "expected": 20253},
}


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_image_path(dataset_dir: Path, raw_path: str) -> Path:
    p = Path(str(raw_path).replace("\\", "/"))
    if p.is_absolute():
        return p
    parts = p.parts
    # manifests store '<folder>/<type>/<id>.png'; the folder may have been renamed
    if len(parts) > 1 and parts[0].startswith("patches"):
        return dataset_dir.joinpath(*parts[1:])
    return dataset_dir / p


def _collect_uc_info(data_root: Path, uc: str, check_files: bool, seed: int) -> Dict:
    ds_dir = data_root / f"patches128_2_{uc}"
    manifest_path = ds_dir / "manifest.csv"
    expr_path = ds_dir / "expr.npy"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{uc}: manifest not found: {manifest_path}")
    if not expr_path.is_file():
        raise FileNotFoundError(f"{uc}: expr.npy not found: {expr_path}")

    expr = np.load(expr_path, mmap_mode="r")
    labels: List[str] = []
    missing_images = 0
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        for row in reader:
            labels.append(str(row.get("cell_type", "")))
            if check_files and not _resolve_image_path(ds_dir, row.get("path", "")).is_file():
                missing_images += 1

    frac = PAPER_SPLIT[uc]["fraction"]
    if frac < 1.0:
        _, idx = train_test_split(np.arange(len(labels)), test_size=frac, random_state=seed, stratify=labels)
        split_size = int(len(idx))
    else:
        split_size = len(labels)

    return {
        "dataset_dir": str(ds_dir),
        "manifest_rows": len(labels),
        "expr_rows": int(expr.shape[0]),
        "expr_genes": int(expr.shape[1]) if expr.ndim > 1 else 1,
        "manifest_columns": columns,
        "n_cell_types": len(set(labels)),
        "split_role": PAPER_SPLIT[uc]["role"],
        "split_fraction": frac,
        "split_size": split_size,
        "manifest_sha256": _sha256_file(manifest_path),
        "expr_sha256": _sha256_file(expr_path),
        "missing_images": missing_images,
    }


def _compare_profile(current: Dict, expected: Dict) -> List[str]:
    mismatches: List[str] = []
    for uc, exp in expected.get("datasets", {}).items():
        cur = current["datasets"].get(uc)
        if cur is None:
            mismatches.append(f"{uc}: missing in the current dataset.")
            continue
        for k in ["manifest_rows", "expr_rows", "expr_genes", "manifest_columns", "manifest_sha256", "expr_sha256"]:
            if cur.get(k) != exp.get(k):
                mismatches.append(f"{uc}: mismatch in '{k}' (current={cur.get(k)} expected={exp.get(k)})")
    return mismatches


def main() -> int:
    p = argparse.ArgumentParser(description="Verify the prepared UC datasets")
    p.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check_files", action="store_true", help="Check that every patch in the manifest exists")
    p.add_argument("--expected_profile", default=None, help="Reference JSON with hashes")
    p.add_argument("--save_profile", default=None, help="Write the current profile to this JSON")
    p.add_argument("--strict", action="store_true", help="Fail unless the dataset matches --expected_profile exactly")
    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    current = {"data_root": str(data_root), "seed": args.seed, "datasets": {}}
    issues: List[str] = []

    print(f"[INFO] Checking datasets in {data_root}")
    for uc in ["UC6", "UC7", "UC1"]:
        info = _collect_uc_info(data_root, uc, args.check_files, args.seed)
        current["datasets"][uc] = info
        print(f"\n[{uc}]  ({info['split_role']}, fraction {info['split_fraction']})")
        print(f"  manifest_rows: {info['manifest_rows']}")
        print(f"  expr_rows:     {info['expr_rows']}")
        print(f"  expr_genes:    {info['expr_genes']}")
        print(f"  cell types:    {info['n_cell_types']}")
        print(f"  split size:    {info['split_size']} (paper: {PAPER_SPLIT[uc]['expected']})")
        if args.check_files:
            print(f"  missing patch files: {info['missing_images']}")
        print(f"  manifest_sha256: {info['manifest_sha256'][:16]}...")
        print(f"  expr_sha256:     {info['expr_sha256'][:16]}...")

        if info["manifest_rows"] != info["expr_rows"]:
            issues.append(f"{uc}: manifest_rows ({info['manifest_rows']}) != expr_rows ({info['expr_rows']})")
        if info["split_size"] != PAPER_SPLIT[uc]["expected"]:
            issues.append(f"{uc}: {info['split_role']} split has {info['split_size']} cells, paper reports {PAPER_SPLIT[uc]['expected']}")
        if args.check_files and info["missing_images"] > 0:
            issues.append(f"{uc}: {info['missing_images']} patch files missing on disk")

    if args.save_profile:
        out = Path(args.save_profile)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"\n[OK] Profile written to {out.resolve()}")

    if args.expected_profile:
        expected = json.loads(Path(args.expected_profile).read_text(encoding="utf-8"))
        mismatches = _compare_profile(current, expected)
        if mismatches:
            issues.extend(mismatches)
            print("\n[WARN] Differences vs expected profile:")
            for m in mismatches:
                print(f"  - {m}")
        else:
            print("\n[OK] Dataset matches the expected profile (hashes and metadata).")

    if args.strict and not args.expected_profile:
        issues.append("--strict requires --expected_profile")

    if issues:
        print("\n[FAIL] Dataset does NOT match the reference:")
        for it in issues:
            print(f"  - {it}")
        return 1
    print("\n[SUCCESS] Dataset consistent with the paper protocol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
