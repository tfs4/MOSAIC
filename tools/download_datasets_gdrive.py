#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper for our own servers: download the prepared UC1 / UC6 / UC7 patch folders from
Google Drive archives and place them under --data_root as patches128_2_<UC>/.

Requires `pip install gdown`. Each archive (.zip / .tar / .tar.gz) must contain a folder
with manifest.csv, expr.npy and the per-class PNG sub-folders.

  python tools/download_datasets_gdrive.py \
      --uc1 "https://drive.google.com/file/d/<ID>/view" \
      --uc6 "<ID>" --uc7 "<ID>" --data_root data
"""
import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def extract_drive_id(link_or_id: str) -> str:
    s = link_or_id.strip()
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", s):
        return s
    for pattern in (r"/file/d/([a-zA-Z0-9_-]+)", r"[?&]id=([a-zA-Z0-9_-]+)"):
        m = re.search(pattern, s)
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract a Drive file ID from: {s}")


def download_file(file_id_or_url: str, download_dir: Path) -> Path:
    if not shutil.which("gdown"):
        raise RuntimeError("gdown not found. Install it with: pip install gdown")
    url = f"https://drive.google.com/uc?id={extract_drive_id(file_id_or_url)}"
    print(f"[DOWNLOAD] {url}")
    before = set(download_dir.iterdir())
    try:
        subprocess.run(["gdown", "--fuzzy", url, "--output", str(download_dir) + "/"], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["gdown", url, "--output", str(download_dir) + "/"], check=True)
    created = sorted(set(download_dir.iterdir()) - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if not created:
        raise RuntimeError(f"Nothing was downloaded into {download_dir}")
    return created[0]


def extract_archive(archive_path: Path, out_dir: Path) -> None:
    print(f"[EXTRACT] {archive_path.name}")
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(out_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(out_dir)
    else:
        raise RuntimeError(f"Unsupported archive: {archive_path.name}")


def is_dataset_dir(path: Path) -> bool:
    return path.is_dir() and (path / "manifest.csv").is_file() and (path / "expr.npy").is_file()


def find_extracted_dataset(base_dir: Path, uc_name: str) -> Optional[Path]:
    dirs = [p for p in base_dir.rglob("*") if is_dataset_dir(p)]
    preferred = [p for p in dirs if uc_name.lower() in p.name.lower()]
    pool = preferred or dirs
    return sorted(pool, key=lambda x: len(str(x)))[0] if pool else None


def validate_dataset(ds_dir: Path, uc_name: str) -> None:
    n_manifest = len(pd.read_csv(ds_dir / "manifest.csv"))
    n_expr = int(np.load(ds_dir / "expr.npy", mmap_mode="r").shape[0])
    if n_manifest != n_expr:
        raise RuntimeError(f"{uc_name}: manifest ({n_manifest}) vs expr ({n_expr}) mismatch")
    print(f"[OK] {uc_name}: {n_manifest} cells (manifest == expr)")


def process_one(uc_name: str, link_or_id: str, staging_dir: Path, data_root: Path, keep_archives: bool) -> None:
    uc_stage = staging_dir / uc_name
    if uc_stage.exists():
        shutil.rmtree(uc_stage)
    uc_stage.mkdir(parents=True)

    archive = download_file(link_or_id, uc_stage)
    extract_archive(archive, uc_stage)
    if not keep_archives:
        archive.unlink(missing_ok=True)

    extracted = find_extracted_dataset(uc_stage, uc_name)
    if extracted is None:
        raise RuntimeError(f"No folder with manifest.csv + expr.npy found for {uc_name} in {uc_stage}")

    target = data_root / f"patches128_2_{uc_name}"
    if target.exists():
        backup = data_root / f"{target.name}_backup_old"
        if backup.exists():
            shutil.rmtree(backup)
        target.rename(backup)
        print(f"[INFO] Previous folder moved to {backup}")
    shutil.move(str(extracted), str(target))
    print(f"[OK] {uc_name} ready at {target}")
    validate_dataset(target, uc_name)
    shutil.rmtree(uc_stage, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Download prepared UC datasets from Google Drive")
    p.add_argument("--uc1", required=True, help="Drive link or file ID of the UC1 archive")
    p.add_argument("--uc6", required=True)
    p.add_argument("--uc7", required=True)
    p.add_argument("--data_root", default="data")
    p.add_argument("--staging_dir", default=".download_tmp")
    p.add_argument("--keep_archives", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    staging_dir = (data_root / args.staging_dir).resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)

    links: Dict[str, str] = {"UC1": args.uc1, "UC6": args.uc6, "UC7": args.uc7}
    for uc, link in links.items():
        print(f"\n=== {uc} ===")
        process_one(uc, link, staging_dir, data_root, keep_archives=args.keep_archives)
    shutil.rmtree(staging_dir, ignore_errors=True)
    print("\n[SUCCESS] Datasets ready under", data_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
