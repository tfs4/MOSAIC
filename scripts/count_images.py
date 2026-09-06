#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count image files in the UC1 / UC6 / UC7 patch folders.

Looks for patches128_2_UC1, patches128_2_UC6 and patches128_2_UC7 under --data_root
(default: ./data, then the current directory). Expected PNG counts (paper):
UC1 202,528; UC6 223,781; UC7 144,702; total 571,011.

  python3 scripts/count_images.py --data_root data
  python3 scripts/count_images.py --data_root /workspace/MOSAIC/data
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

SLIDES = ("UC1", "UC6", "UC7")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}


def find_slide_dir(data_root: Path, uc: str) -> Path | None:
    candidates = [
        data_root / f"patches128_2_{uc}",
        data_root / uc,
        data_root / f"{uc}_I",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return None


def count_images(folder: Path) -> int:
    n = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if Path(name).suffix.lower() in IMAGE_EXTS:
                n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="Count images in UC1 / UC6 / UC7 folders")
    parser.add_argument(
        "--data_root",
        default=None,
        help="Parent folder of patches128_2_UC* (default: ./data if it exists, else .)",
    )
    args = parser.parse_args()

    if args.data_root:
        roots = [Path(args.data_root).resolve()]
    else:
        cwd = Path.cwd()
        roots = []
        for guess in (cwd / "data", cwd):
            if guess not in roots:
                roots.append(guess.resolve())

    print(f"{'slide':<8} {'folder':<40} {'images':>12}")
    print("-" * 64)

    total = 0
    missing = []
    found_any = False
    used_root = None

    for data_root in roots:
        rows = []
        for uc in SLIDES:
            folder = find_slide_dir(data_root, uc)
            if folder is None:
                rows.append((uc, None, None))
            else:
                n = count_images(folder)
                rows.append((uc, folder, n))
        if any(folder is not None for _, folder, _ in rows):
            used_root = data_root
            for uc, folder, n in rows:
                if folder is None:
                    missing.append(uc)
                    print(f"{uc:<8} {'(not found)':<40} {'-':>12}")
                else:
                    found_any = True
                    total += n
                    print(f"{uc:<8} {str(folder):<40} {n:>12,}")
            break

    if used_root is None:
        print("No patches128_2_UC* folder found. Pass --data_root pointing to their parent.")
        return 1

    print("-" * 64)
    print(f"{'TOTAL':<8} {str(used_root):<40} {total:>12,}")
    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        return 1
    if not found_any:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
