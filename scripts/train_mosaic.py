#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the full MOSAIC experiment (Phase A + Phase B) with the slide-disjoint protocol.

Paper configuration (UC6 train | UC7 20% val | UC1 10% test, K=6, attention, seed 42):

  python scripts/train_mosaic.py --k_neighbors 6 --aggregation_mode attention \
      --seed 42 --log1p_normalize --out_dir results/mosaic_k6_seed42

All options: python scripts/train_mosaic.py --help
"""
import _repo  # noqa: F401

from mosaic.pipeline import main

if __name__ == "__main__":
    main()
