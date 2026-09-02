#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPERIMENTAL - not used for the results reported in the paper.

Same grid as scripts/run_ablation.py but with the patch pre-processing variant.

  python experimental/run_ablation_preprocessing.py --ks 0,4,6 \
      --ablation_out_dir results/experimental/ablation_preproc
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import patch_preprocessing  # noqa: E402
import run_ablation  # noqa: E402

if __name__ == "__main__":
    patch_preprocessing.enable_preproc()
    run_ablation.main()
