#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPERIMENTAL - not used for the results reported in the paper.

Variant of the MOSAIC pipeline that applies light image pre-processing to every patch
(target and neighbours) before the network, keeping training/evaluation unchanged:
  - robust per-channel percentile normalisation
  - colour normalisation towards a fixed target (light stain-normalisation proxy)
  - QC: very white / low-detail patches receive autocontrast
  - mild contrast gain

The pre-processing is injected by replacing the dataset classes of mosaic.pipeline at
run time; the core module is not modified.

  python experimental/patch_preprocessing.py --k_neighbors 6 --seed 42 \
      --out_dir results/experimental/preproc_k6
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageEnhance, ImageOps  # noqa: E402

import mosaic.pipeline as base  # noqa: E402

IMAGE_PATHS = base.IMAGE_PATHS

DEFAULT_PREPROC_CFG = {
    "clip_low_pct": 1.0,
    "clip_high_pct": 99.0,
    "target_mean": (0.70, 0.58, 0.70),
    "target_std": (0.17, 0.16, 0.18),
    "white_threshold": 0.96,
    "max_white_ratio": 0.65,
    "min_sharpness": 0.0012,
    "autocontrast_cutoff": 1,
    "contrast_gain": 1.08,
}


def _to_rgb_float(img: Image.Image) -> np.ndarray:
    return np.clip(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0, 0.0, 1.0)


def _estimate_qc(arr: np.ndarray, cfg: dict):
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    white_ratio = float((gray > float(cfg["white_threshold"])).mean())
    sharpness = float(np.diff(gray, axis=1).var() + np.diff(gray, axis=0).var())
    return white_ratio, sharpness


def _robust_channel_normalize(arr: np.ndarray, cfg: dict) -> np.ndarray:
    lo = np.percentile(arr, float(cfg["clip_low_pct"]), axis=(0, 1), keepdims=True)
    hi = np.percentile(arr, float(cfg["clip_high_pct"]), axis=(0, 1), keepdims=True)
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    std = arr.std(axis=(0, 1), keepdims=True) + 1e-6
    target_mean = np.array(cfg["target_mean"], dtype=np.float32).reshape(1, 1, 3)
    target_std = np.array(cfg["target_std"], dtype=np.float32).reshape(1, 1, 3)
    return np.clip(((arr - mean) / std) * target_std + target_mean, 0.0, 1.0)


def preprocess_patch_image(img: Image.Image, cfg: dict) -> Image.Image:
    arr = _to_rgb_float(img)
    white_ratio, sharpness = _estimate_qc(arr, cfg)
    arr = _robust_channel_normalize(arr, cfg)
    out = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
    if white_ratio > float(cfg["max_white_ratio"]) or sharpness < float(cfg["min_sharpness"]):
        out = ImageOps.autocontrast(out, cutoff=int(cfg["autocontrast_cutoff"]))
    return ImageEnhance.Contrast(out).enhance(float(cfg["contrast_gain"]))


def _load_patch(path: str, cfg: dict) -> Image.Image:
    return preprocess_patch_image(Image.open(path).convert("RGB"), cfg)


def _neighbour_tensors(self, rec, img_t, neigh_ids):
    neigh_imgs, pos_enc = [], []
    for n_idx in neigh_ids:
        n_rec = self.records[int(n_idx)]
        neigh_imgs.append(self.tx(_load_patch(n_rec["path"], self.preproc_cfg)))
        if self.use_graph:
            dx = n_rec["cx"] - rec["cx"]
            dy = n_rec["cy"] - rec["cy"]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-6
            ang = np.arctan2(dy, dx)
            pos_enc.append([dx / self.pos_scale, dy / self.pos_scale, dist / self.pos_scale, np.sin(ang), np.cos(ang)])
    pos_enc = torch.tensor(pos_enc, dtype=torch.float32) if self.use_graph else torch.zeros(len(neigh_imgs), 5)
    if len(neigh_imgs) == 0:
        return img_t.unsqueeze(0), torch.zeros(1, 5, dtype=torch.float32)
    return torch.stack(neigh_imgs, dim=0), pos_enc


class Img2ExprDatasetPreproc(base.Img2ExprDataset):
    preproc_cfg = DEFAULT_PREPROC_CFG

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_t = self.tx(_load_patch(rec["path"], self.preproc_cfg))
        neigh_imgs, pos_enc = _neighbour_tensors(self, rec, img_t, self.neighbor_idx[idx])
        expr_t = torch.tensor(self.expr[idx], dtype=torch.float32)
        return img_t, neigh_imgs, pos_enc, expr_t


class Img2ExprInferDatasetPreproc(base.Img2ExprInferDataset):
    preproc_cfg = DEFAULT_PREPROC_CFG

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_t = self.tx(_load_patch(rec["path"], self.preproc_cfg))
        neigh_imgs, pos_enc = _neighbour_tensors(self, rec, img_t, self.neighbor_idx[idx])
        return img_t, neigh_imgs, pos_enc


def enable_preproc(cfg: dict = None):
    """Swap the dataset classes of mosaic.pipeline for the pre-processing versions."""
    cfg = dict(DEFAULT_PREPROC_CFG if cfg is None else cfg)
    Img2ExprDatasetPreproc.preproc_cfg = cfg
    Img2ExprInferDatasetPreproc.preproc_cfg = cfg
    base.Img2ExprDataset = Img2ExprDatasetPreproc
    base.Img2ExprInferDataset = Img2ExprInferDatasetPreproc
    print("[INFO] Patch pre-processing enabled: robust_norm + stain_like + qc/autocontrast + contrast_gain")
    print(f"[INFO] Config: {cfg}")


def run(args):
    enable_preproc()
    return base.run(args)


def main(argv=None):
    run(base.parse_args(argv))


if __name__ == "__main__":
    main()
