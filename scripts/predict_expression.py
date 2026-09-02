#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase A inference: predict the expression matrix of a whole slide from a trained
checkpoint (expr_model_<TRAIN>.pt) and save it as expr_pred_full_<SLIDE>.npy.

  python scripts/predict_expression.py --checkpoint results/mosaic_k6_seed42/expr_model_UC6.pt \
      --image UC1 --k_neighbors 6

The architecture flags (--k_neighbors, --aggregation_mode, --use_graph, --pos_dim) must
match the ones used for training.
"""
import _repo  # noqa: F401

import argparse
import os

import numpy as np
import torch

from mosaic.pipeline import DEFAULT_DATA_ROOT, Img2ExprGnn, generate_predictions, load_image_data


def parse_args():
    p = argparse.ArgumentParser(description="Predict expression for a whole slide (MOSAIC Phase A)")
    p.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--image", default="UC6", choices=["UC6", "UC7", "UC1"])
    p.add_argument("--checkpoint", required=True, help="expr_model_<TRAIN>.pt")
    p.add_argument("--out_npy", default=None,
                   help="Output .npy (default: expr_pred_full_<IMAGE>.npy next to the checkpoint)")
    p.add_argument("--k_neighbors", type=int, default=6)
    p.add_argument("--aggregation_mode", default="attention", choices=["attention", "mean", "max"])
    p.add_argument("--use_graph", action="store_true")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pos_dim", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    out_npy = args.out_npy or os.path.join(os.path.dirname(ckpt), f"expr_pred_full_{args.image.upper()}.npy")
    out_npy = os.path.abspath(out_npy)

    print(f"[INFO] device={device}")
    print(f"[INFO] checkpoint={ckpt}")
    print(f"[INFO] image={args.image} k={args.k_neighbors} graph={args.use_graph} agg={args.aggregation_mode}")

    _, expr_raw, records = load_image_data(args.image.upper(), args.data_root)
    num_genes = expr_raw.shape[1]

    model = Img2ExprGnn(
        num_genes, dropout=0.1, pos_dim=args.pos_dim, use_graph=args.use_graph,
        aggregation_mode=args.aggregation_mode, use_context=args.k_neighbors > 0,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"[INFO] model loaded | genes={num_genes} | cells={len(records)}")

    pred = generate_predictions(model, records, args, device)
    assert pred.shape == (len(records), num_genes), f"unexpected shape: {pred.shape}"

    np.save(out_npy, pred.astype(np.float32))
    print(f"[OK] saved: {out_npy} shape={pred.shape} dtype=float32")


if __name__ == "__main__":
    main()
