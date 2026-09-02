#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSAIC two-phase pipeline (slide-disjoint protocol).

Split
  Train : 100% of the training slide      (--train_image, default UC6)
  Val   : a stratified fraction of a second slide (--val_image,  default UC7, 20%)
  Test  : a stratified fraction of a third slide  (--test_image, default UC1, 10%)

Phase A  H&E patch + K spatial neighbours -> 460-gene expression (ConvNeXt-Tiny +
         attention aggregation). Early stopping on mean per-gene PCC (validation);
         mean per-gene PCC reported on the test subset.
Phase B  MLP (460 -> 256 -> 128 -> 9) trained on Phase A predictions of the training
         slide; early stopping on validation macro-F1; accuracy / F1 reported on the
         test subset. No image is used in Phase B.

Reproducibility: a fixed --seed selects the same validation/test cells on every run.
The selected indices are saved in --out_dir and can be re-used through --indices_dir.

Usage (from the repository root):
  python scripts/train_mosaic.py --k_neighbors 6 --seed 42 --out_dir results/k6_seed42
"""
import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ConvNeXt_Tiny_Weights
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


# ==============================================================================
# PATHS
# ==============================================================================

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "data")
DEFAULT_MARKERS_CSV = os.path.join(REPO_ROOT, "resources", "marker_genes_top10_per_cell_type.csv")
DEFAULT_GENES_LIST = os.path.join(REPO_ROOT, "resources", "gene_panel_460.txt")

# Relative to --data_root. Each folder holds manifest.csv, expr.npy and one PNG per cell.
IMAGE_PATHS = {
    "UC6": {"manifest": "patches128_2_UC6/manifest.csv", "expr": "patches128_2_UC6/expr.npy"},
    "UC7": {"manifest": "patches128_2_UC7/manifest.csv", "expr": "patches128_2_UC7/expr.npy"},
    "UC1": {"manifest": "patches128_2_UC1/manifest.csv", "expr": "patches128_2_UC1/expr.npy"},
}


# ==============================================================================
# UTILS
# ==============================================================================

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _make_seed_worker_fn(base_seed: int):
    """Per-worker seeding so augmentations and shuffling are reproducible with num_workers > 0."""
    def _seed_worker(worker_id: int):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)
    return _seed_worker


def normalize_expr(expr: np.ndarray, log1p: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optional log1p followed by per-gene z-score. Returns (expr_norm, mean, std)."""
    if log1p:
        expr = np.log1p(expr)
    mean = expr.mean(axis=0, keepdims=True)
    std = expr.std(axis=0, keepdims=True) + 1e-6
    return (expr - mean) / std, mean, std


# Marker-gene groups (rows of the marker CSV) -> manifest class names.
MARKER_GROUP_TO_CLASS = {
    "Epithelial Cells": ["Epithelial_Cells"],
    "Myeloid Cells": ["Myeloid_Cells"],
    "Fibroblasts": ["Fibroblasts"],
    "Plasma Cells": ["B_Cells"],
    "T Cells": ["T_Cells"],
    "B Cells": ["B_Cells"],
    "Glia Cells": ["Glia_Cells"],
    "Endothelial Cells": ["Endothelial_Cells", "Lymphatic_Endothelial_Cells"],
    "Vascular Smooth Muscle": ["Vascular_Smooth_Muscle"],
}


def _load_marker_gene_names(markers_csv: str, class_names: list) -> set:
    df = pd.read_csv(markers_csv)
    if "names" not in df.columns or "group" not in df.columns:
        return set()
    class_set = set(class_names)
    genes = set()
    for _, row in df.iterrows():
        group = str(row["group"]).strip()
        if group not in MARKER_GROUP_TO_CLASS:
            continue
        for c in MARKER_GROUP_TO_CLASS[group]:
            if c in class_set:
                genes.add(str(row["names"]).strip())
                break
    return genes


def _load_genes_list(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_marker_gene_weights(markers_csv: str, genes_list_path: str, class_names: list,
                            num_genes: int, marker_weight: float = 2.0) -> Optional[np.ndarray]:
    """Per-gene loss weights: `marker_weight` for curated marker genes, 1.0 otherwise.

    Returns None when the gene list does not match the expression matrix width, in
    which case a plain (unweighted) MSE is used.
    """
    genes_list = _load_genes_list(genes_list_path)
    if len(genes_list) != num_genes:
        return None
    marker_names = _load_marker_gene_names(markers_csv, class_names)
    w = np.ones(num_genes, dtype=np.float32)
    for i, g in enumerate(genes_list):
        if g in marker_names:
            w[i] = marker_weight
    return w


def knn_indices(coords: np.ndarray, k: int) -> np.ndarray:
    """Indices of the K nearest neighbours (self excluded). For k=0 returns (n, 1) with the
    self index so that tensors are never empty (backbone-only baseline)."""
    if k <= 0:
        return np.arange(len(coords), dtype=np.int64)[:, np.newaxis]
    k_eff = min(k, len(coords) - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto").fit(coords)
    _, indices = nbrs.kneighbors(coords)
    return indices[:, 1:]


def build_backbone(name: str, pretrained: bool):
    if name.lower() == "convnext_tiny":
        backbone = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    return backbone, feat_dim


def resolve_paths(records: List[Dict], manifest_path: str, manifest_dir: str = None) -> None:
    """Rewrite patch paths so they point inside the folder that contains the manifest.

    Manifests may store paths such as 'patches128_2/<type>/<id>.png' or
    'patches128_2_UC6/<type>/<id>.png'; the first segment is discarded and replaced by
    the manifest directory, so the dataset folder can be moved or renamed freely.
    """
    if manifest_dir is None:
        manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    for rec in records:
        p = rec["path"].replace("\\", "/")
        if "/" in p:
            rest = p.split("/", 1)[1]
            rec["path"] = os.path.normpath(os.path.join(manifest_dir, rest))
        else:
            rec["path"] = os.path.normpath(os.path.join(manifest_dir, p))


# ==============================================================================
# PHASE A: image (+ neighbours) -> expression
# ==============================================================================

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transform(img_size: int, augment: bool):
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.03, hue=0.01),
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class Img2ExprDataset(Dataset):
    """Target patch, K neighbour patches, positional features and target expression."""

    def __init__(self, records: List[Dict], expr: np.ndarray, img_size: int,
                 neighbor_idx: np.ndarray, pos_scale: float, augment: bool = False,
                 use_graph: bool = True):
        self.records = records
        self.expr = expr.astype(np.float32)
        self.neighbor_idx = neighbor_idx
        self.pos_scale = float(pos_scale)
        self.use_graph = bool(use_graph)
        self.tx = _build_transform(img_size, augment)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        img_t = self.tx(img)
        expr_t = torch.tensor(self.expr[idx], dtype=torch.float32)
        neigh_ids = self.neighbor_idx[idx]
        neigh_imgs = []
        pos_enc = []
        for n_idx in neigh_ids:
            n_rec = self.records[int(n_idx)]
            n_img = Image.open(n_rec["path"]).convert("RGB")
            neigh_imgs.append(self.tx(n_img))
            if self.use_graph:
                dx = n_rec["cx"] - rec["cx"]
                dy = n_rec["cy"] - rec["cy"]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-6
                ang = np.arctan2(dy, dx)
                pos_enc.append([dx / self.pos_scale, dy / self.pos_scale,
                                dist / self.pos_scale, np.sin(ang), np.cos(ang)])
        if self.use_graph:
            pos_enc = torch.tensor(pos_enc, dtype=torch.float32)
        else:
            pos_enc = torch.zeros(len(neigh_imgs), 5, dtype=torch.float32)
        # K=0 / empty neighbour list: use the target itself so the stack is never empty
        if len(neigh_imgs) == 0:
            neigh_imgs = img_t.unsqueeze(0)
            pos_enc = torch.zeros(1, 5, dtype=torch.float32)
        else:
            neigh_imgs = torch.stack(neigh_imgs, dim=0)
        return img_t, neigh_imgs, pos_enc, expr_t


class Img2ExprInferDataset(Dataset):
    """Same as Img2ExprDataset but without targets (inference on a whole slide)."""

    def __init__(self, records: List[Dict], img_size: int, neighbor_idx: np.ndarray,
                 pos_scale: float, use_graph: bool = True):
        self.records = records
        self.neighbor_idx = neighbor_idx
        self.pos_scale = float(pos_scale)
        self.use_graph = bool(use_graph)
        self.tx = _build_transform(img_size, augment=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        img_t = self.tx(img)
        neigh_ids = self.neighbor_idx[idx]
        neigh_imgs = []
        pos_enc = []
        for n_idx in neigh_ids:
            n_rec = self.records[int(n_idx)]
            n_img = Image.open(n_rec["path"]).convert("RGB")
            neigh_imgs.append(self.tx(n_img))
            if self.use_graph:
                dx = n_rec["cx"] - rec["cx"]
                dy = n_rec["cy"] - rec["cy"]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-6
                ang = np.arctan2(dy, dx)
                pos_enc.append([dx / self.pos_scale, dy / self.pos_scale,
                                dist / self.pos_scale, np.sin(ang), np.cos(ang)])
        if self.use_graph:
            pos_enc = torch.tensor(pos_enc, dtype=torch.float32)
        else:
            pos_enc = torch.zeros(len(neigh_imgs), 5, dtype=torch.float32)
        if len(neigh_imgs) == 0:
            neigh_imgs = img_t.unsqueeze(0)
            pos_enc = torch.zeros(1, 5, dtype=torch.float32)
        else:
            neigh_imgs = torch.stack(neigh_imgs, dim=0)
        return img_t, neigh_imgs, pos_enc


class Img2ExprGnn(nn.Module):
    """ConvNeXt-Tiny encoder (GAP || GeM = 1536-d) + neighbour aggregation + expression head.

    aggregation_mode: "attention" (scaled dot-product, target = query, neighbours =
    keys/values), "mean" or "max" pooling over projected neighbour features.
    use_graph adds a learned embedding of the relative position of each neighbour
    (dx, dy, distance, sin, cos); the MOSAIC paper configuration uses use_graph=False.
    use_context=False disables the neighbourhood entirely (K=0 baseline).
    """

    def __init__(self, num_genes: int, dropout: float = 0.1, pos_dim: int = 96, use_graph: bool = True,
                 aggregation_mode: str = "attention", use_context: bool = True):
        super().__init__()
        self.use_graph = bool(use_graph)
        self.pos_dim = pos_dim
        self.aggregation_mode = aggregation_mode
        self.use_context = bool(use_context)
        self.backbone, feat_dim = build_backbone("convnext_tiny", pretrained=True)
        in_dim = feat_dim * 2
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)
        self.pos_mlp = nn.Sequential(
            nn.Linear(5, pos_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(pos_dim, pos_dim),
        )
        self.attn_q = nn.Linear(in_dim, in_dim)
        self.attn_k = nn.Linear(in_dim + pos_dim, in_dim)
        self.attn_v = nn.Linear(in_dim + pos_dim, in_dim)
        # mean/max aggregation: project (in_dim + pos_dim) -> in_dim so feat_all stays in_dim * 2
        self.agg_proj = nn.Linear(in_dim + pos_dim, in_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim * 2, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, num_genes),
        )
        self.gene_block = nn.Sequential(
            nn.Linear(num_genes, 512), nn.GELU(), nn.Linear(512, num_genes),
            nn.Linear(num_genes, 512), nn.GELU(), nn.Linear(512, num_genes),
        )

    def _encode_single(self, img):
        x = self.backbone.features(img)
        gap = self.backbone.avgpool(x)
        feat_gap = torch.flatten(gap, 1)
        p = torch.clamp(self.gem_p, min=1e-3)
        x_clamped = torch.clamp(x, min=1e-6)
        gem = torch.pow(torch.mean(torch.pow(x_clamped, p.unsqueeze(-1).unsqueeze(-1)),
                                   dim=[2, 3], keepdim=True), 1.0 / p.unsqueeze(-1).unsqueeze(-1))
        feat_gem = torch.flatten(gem, 1)
        return torch.cat([feat_gap, feat_gem], dim=1)

    def forward(self, img, neigh_imgs, pos_enc):
        B, K, C, H, W = neigh_imgs.shape
        feat_self = self._encode_single(img)
        if not self.use_context:
            # K=0: backbone only; zero context keeps the head input dimension unchanged
            neigh_context = torch.zeros(B, feat_self.size(-1), device=img.device, dtype=img.dtype)
        else:
            neigh_flat = neigh_imgs.view(B * K, C, H, W)
            feat_neigh = self._encode_single(neigh_flat).view(B, K, -1)
            if self.use_graph:
                pos_embed = self.pos_mlp(pos_enc)
            else:
                pos_embed = torch.zeros(B, K, self.pos_dim, device=img.device, dtype=img.dtype)
            feat_neigh_pos = torch.cat([feat_neigh, pos_embed], dim=-1)
            if self.aggregation_mode == "mean":
                neigh_context = self.agg_proj(feat_neigh_pos).mean(dim=1)
            elif self.aggregation_mode == "max":
                neigh_context = self.agg_proj(feat_neigh_pos).max(dim=1)[0]
            else:
                q = self.attn_q(feat_self).unsqueeze(1)
                k = self.attn_k(feat_neigh_pos)
                v = self.attn_v(feat_neigh_pos)
                attn_logits = torch.sum(q * k, dim=-1) / (k.size(-1) ** 0.5)
                attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)
                neigh_context = torch.sum(attn_weights * v, dim=1)
        feat_all = torch.cat([feat_self, neigh_context], dim=-1)
        expr_out = self.head(feat_all)
        expr_out = expr_out + self.gene_block(expr_out)
        return expr_out


class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model):
        model.load_state_dict(self.shadow)


# ==============================================================================
# PHASE B: expression -> cell lineage
# ==============================================================================

class ExprDataset(Dataset):
    def __init__(self, expr: np.ndarray, labels: np.ndarray):
        self.expr = torch.tensor(expr, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.expr)

    def __getitem__(self, idx):
        return self.expr[idx], self.labels[idx]


class ExprClassifier(nn.Module):
    """MLP 460 -> 256 -> 128 -> n_classes (ReLU, dropout 0.5)."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dims: list = None, dropout: float = 0.5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# TRAINING / INFERENCE
# ==============================================================================

def _mean_per_gene_pcc(pred: np.ndarray, gt: np.ndarray) -> float:
    vals = []
    for g in range(gt.shape[1]):
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(pred[:, g], gt[:, g])[0, 1]
        if not np.isnan(c):
            vals.append(float(c))
    return float(np.mean(vals)) if vals else 0.0


def train_expression_model(
    train_records, train_expr, val_records, val_expr,
    args, device, gene_weights: np.ndarray = None
) -> Tuple[nn.Module, float, float]:
    def get_coords(recs):
        return np.array([[r["cx"], r["cy"]] for r in recs], dtype=np.float32)
    train_coords = get_coords(train_records)
    val_coords = get_coords(val_records)
    pos_scale = max(np.max(np.abs(train_coords)), np.max(np.abs(val_coords))) + 1e-6
    train_knn = knn_indices(train_coords, args.k_neighbors)
    val_knn = knn_indices(val_coords, args.k_neighbors)
    use_graph = getattr(args, "use_graph", False)

    train_ds = Img2ExprDataset(train_records, train_expr, args.img_size, train_knn, pos_scale,
                               augment=True, use_graph=use_graph)
    val_ds = Img2ExprDataset(val_records, val_expr, args.img_size, val_knn, pos_scale,
                             augment=False, use_graph=use_graph)
    g = torch.Generator()
    g.manual_seed(args.seed)
    worker_init_fn = _make_seed_worker_fn(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, generator=g,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    num_genes = train_expr.shape[1]
    use_context = getattr(args, "k_neighbors", 12) > 0
    aggregation = getattr(args, "aggregation_mode", "attention")
    model = Img2ExprGnn(
        num_genes, dropout=0.1, pos_dim=args.pos_dim, use_graph=use_graph,
        aggregation_mode=aggregation, use_context=use_context,
    ).to(device)
    ema = ModelEMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_expr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.3, patience=2)
    gene_weights_t = torch.tensor(gene_weights, dtype=torch.float32, device=device) if gene_weights is not None else None

    def _loss(pred, target):
        if gene_weights_t is not None:
            return (gene_weights_t.unsqueeze(0) * (pred - target) ** 2).mean()
        return F.mse_loss(pred, target)

    best_pearson = -1
    best_state = None
    wait = 0
    mse = 0.0
    for epoch in range(1, args.epochs_expr + 1):
        model.train()
        pbar_batch = tqdm(train_loader, desc=f"Phase A epoch {epoch}/{args.epochs_expr}", leave=True)
        for img, neigh_imgs, pos_enc, expr_gt in pbar_batch:
            img = img.to(device)
            neigh_imgs = neigh_imgs.to(device)
            pos_enc = pos_enc.to(device)
            expr_gt = expr_gt.to(device)
            opt.zero_grad()
            pred = model(img, neigh_imgs, pos_enc)
            loss = _loss(pred, expr_gt)
            loss.backward()
            opt.step()
            ema.update(model)
            pbar_batch.set_postfix(loss=f"{loss.item():.4f}")

        ema.apply(model)
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for img, neigh_imgs, pos_enc, expr_gt in val_loader:
                img = img.to(device)
                neigh_imgs = neigh_imgs.to(device)
                pos_enc = pos_enc.to(device)
                pred = model(img, neigh_imgs, pos_enc)
                preds.append(pred.cpu().numpy())
                gts.append(expr_gt.numpy())
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        mean_pearson = _mean_per_gene_pcc(preds, gts)
        mse = float(np.mean((preds - gts) ** 2))
        scheduler.step(mean_pearson)
        print(f"     epoch {epoch}: val mean per-gene PCC = {mean_pearson:.4f} | val MSE = {mse:.6f}")

        if mean_pearson > best_pearson:
            best_pearson = mean_pearson
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.early_stop_expr:
                print(f"     early stopping after {epoch} epochs (patience {args.early_stop_expr})")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_pearson, mse


def generate_predictions(model, records, args, device) -> np.ndarray:
    """Predict expression for every cell of a slide (KNN computed on the whole slide)."""
    coords = np.array([[r["cx"], r["cy"]] for r in records], dtype=np.float32)
    pos_scale = float(np.max(np.abs(coords)) + 1e-6)
    knn = knn_indices(coords, args.k_neighbors)
    use_graph = getattr(args, "use_graph", False)
    ds = Img2ExprInferDataset(records, args.img_size, knn, pos_scale, use_graph=use_graph)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        worker_init_fn=_make_seed_worker_fn(args.seed))
    model.eval()
    preds = []
    with torch.no_grad():
        for img, neigh_imgs, pos_enc in tqdm(loader, desc="Inference"):
            img = img.to(device)
            neigh_imgs = neigh_imgs.to(device)
            pos_enc = pos_enc.to(device)
            pred = model(img, neigh_imgs, pos_enc)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_classifier(
    train_expr, train_labels, val_expr, val_labels,
    num_classes, args, device, class_weights=None
) -> Tuple[nn.Module, float, np.ndarray, np.ndarray]:
    """Returns (model, best_val_macro_f1, mean, std); mean/std are the training statistics
    used to z-score the classifier input and must be re-used at evaluation time."""
    mean = train_expr.mean(axis=0, keepdims=True)
    std = train_expr.std(axis=0, keepdims=True) + 1e-6
    train_norm = (train_expr - mean) / std
    val_norm = (val_expr - mean) / std
    train_ds = ExprDataset(train_norm, train_labels)
    val_ds = ExprDataset(val_norm, val_labels)
    num_workers = getattr(args, "num_workers", 4)
    g = torch.Generator()
    g.manual_seed(args.seed)
    worker_init_fn = _make_seed_worker_fn(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True,
        generator=g, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    model = ExprClassifier(train_expr.shape[1], num_classes, [256, 128], dropout=0.5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    cls_weights = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None

    best_f1 = -1
    best_state = None
    wait = 0
    for epoch in tqdm(range(1, args.epochs_cls + 1), desc="Phase B classifier"):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=cls_weights)
            loss.backward()
            opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                preds.append(logits.argmax(1).cpu().numpy())
                trues.append(y.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        f1 = f1_score(trues, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.early_stop_cls:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1, mean, std


def evaluate_classifier(model, expr, labels, device, cls_mean=None, cls_std=None) -> Dict:
    """Evaluate the classifier. When cls_mean/cls_std are given (training statistics) they
    are used for z-scoring; otherwise the statistics of `expr` itself are used."""
    if cls_mean is not None and cls_std is not None:
        expr_norm = (expr - cls_mean) / cls_std
    else:
        mean = expr.mean(axis=0, keepdims=True)
        std = expr.std(axis=0, keepdims=True) + 1e-6
        expr_norm = (expr - mean) / std
    ds = ExprDataset(expr_norm, labels)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(y.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return {
        "accuracy": accuracy_score(trues, preds),
        "f1_macro": f1_score(trues, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(trues, preds, average="weighted", zero_division=0),
        "predictions": preds,
        "true_labels": trues,
    }


def select_genes(train_pred, train_labels, method="none", n_genes=200) -> np.ndarray:
    """Optional gene subset for Phase B (not used in the paper; default keeps all genes)."""
    n_cells, n_genes_total = train_pred.shape
    if method == "none" or n_genes_total == 0:
        return np.arange(n_genes_total)
    n_keep = min(n_genes, n_genes_total)
    if method == "kbest":
        selector = SelectKBest(score_func=f_classif, k=n_keep)
        selector.fit(train_pred, train_labels)
        return np.where(selector.get_support())[0]
    return np.arange(n_genes_total)


def select_genes_by_val_pcc(val_pred: np.ndarray, val_gt: np.ndarray, n_genes: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Top genes by |PCC| between predicted and measured expression on the validation set."""
    n_genes_total = val_pred.shape[1]
    if n_genes_total == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    n_keep = min(max(int(n_genes), 1), n_genes_total)
    pcc = np.zeros(n_genes_total, dtype=np.float32)
    for g in range(n_genes_total):
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(val_pred[:, g], val_gt[:, g])[0, 1]
        pcc[g] = 0.0 if np.isnan(c) else float(c)
    ranked = np.argsort(np.abs(pcc))[::-1]
    keep_idx = ranked[:n_keep].astype(np.int64)
    return np.sort(keep_idx), pcc


# ==============================================================================
# DATA LOADING AND SPLITS
# ==============================================================================

def load_image_data(image_key: str, data_root: str) -> Tuple[pd.DataFrame, np.ndarray, List[Dict]]:
    """Load manifest.csv, expr.npy and per-cell records for one slide."""
    cfg = IMAGE_PATHS[image_key]
    manifest_path = os.path.join(data_root, cfg["manifest"])
    expr_path = os.path.join(data_root, cfg["expr"])
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not os.path.exists(expr_path):
        raise FileNotFoundError(f"Expression matrix not found: {expr_path}")

    manifest = pd.read_csv(manifest_path)
    expr_raw = np.load(expr_path).astype(np.float32)
    if len(manifest) != len(expr_raw):
        raise ValueError(f"{image_key}: manifest ({len(manifest)}) != expr ({len(expr_raw)})")

    records = manifest.to_dict(orient="records")
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    resolve_paths(records, manifest_path, manifest_dir)
    return manifest, expr_raw, records


def stratified_subset_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Indices of a stratified random subset (sklearn train_test_split, test_size=fraction)."""
    _, idx = train_test_split(
        np.arange(len(labels)), test_size=fraction, random_state=seed, stratify=labels,
    )
    return np.asarray(idx, dtype=np.int64)


def _indices_filename(kind: str, image: str, fraction: float) -> str:
    return f"{kind}_indices_{image}_pct{int(round(fraction * 100))}.npy"


def _load_or_make_indices(kind: str, image: str, fraction: float, labels: np.ndarray,
                          seed: int, indices_dir: Optional[str]) -> np.ndarray:
    if indices_dir:
        path = os.path.join(indices_dir, _indices_filename(kind, image, fraction))
        if os.path.isfile(path):
            idx = np.load(path).astype(np.int64)
            if idx.max() >= len(labels):
                raise ValueError(f"{path}: indices exceed the number of cells in {image}")
            print(f"     {kind} indices loaded from {path} ({len(idx)} cells)")
            return idx
        print(f"     [WARN] {path} not found; computing {kind} split from seed {seed}")
    return stratified_subset_indices(labels, fraction, seed)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = os.path.abspath(getattr(args, "data_root", None) or DEFAULT_DATA_ROOT)
    use_graph = getattr(args, "use_graph", False)
    train_image = args.train_image.upper()
    val_image = args.val_image.upper()
    test_image = args.test_image.upper()
    val_frac = getattr(args, "val_fraction", 0.2)
    test_frac = getattr(args, "test_fraction", 0.1)
    indices_dir = getattr(args, "indices_dir", None)

    if len({train_image, val_image, test_image}) != 3:
        raise ValueError("--train_image, --val_image and --test_image must be three different slides")
    for name in (train_image, val_image, test_image):
        if name not in IMAGE_PATHS:
            raise ValueError(f"Slide must be one of: {list(IMAGE_PATHS.keys())}")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Data root: {data_root}")
    print(f"[INFO] Train: {train_image} 100% | Val: {val_image} {int(val_frac*100)}% | Test: {test_image} {int(test_frac*100)}%")
    print(f"[INFO] Seed: {args.seed}")
    print(f"[INFO] K neighbours: {args.k_neighbors} | aggregation: {getattr(args, 'aggregation_mode', 'attention')} "
          f"| positional encoding: {'on' if use_graph else 'off'}")

    # --- Load the three slides ---
    print(f"\n[1] Loading {train_image}, {val_image} and {test_image}...")
    manifest_train, expr_train, records_train = load_image_data(train_image, data_root)
    manifest_val, expr_val, records_val = load_image_data(val_image, data_root)
    manifest_test, expr_test, records_test = load_image_data(test_image, data_root)
    n_genes_ref = expr_train.shape[1]
    for name, expr in [(val_image, expr_val), (test_image, expr_test)]:
        if expr.shape[1] != n_genes_ref:
            raise ValueError(
                f"Gene panel mismatch: {train_image} has {n_genes_ref} genes, {name} has {expr.shape[1]}"
            )

    expr_norm_train, _, _ = normalize_expr(expr_train, log1p=args.log1p_normalize)
    expr_norm_val, _, _ = normalize_expr(expr_val, log1p=args.log1p_normalize)
    expr_norm_test, _, _ = normalize_expr(expr_test, log1p=args.log1p_normalize)

    all_classes = sorted(
        set(manifest_train["cell_type"].values)
        | set(manifest_val["cell_type"].values)
        | set(manifest_test["cell_type"].values)
    )
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    classes = all_classes

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Train: 100% of the training slide ---
    train_records = list(records_train)
    train_expr = expr_norm_train.copy()
    train_labels = np.array([class_to_idx[c] for c in manifest_train["cell_type"].values])
    print(f"     Train {train_image}: {len(train_records)} cells (100%)")

    # --- Val / Test: fixed stratified subsets (seed) ---
    labels_val = np.array([class_to_idx[c] for c in manifest_val["cell_type"].values])
    val_idx = _load_or_make_indices("val", val_image, val_frac, labels_val, args.seed, indices_dir)
    print(f"     Val {val_image}: {len(val_idx)} cells ({int(val_frac*100)}%, seed={args.seed})")
    val_records = [records_val[i] for i in val_idx]
    val_expr = expr_norm_val[val_idx]
    val_labels = labels_val[val_idx]

    labels_test = np.array([class_to_idx[c] for c in manifest_test["cell_type"].values])
    test_idx = _load_or_make_indices("test", test_image, test_frac, labels_test, args.seed, indices_dir)
    print(f"     Test {test_image}: {len(test_idx)} cells ({int(test_frac*100)}%, seed={args.seed})")
    test_records = [records_test[i] for i in test_idx]
    test_expr_gt = expr_norm_test[test_idx]
    test_labels = labels_test[test_idx]

    np.save(os.path.join(args.out_dir, _indices_filename("val", val_image, val_frac)), val_idx)
    np.save(os.path.join(args.out_dir, _indices_filename("test", test_image, test_frac)), test_idx)
    with open(os.path.join(args.out_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    # --- Optional marker-gene weights for the Phase A loss ---
    gene_weights = None
    markers_csv = getattr(args, "markers_csv", None)
    genes_list = getattr(args, "genes_list", None)
    if markers_csv and genes_list and os.path.isfile(markers_csv) and os.path.isfile(genes_list):
        gene_weights = get_marker_gene_weights(
            markers_csv, genes_list, classes,
            expr_norm_train.shape[1], marker_weight=getattr(args, "marker_loss_weight", 2.0),
        )
        if gene_weights is not None:
            n_marker = int((gene_weights > 1.0).sum())
            print(f"     Marker genes: {n_marker} genes weighted x{getattr(args, 'marker_loss_weight', 2.0)} in the loss")
    if gene_weights is None:
        print("     Phase A loss: unweighted MSE")

    # --- PHASE A: train on the training slide, monitor on the validation subset ---
    print(f"\n[2] PHASE A: training the expression model on {train_image}...")
    print(f"     (the first batch may take 1-2 min with {args.num_workers} workers)")
    expr_model, val_pearson, val_mse = train_expression_model(
        train_records, train_expr, val_records, val_expr, args, device, gene_weights=gene_weights
    )
    print(f"     Best val mean per-gene PCC: {val_pearson:.4f} | val MSE: {val_mse:.6f}")
    model_path = os.path.join(args.out_dir, f"expr_model_{train_image}.pt")
    torch.save(expr_model.state_dict(), model_path)
    print(f"     Model saved to {model_path}")

    # --- PHASE A: predict expression for the whole validation and test slides ---
    print(f"\n[3] PHASE A: predicting expression on {val_image} and {test_image}...")
    val_pred_full = generate_predictions(expr_model, records_val, args, device)
    test_pred_full = generate_predictions(expr_model, records_test, args, device)

    val_pred = val_pred_full[val_idx]
    test_pred_sub = test_pred_full[test_idx]

    val_pearson_final = _mean_per_gene_pcc(val_pred, val_expr)
    print(f"     Val mean per-gene PCC ({val_image} {int(val_frac*100)}%): {val_pearson_final:.4f}")
    test_pearson = _mean_per_gene_pcc(test_pred_sub, test_expr_gt)
    print(f"     Test mean per-gene PCC ({test_image} {int(test_frac*100)}%): {test_pearson:.4f}")

    if not getattr(args, "no_save_full_predictions", False):
        np.save(os.path.join(args.out_dir, f"expr_pred_full_{val_image}.npy"), val_pred_full)
        np.save(os.path.join(args.out_dir, f"expr_pred_full_{test_image}.npy"), test_pred_full)
        print("     Predicted expression matrices saved")
    else:
        print("     Predicted expression matrices not saved (--no_save_full_predictions)")

    # --- PHASE B: classifier on predicted expression ---
    print(f"\n[4] PHASE B: training the lineage classifier ({train_image} train, {val_image} val)...")
    train_pred = generate_predictions(expr_model, train_records, args, device)
    if not getattr(args, "no_save_full_predictions", False):
        np.save(os.path.join(args.out_dir, f"expr_pred_full_{train_image}.npy"), train_pred)

    gene_select_method = getattr(args, "gene_select_method", "none")
    gene_select_n = int(getattr(args, "gene_select_n", 200))
    selected_gene_idx = np.arange(train_pred.shape[1], dtype=np.int64)
    selected_gene_pcc = None
    if gene_select_method == "pcc_topk":
        selected_gene_idx, selected_gene_pcc = select_genes_by_val_pcc(val_pred, val_expr, n_genes=gene_select_n)
        print(f"     Gene selection (val PCC): top-{len(selected_gene_idx)} of {train_pred.shape[1]}")
    elif gene_select_method == "kbest":
        selected_gene_idx = select_genes(
            train_pred, train_labels, method="kbest", n_genes=gene_select_n
        ).astype(np.int64)
        print(f"     Gene selection (ANOVA k-best): top-{len(selected_gene_idx)} of {train_pred.shape[1]}")
    else:
        print(f"     Gene selection: off (all {len(selected_gene_idx)} genes)")

    train_pred_sel = train_pred[:, selected_gene_idx]
    val_pred_sel = val_pred[:, selected_gene_idx]
    test_pred_sel = test_pred_sub[:, selected_gene_idx]

    np.save(os.path.join(args.out_dir, "selected_gene_idx.npy"), selected_gene_idx.astype(np.int64))
    gene_sel_meta = {
        "method": gene_select_method,
        "n_requested": gene_select_n,
        "n_selected": int(len(selected_gene_idx)),
        "n_total": int(train_pred.shape[1]),
    }
    if selected_gene_pcc is not None:
        top_order = np.argsort(np.abs(selected_gene_pcc[selected_gene_idx]))[::-1]
        top_idx = selected_gene_idx[top_order][:20]
        gene_sel_meta["top20_gene_idx_by_abs_pcc"] = [int(i) for i in top_idx]
        gene_sel_meta["top20_gene_abs_pcc"] = [float(abs(selected_gene_pcc[i])) for i in top_idx]
    with open(os.path.join(args.out_dir, "gene_selection.json"), "w", encoding="utf-8") as f:
        json.dump(gene_sel_meta, f, indent=2)

    classifier, cls_val_f1, cls_mean, cls_std = train_classifier(
        train_pred_sel, train_labels,
        val_pred_sel, val_labels,
        len(classes), args, device,
        class_weights=None
    )
    cls_path = os.path.join(args.out_dir, f"classifier_{train_image}.pt")
    torch.save(classifier.state_dict(), cls_path)
    np.savez(os.path.join(args.out_dir, f"classifier_{train_image}_norm.npz"), mean=cls_mean, std=cls_std)
    print(f"     Classifier saved to {cls_path}")

    # --- PHASE B: evaluate on the test subset (z-score with TRAINING statistics) ---
    print(f"\n[5] PHASE B: classifying {test_image} {int(test_frac*100)}%...")
    test_results = evaluate_classifier(
        classifier, test_pred_sel, test_labels, device,
        cls_mean=cls_mean, cls_std=cls_std
    )
    print(f"     Accuracy: {test_results['accuracy']:.4f} | F1 macro: {test_results['f1_macro']:.4f} "
          f"| F1 weighted: {test_results['f1_weighted']:.4f}")

    # --- Save results ---
    results = {
        "train_image": train_image,
        "val_image": val_image,
        "test_image": test_image,
        "val_fraction": val_frac,
        "test_fraction": test_frac,
        "train_size": len(train_records),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "seed": args.seed,
        "k_neighbors": args.k_neighbors,
        "aggregation_mode": getattr(args, "aggregation_mode", "attention"),
        "use_graph": bool(use_graph),
        "log1p_normalize": bool(args.log1p_normalize),
        "marker_weighted_loss": bool(gene_weights is not None),
        "expr_val_pearson": val_pearson_final,
        "expr_test_pearson": test_pearson,
        "gene_select_method": gene_select_method,
        "gene_select_n": gene_select_n,
        "gene_selected_count": int(len(selected_gene_idx)),
        "cls_val_f1": cls_val_f1,
        "test_accuracy": test_results["accuracy"],
        "test_f1_macro": test_results["f1_macro"],
        "test_f1_weighted": test_results["f1_weighted"],
    }
    results_path = os.path.join(args.out_dir, "results.csv")
    pd.DataFrame([results]).to_csv(results_path, index=False)
    print(f"\n[OK] Results saved to {results_path}")

    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                   for k, v in vars(args).items()}, f, indent=2)

    cm = confusion_matrix(test_results["true_labels"], test_results["predictions"],
                          labels=list(range(len(classes))))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        os.path.join(args.out_dir, f"confusion_{test_image}.csv")
    )
    plt.figure(figsize=(12, 10))
    cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-6)
    ax = sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                     xticklabels=classes, yticklabels=classes)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(f"Confusion matrix - {train_image} train / {val_image} val {int(val_frac*100)}% / "
              f"{test_image} test {int(test_frac*100)}%\nAccuracy: {test_results['accuracy']:.4f}")
    plt.tight_layout()
    cm_path = os.path.join(args.out_dir, f"confusion_{test_image}.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[OK] Confusion matrix saved to {cm_path}")

    report = classification_report(
        test_results["true_labels"], test_results["predictions"],
        labels=list(range(len(classes))), target_names=classes,
        output_dict=True, zero_division=0,
    )
    metrics_path = os.path.join(args.out_dir, f"metrics_{test_image}.csv")
    pd.DataFrame(report).T.to_csv(metrics_path)
    print(f"[OK] Per-class metrics saved to {metrics_path}")
    return results


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="MOSAIC: train 100%% of one slide | validate on a fraction of a second | test on a fraction of a third"
    )
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT,
                    help="Folder containing patches128_2_UC1/, patches128_2_UC6/, patches128_2_UC7/ (default: <repo>/data)")
    ap.add_argument("--train_image", default="UC6", choices=list(IMAGE_PATHS.keys()),
                    help="Slide used for training (100%%)")
    ap.add_argument("--val_image", default="UC7", choices=list(IMAGE_PATHS.keys()),
                    help="Slide used for validation / early stopping")
    ap.add_argument("--test_image", default="UC1", choices=list(IMAGE_PATHS.keys()),
                    help="Slide used for testing")
    ap.add_argument("--val_fraction", type=float, default=0.2,
                    help="Stratified fraction of the validation slide (default 0.2)")
    ap.add_argument("--test_fraction", type=float, default=0.1,
                    help="Stratified fraction of the test slide (default 0.1)")
    ap.add_argument("--indices_dir", default=None,
                    help="Folder with val_indices_*.npy / test_indices_*.npy from a previous run (re-use the exact same cells)")
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "mosaic_run"),
                    help="Output folder")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--k_neighbors", type=int, default=6,
                    help="Number of spatial nearest neighbours (0 = backbone-only baseline)")
    ap.add_argument("--aggregation_mode", default="attention", choices=["attention", "mean", "max"],
                    help="Neighbour aggregation: attention (scaled dot-product), mean or max pooling")
    ap.add_argument("--use_graph", action="store_true",
                    help="Add learned relative-position encoding of neighbours (off in the paper)")
    ap.add_argument("--pos_dim", type=int, default=96)
    ap.add_argument("--lr_expr", type=float, default=2e-4)
    ap.add_argument("--epochs_expr", type=int, default=30)
    ap.add_argument("--early_stop_expr", type=int, default=6)
    ap.add_argument("--ema_decay", type=float, default=0.9995)
    ap.add_argument("--log1p_normalize", action="store_true",
                    help="Apply log(1+x) before per-gene z-scoring of the expression targets")
    ap.add_argument("--epochs_cls", type=int, default=30)
    ap.add_argument("--early_stop_cls", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--markers_csv", default=DEFAULT_MARKERS_CSV,
                    help="Marker-gene CSV (group,names); pass '' to disable marker weighting")
    ap.add_argument("--genes_list", default=DEFAULT_GENES_LIST,
                    help="Gene panel, one symbol per line, in the column order of expr.npy")
    ap.add_argument("--marker_loss_weight", type=float, default=2.0,
                    help="Loss weight applied to marker genes when --markers_csv/--genes_list are valid")
    ap.add_argument("--no_save_full_predictions", action="store_true",
                    help="Do not save expr_pred_full_*.npy (saves ~260 MB per slide)")
    ap.add_argument("--gene_select_method", default="none", choices=["none", "pcc_topk", "kbest"],
                    help="Optional gene subset for Phase B (not used in the paper)")
    ap.add_argument("--gene_select_n", type=int, default=200,
                    help="Number of genes kept when --gene_select_method != none")
    return ap


def parse_args(argv=None):
    args = build_argparser().parse_args(argv)
    if not (0 < args.val_fraction < 1) or not (0 < args.test_fraction < 1):
        raise ValueError("--val_fraction and --test_fraction must be in (0, 1)")
    return args


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
