#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-panel figure of a target cell and its K spatial nearest neighbours (Extended Data Fig. 1).

Panel (a): patch of the target cell.  Panel (b): same patch (or a zoomed-out field read
from the original Zarr store) with the target (blue) and its K neighbours (green).

  # from the 128x128 patch only
  python scripts/plot_neighborhood_figure.py --image_key UC6 --k 6 --seed 42 \
      --out_path results/figures/k6_neighbors_UC6.pdf

  # zoomed-out field of view read from the raw slide, ring markers with labels
  python scripts/plot_neighborhood_figure.py --image_key UC6 --k 6 --auto_select_best_patch \
      --zoom_out_factor 6 --marker_style ring --zarr_path data/raw/UC6_I.zarr \
      --out_path results/figures/k6_neighbors_UC6_zoom.pdf
"""
import _repo  # noqa: F401

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from mosaic.pipeline import DEFAULT_DATA_ROOT, IMAGE_PATHS  # noqa: E402


@dataclass
class CellRec:
    path: str
    cell_type: str
    cell_id: int
    cx: float
    cy: float


def resolve_patch_path(manifest_path: str, p: str) -> str:
    p = p.replace("\\", "/")
    if os.path.isabs(p):
        return p
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    rest = p.split("/", 1)[1] if "/" in p else p
    return os.path.normpath(os.path.join(manifest_dir, rest))


def load_records(manifest_path: str) -> List[CellRec]:
    out: List[CellRec] = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"path", "cell_type", "cell_id", "cx", "cy"}.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        for row in reader:
            out.append(CellRec(
                path=resolve_patch_path(manifest_path, str(row["path"])),
                cell_type=str(row["cell_type"]), cell_id=int(float(row["cell_id"])),
                cx=float(row["cx"]), cy=float(row["cy"]),
            ))
    if not out:
        raise ValueError("empty manifest")
    return out


def choose_center_index(records: List[CellRec], cell_id: Optional[int], seed: int) -> int:
    if cell_id is not None:
        for i, r in enumerate(records):
            if r.cell_id == cell_id:
                return i
        raise ValueError(f"cell_id {cell_id} not found in the manifest")
    random.seed(seed)
    return random.randrange(len(records))


def compute_knn_for_center(records: List[CellRec], center_idx: int, k: int) -> np.ndarray:
    coords = np.array([[r.cx, r.cy] for r in records], dtype=np.float32)
    k_eff = min(max(k, 1), len(records) - 1)
    diff = coords - coords[center_idx][None, :]
    dist2 = np.sum(diff * diff, axis=1)
    dist2[center_idx] = np.inf
    idx = np.argpartition(dist2, kth=k_eff - 1)[:k_eff]
    return idx[np.argsort(dist2[idx])]


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.nanpercentile(arr, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(arr, dtype=np.uint8)
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255.0 + 0.5).astype(np.uint8)


def _arr_to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        g = _to_uint8(arr)
        return np.stack([g, g, g], axis=-1)
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4):  # CHW
            x = arr[:3] if arr.shape[0] > 1 else np.repeat(arr, 3, axis=0)
            return _to_uint8(np.moveaxis(x, 0, -1))
        if arr.shape[-1] in (1, 3, 4):  # HWC
            x = arr[..., :3] if arr.shape[-1] > 1 else np.repeat(arr, 3, axis=-1)
            return _to_uint8(x)
    raise ValueError(f"Unsupported image layout: {arr.shape}")


def load_zoomed_context_from_zarr(zarr_path: str, center_cx: float, center_cy: float,
                                  zoom_out_factor: float, output_size: int) -> Tuple[Image.Image, float]:
    import zarr

    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr store not found for zoom-out: {zarr_path}")
    zg = zarr.open_group(zarr_path, mode="r")
    img_arr = zg["images/HE_original/0"]  # C, H, W
    _, h, w = img_arr.shape
    half_side = int(round((output_size * zoom_out_factor) / 2.0))
    cx, cy = int(round(center_cx)), int(round(center_cy))
    x0, x1 = max(0, cx - half_side), min(w, cx + half_side)
    y0, y1 = max(0, cy - half_side), min(h, cy + half_side)
    rgb = _arr_to_rgb(img_arr[:, y0:y1, x0:x1])
    pil = Image.fromarray(rgb, mode="RGB").resize((output_size, output_size), Image.Resampling.LANCZOS)
    return pil, float(half_side)


def _patch_focus_score(path: str) -> float:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return -1.0
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    h, w = gray.shape
    c = gray[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
    sharp = float(np.diff(c, axis=1).var() + np.diff(c, axis=0).var())
    white_ratio = float((c > 0.92).mean())  # penalise washed-out / background patches
    return sharp - 0.4 * white_ratio


def choose_best_center_index(records: List[CellRec], seed: int, candidates: int) -> int:
    rnd = random.Random(seed)
    n = len(records)
    cands = min(max(candidates, 20), n)
    idxs = rnd.sample(range(n), cands) if cands < n else list(range(n))
    scores = [(_patch_focus_score(records[i].path), i) for i in idxs]
    return max(scores)[1]


def neighbor_points_on_patch(records: List[CellRec], center_idx: int, neigh_ids: np.ndarray,
                             img_size: int, coord_radius: Optional[float] = None) -> np.ndarray:
    center = records[center_idx]
    dx = np.array([records[int(i)].cx - center.cx for i in neigh_ids], dtype=np.float32)
    dy = np.array([records[int(i)].cy - center.cy for i in neigh_ids], dtype=np.float32)
    center_px = img_size / 2.0
    if coord_radius is not None and coord_radius > 0:
        scale, radius = float(coord_radius), img_size / 2.0  # true geometry of the zoomed-out crop
    else:
        scale = max(float(np.max(np.abs(dx))) if dx.size else 1.0, float(np.max(np.abs(dy))) if dy.size else 1.0, 1.0)
        radius = img_size * 0.35  # local rescaling so that all points stay inside the patch
    x = np.clip(center_px + (dx / scale) * radius, 4, img_size - 4)
    y = np.clip(center_px + (dy / scale) * radius, 4, img_size - 4)
    return np.stack([x, y], axis=1)


def make_figure(img: Image.Image, pts: np.ndarray, out_path: str, dpi: int = 600, upscale: int = 4,
                marker_size_center: int = 60, marker_size_neigh: int = 48, marker_style: str = "dot",
                ring_radius_frac: float = 0.06, draw_edges: bool = True, annotate: bool = True):
    if upscale > 1:
        img = img.resize((img.size[0] * upscale, img.size[1] * upscale), Image.Resampling.LANCZOS)
        if pts.size > 0:
            pts = pts * upscale
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=dpi)
    for ax in axes:
        ax.imshow(arr)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    color_center, color_neigh = "#1f5fff", "#12b83a"
    if marker_style == "ring":
        from matplotlib.patches import Circle

        radius = ring_radius_frac * min(w, h)
        lw = max(1.2, radius * 0.10)

        def _ring(ax, x, y, color):
            ax.add_patch(Circle((x, y), radius=radius, facecolor="none", edgecolor=color, linewidth=lw))

        _ring(axes[0], cx, cy, color_center)
        if draw_edges and pts.size > 0:
            for px, py in pts:
                axes[1].plot([cx, px], [cy, py], color="#ffd21f", linewidth=max(0.8, lw * 0.6),
                             alpha=0.9, zorder=2, solid_capstyle="round")
        _ring(axes[1], cx, cy, color_center)
        for i, (px, py) in enumerate(pts):
            _ring(axes[1], px, py, color_neigh)
            if annotate:
                axes[1].text(px, py, str(i + 1), color="white", fontsize=7, ha="center", va="center",
                             fontweight="bold", zorder=5)
        if annotate:
            axes[1].text(cx, cy, "C", color="white", fontsize=8, ha="center", va="center", fontweight="bold", zorder=5)
    else:
        axes[0].scatter([cx], [cy], s=marker_size_center, c=color_center)
        axes[1].scatter([cx], [cy], s=marker_size_center, c=color_center)
        if pts.size > 0:
            axes[1].scatter(pts[:, 0], pts[:, 1], s=marker_size_neigh, c="#22e85a")

    axes[0].text(0.5, -0.08, "(a)", transform=axes[0].transAxes, ha="center", va="top", fontsize=12, family="serif")
    axes[1].text(0.5, -0.08, "(b)", transform=axes[1].transAxes, ha="center", va="top", fontsize=12, family="serif")
    plt.tight_layout(w_pad=2.0)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Target cell + K neighbours figure")
    p.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--image_key", default="UC6", choices=sorted(IMAGE_PATHS.keys()))
    p.add_argument("--manifest_path", default=None, help="Optional explicit manifest.csv")
    p.add_argument("--zarr_path", default=None, help="Raw Zarr store, required when --zoom_out_factor > 1")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--cell_id", type=int, default=None, help="Target cell (default: random by --seed)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--auto_select_best_patch", action="store_true", help="Pick the sharpest of --auto_candidates random patches")
    p.add_argument("--auto_candidates", type=int, default=500)
    p.add_argument("--out_path", default="results/figures/neighborhood.pdf")
    p.add_argument("--zoom_out_factor", type=float, default=1.0, help="> 1 reads a wider field from --zarr_path")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--marker_size_center", type=int, default=60)
    p.add_argument("--marker_size_neigh", type=int, default=48)
    p.add_argument("--marker_style", default="dot", choices=["dot", "ring"])
    p.add_argument("--ring_radius_frac", type=float, default=0.06)
    p.add_argument("--no_edges", action="store_true")
    p.add_argument("--no_annotate", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    manifest_path = args.manifest_path or os.path.join(args.data_root, IMAGE_PATHS[args.image_key]["manifest"])
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    records = load_records(manifest_path)
    if args.cell_id is not None:
        center_idx = choose_center_index(records, args.cell_id, args.seed)
    elif args.auto_select_best_patch:
        center_idx = choose_best_center_index(records, args.seed, args.auto_candidates)
    else:
        center_idx = choose_center_index(records, None, args.seed)
    neigh_ids = compute_knn_for_center(records, center_idx, args.k)

    coord_radius = None
    if args.zoom_out_factor > 1.0:
        if not args.zarr_path:
            raise ValueError("--zoom_out_factor > 1 requires --zarr_path")
        img, coord_radius = load_zoomed_context_from_zarr(
            args.zarr_path, records[center_idx].cx, records[center_idx].cy, args.zoom_out_factor, output_size=128)
    else:
        img = Image.open(records[center_idx].path).convert("RGB")
    pts = neighbor_points_on_patch(records, center_idx, neigh_ids, img.size[0], coord_radius=coord_radius)
    make_figure(img, pts, args.out_path, dpi=args.dpi, upscale=args.upscale,
                marker_size_center=args.marker_size_center, marker_size_neigh=args.marker_size_neigh,
                marker_style=args.marker_style, ring_radius_frac=args.ring_radius_frac,
                draw_edges=not args.no_edges, annotate=not args.no_annotate)
    print(f"[OK] Figure saved to {os.path.abspath(args.out_path)}")
    print(f"[INFO] image={args.image_key} | cell_id={records[center_idx].cell_id} | k={len(neigh_ids)}")


if __name__ == "__main__":
    main()
