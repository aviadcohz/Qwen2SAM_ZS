"""Shared dataset I/O + SAM3 preprocessing for the eval suite.

Import from any eval_*.py that consumes images / masks / SAM3 tensors.
Kept dependency-light (numpy, cv2, torch) so it loads in every model env.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch


SAM3_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SAM3_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SAM3_SIZE = 1008


# --------------------------------------------------------------------- #
# Dataset loaders (unified schema only — all eval-suite datasets use it) #
# --------------------------------------------------------------------- #

def load_samples(metadata_path: str, limit: int | None = None,
                 wanted_ids: set | None = None) -> List[dict]:
    with open(metadata_path) as f:
        raw = json.load(f)

    samples = []
    for entry in raw:
        sid = entry.get("id") or entry.get("source_image_id") \
            or Path(entry["image_path"]).stem
        if wanted_ids and sid not in wanted_ids:
            continue
        textures = entry.get("textures", [])
        samples.append({
            "id": sid,
            "image_path": entry["image_path"],
            "gt_masks": [t["mask_path"] for t in textures],
            "gt_descs": [t.get("description", "") for t in textures],
        })
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"mask not found: {path}")
    return (m.astype(np.float32) / 255.0 > 0.5).astype(np.float32)


def load_gt_masks(mask_paths: List[str], min_area_frac: float = 0.0):
    """Return (masks, keep_idx). Empty and too-small masks are dropped."""
    masks, keep_idx = [], []
    for i, p in enumerate(mask_paths):
        try:
            m = _load_mask(p)
        except FileNotFoundError:
            continue
        frac = float(m.mean())
        if frac <= 0.0 or frac < min_area_frac:
            continue
        masks.append(m)
        keep_idx.append(i)
    return masks, keep_idx


def load_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"image not found: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------- #
# SAM3 preprocessing (1008×1008, mean/std = 0.5)                         #
# --------------------------------------------------------------------- #

def preprocess_image_for_sam3(image_rgb: np.ndarray,
                              size: int = SAM3_SIZE) -> torch.Tensor:
    if image_rgb.shape[0] != size or image_rgb.shape[1] != size:
        image_rgb = cv2.resize(image_rgb, (size, size),
                               interpolation=cv2.INTER_LINEAR)
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - SAM3_MEAN) / SAM3_STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)
