"""Shared scoring for the multi-model zero-shot evaluation suite.

All eval_*.py scripts MUST use these functions to guarantee identical
Softmax+Argmax dustbin resolution, identical Hungarian matching, and
identical mIoU / ARI / Coverage accounting across model families.

No torch / no cv2 / no SAM-specific imports here — every scorer works on
numpy arrays produced upstream. That keeps this module importable from
any environment (HF GroundingDINO, SAM3, LISA, Qwen, ...).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# --------------------------------------------------------------------- #
# Stage 3 — Softmax + Argmax with Dustbin Channel                        #
# --------------------------------------------------------------------- #

def resolve_conflicts_softmax_argmax(
    logits: np.ndarray, dustbin_logit: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Argmax over (K+1)-channel stack: K texture logits + 1 static dustbin.

    Args:
        logits: (K, H, W) raw per-channel logits.
        dustbin_logit: static logit for the background channel. 0.0 means a
            pixel goes to background unless some texture beat it.

    Returns:
        argmax_map: (H, W) int32 in {0..K-1, K=background}.
        probs:      (K+1, H, W) softmaxed probs.
    """
    if logits.ndim != 3:
        raise ValueError(f"expected (K,H,W) logits, got {logits.shape}")
    K, H, W = logits.shape
    dustbin = np.full((1, H, W), dustbin_logit, dtype=logits.dtype)
    stacked = np.concatenate([logits, dustbin], axis=0)        # (K+1, H, W)
    stacked = stacked - stacked.max(axis=0, keepdims=True)
    exp = np.exp(stacked)
    probs = exp / exp.sum(axis=0, keepdims=True)
    return probs.argmax(axis=0).astype(np.int32), probs


# --------------------------------------------------------------------- #
# Per-pair similarity                                                    #
# --------------------------------------------------------------------- #

def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pb, gb = pred > 0.5, gt > 0.5
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pb, gb = pred > 0.5, gt > 0.5
    inter = np.logical_and(pb, gb).sum()
    total = pb.sum() + gb.sum()
    return 1.0 if total == 0 else float(2.0 * inter) / float(total)


# --------------------------------------------------------------------- #
# Stage 4 — Hungarian matching                                           #
# --------------------------------------------------------------------- #

def hungarian_match(
    preds: List[np.ndarray], gts: List[np.ndarray],
) -> dict:
    """Optimal K×M assignment between predicted and GT binary masks.

    Unmatched predictions / GTs are charged IoU = 0, so
        panoptic_iou = sum_matched(IoU) / max(K, M)
    penalises over- and under-prediction symmetrically.
    """
    from scipy.optimize import linear_sum_assignment

    K, M = len(preds), len(gts)
    iou_mat = np.zeros((max(K, 1), max(M, 1)), dtype=np.float32)
    dice_mat = np.zeros_like(iou_mat)

    for i in range(K):
        for j in range(M):
            iou_mat[i, j] = _iou(preds[i], gts[j])
            dice_mat[i, j] = _dice(preds[i], gts[j])

    if K == 0 or M == 0:
        return {
            "iou_matrix": iou_mat.tolist(),
            "assignment": [],
            "matched_ious": [], "matched_dices": [],
            "panoptic_iou": 0.0, "panoptic_dice": 0.0,
            "matched_mean_iou": 0.0, "matched_mean_dice": 0.0,
            "n_preds": K, "n_gts": M,
        }

    cost = 1.0 - iou_mat[:K, :M]
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_ious = [float(iou_mat[r, c]) for r, c in zip(row_ind, col_ind)]
    matched_dices = [float(dice_mat[r, c]) for r, c in zip(row_ind, col_ind)]

    denom = max(K, M)
    return {
        "iou_matrix": iou_mat[:K, :M].tolist(),
        "assignment": [(int(r), int(c)) for r, c in zip(row_ind, col_ind)],
        "matched_ious": matched_ious,
        "matched_dices": matched_dices,
        "panoptic_iou": float(sum(matched_ious) / denom),
        "panoptic_dice": float(sum(matched_dices) / denom),
        "matched_mean_iou": float(np.mean(matched_ious)),
        "matched_mean_dice": float(np.mean(matched_dices)),
        "n_preds": K, "n_gts": M,
    }


# --------------------------------------------------------------------- #
# ARI (permutation-invariant partition agreement)                        #
# --------------------------------------------------------------------- #

def adjusted_rand_index(argmax_map: np.ndarray, gt_masks: List[np.ndarray]) -> float:
    """Adjusted Rand Index between predicted partition and GT partition.

    GT class map: label i+1 for each GT mask, 0 = background.
    Pred class map: argmax_map as-is (0..K-1 = textures, K = dustbin).
    ARI is permutation-invariant so Hungarian alignment is not required.
    """
    from sklearn.metrics import adjusted_rand_score

    if not gt_masks:
        return 0.0
    H, W = argmax_map.shape
    gt_class = np.zeros((H, W), dtype=np.int32)
    for j, m in enumerate(gt_masks):
        gt_class[m > 0.5] = j + 1
    return float(adjusted_rand_score(gt_class.ravel(), argmax_map.ravel()))


# --------------------------------------------------------------------- #
# One-stop per-sample scorer                                             #
# --------------------------------------------------------------------- #

def compute_sample_metrics(
    logits: np.ndarray, gt_masks: List[np.ndarray],
    dustbin_logit: float = 0.0,
) -> dict:
    """Run the full Stage-3 + Stage-4 + ARI + Coverage pipeline on one sample.

    Args:
        logits:   (K, H, W) per-channel pred logits, already at GT resolution.
        gt_masks: list of M binary GT masks, each (H, W), float in [0,1].
        dustbin_logit: static logit for background channel (default 0.0).

    Returns:
        dict with:
            argmax_map, probs  — the raw partition
            pred_masks         — list of K binary (H,W) masks (argmax==i)
            panoptic_iou, panoptic_dice
            matched_mean_iou, matched_mean_dice
            assignment, iou_matrix
            ari
            bg_coverage  — fraction of pixels sent to dustbin
            n_pred, n_gt
    """
    argmax_map, probs = resolve_conflicts_softmax_argmax(
        logits, dustbin_logit=dustbin_logit,
    )
    K = logits.shape[0]
    pred_masks = [(argmax_map == i).astype(np.float32) for i in range(K)]

    match = hungarian_match(pred_masks, gt_masks)
    ari = adjusted_rand_index(argmax_map, gt_masks)
    bg_coverage = float((argmax_map == K).mean())

    return {
        "argmax_map": argmax_map,
        "probs": probs,
        "pred_masks": pred_masks,
        "panoptic_iou": match["panoptic_iou"],
        "panoptic_dice": match["panoptic_dice"],
        "matched_mean_iou": match["matched_mean_iou"],
        "matched_mean_dice": match["matched_mean_dice"],
        "assignment": match["assignment"],
        "iou_matrix": match["iou_matrix"],
        "ari": ari,
        "bg_coverage": bg_coverage,
        "n_pred": K,
        "n_gt": len(gt_masks),
    }


# --------------------------------------------------------------------- #
# Dataset-level aggregation                                              #
# --------------------------------------------------------------------- #

def masks_to_logits(
    masks: List[np.ndarray],
    target_hw: tuple,
    missing_logit: float,
    k_target: int,
    pos_logit: float = 5.0,
) -> np.ndarray:
    """Convert a list of soft binary masks into a (K, H, W) logit stack.

    - Each mask m (float in [0,1]) is mapped to pos_logit * (2*m - 1) so that
      m=0 → -pos_logit, m=1 → +pos_logit, m=0.5 → 0. This produces well-
      behaved Softmax inputs when stacked across channels.
    - Missing channels (fewer masks than k_target) are padded with the
      missing_logit plane so they collapse to the dustbin and the Hungarian
      matcher charges 0 IoU for the missed GT.
    - Masks are resized to target_hw with bilinear interpolation (cv2).

    Used by every family that comes out of a mask-generating decoder:
    vlm_end2end (LISA/Sa2VA), sam3_vanilla proposal-mode, etc.
    """
    import cv2   # local import keeps metrics_utils dependency-light at top
    gt_h, gt_w = target_hw
    out = []
    for m in masks:
        if m.shape != (gt_h, gt_w):
            m = cv2.resize(m.astype(np.float32), (gt_w, gt_h),
                           interpolation=cv2.INTER_LINEAR)
        out.append(pos_logit * (2.0 * m - 1.0))
    for _ in range(k_target - len(out)):
        out.append(np.full((gt_h, gt_w), missing_logit, dtype=np.float32))
    return np.stack(out, axis=0).astype(np.float32)


def aggregate_run(per_sample_results: List[dict]) -> dict:
    """Aggregate a list of per-sample result dicts into a run-level summary.

    Only entries with status == 'ok' contribute. Reports mIoU / mDice / ARI /
    Coverage + per-K_GT breakdown of pIoU.
    """
    ok = [r for r in per_sample_results if r.get("status") == "ok"]
    if not ok:
        return {"n_total": len(per_sample_results), "n_ok": 0}

    summary = {
        "n_total": len(per_sample_results),
        "n_ok": len(ok),
        "panoptic_iou": float(np.mean([r["panoptic_iou"] for r in ok])),
        "panoptic_dice": float(np.mean([r["panoptic_dice"] for r in ok])),
        "matched_mean_iou": float(np.mean([r["matched_mean_iou"] for r in ok])),
        "matched_mean_dice": float(np.mean([r["matched_mean_dice"] for r in ok])),
        "mean_ari": float(np.mean([r["ari"] for r in ok])),
        "mean_bg_coverage": float(np.mean([r["bg_coverage"] for r in ok])),
        "mean_n_pred": float(np.mean([r["n_pred"] for r in ok])),
        "mean_n_gt": float(np.mean([r["n_gt"] for r in ok])),
    }
    by_m: dict = {}
    for r in ok:
        by_m.setdefault(r["n_gt"], []).append(r["panoptic_iou"])
    summary["panoptic_iou_by_n_gt"] = {
        str(k): float(np.mean(v)) for k, v in sorted(by_m.items())
    }
    return summary
