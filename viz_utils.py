"""Shared per-sample visualization for every eval_*.py script.

Produces one PNG per sample in the SAME layout as the original
evaluate_zero_shot_pipeline.py figures:

    • Top row  — image, GT overlay, Pred overlay, metric banner.
    • N rows  — one per predicted channel: logit heatmap, pred mask,
                matched GT mask (or 'unmatched'), contour overlay.

Factored out so sam3_vanilla / grounded_sam3 / vlm_end2end / qwen2sam_zs
all produce comparable figures — only the model-family header and the
per-prediction `descs` text change.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np


# Distinct colors for up to 6 textures + dustbin (final slot).
PALETTE = np.array([
    [230,  55,  55],   # 0 red
    [ 55, 120, 230],   # 1 blue
    [ 80, 200,  90],   # 2 green
    [240, 180,  40],   # 3 amber
    [180,  90, 220],   # 4 purple
    [255, 120, 180],   # 5 pink
    [ 90,  90,  90],   # N = background (dustbin)
], dtype=np.uint8)


def _class_map_to_rgb(class_map: np.ndarray, n_tex: int) -> np.ndarray:
    palette = PALETTE.copy()
    palette[n_tex] = PALETTE[-1]    # move dustbin colour to the N-th slot
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_tex + 1):
        rgb[class_map == c] = palette[c]
    return rgb


def visualize_sample(
    image_rgb: np.ndarray,
    gt_masks: List[np.ndarray],
    logits: np.ndarray,                 # (N, H, W)
    argmax_map: np.ndarray,             # (H, W)
    descs: List[str],
    gt_descs: List[str],
    match_info: dict,
    sample_id: str,
    dataset_name: str,
    save_path: Path,
    model_family: str = "zero-shot",
    logit_title: str = "Logits",
    pred_row_header: str = "pred",      # shown in each per-prediction row
) -> None:
    """Write one PNG showing the full sample → prediction → match breakdown.

    Accepts already-computed `logits` / `argmax_map` / `match_info` so every
    eval_*.py script can call this with its own numbers — the function
    itself does no inference.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    N = logits.shape[0]
    M = len(gt_masks)
    H, W = argmax_map.shape
    img_rs = cv2.resize(image_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

    # GT class map: index by GT mask order (0..M-1), background last.
    gt_class = np.full((H, W), M, dtype=np.int32)
    for j, m in enumerate(gt_masks):
        gt_class[m > 0.5] = j
    gt_rgb = _class_map_to_rgb(gt_class, M)
    gt_overlay = (0.45 * gt_rgb + 0.55 * img_rs).astype(np.uint8)

    pred_rgb = _class_map_to_rgb(argmax_map, N)
    pred_overlay = (0.55 * pred_rgb + 0.45 * img_rs).astype(np.uint8)

    n_pred_rows = max(N, 1)
    fig_rows = 1 + n_pred_rows
    fig, axes = plt.subplots(fig_rows, 4, figsize=(16, 3.6 * fig_rows))
    if fig_rows == 1:
        axes = axes[None, :]

    axes[0, 0].imshow(img_rs)
    axes[0, 0].set_title(f"Image ({sample_id})")
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title(f"GT overlay (M={M})")
    axes[0, 2].imshow(pred_overlay)
    axes[0, 2].set_title(
        f"Pred overlay (N={N})  pIoU={match_info['panoptic_iou']:.3f}"
    )

    axes[0, 3].axis("off")
    gt_desc_lines = "\n".join(
        f"  [{j}] {(d or '').strip()[:78]}" for j, d in enumerate(gt_descs[:M])
    ) or "  (no GT descriptions)"
    pred_desc_lines = "\n".join(
        f"  [{i}] {(d or '').strip()[:78]}" for i, d in enumerate(descs[:N])
    ) or "  (no predictions)"
    info = (
        f"Model   : {model_family}\n"
        f"Dataset : {dataset_name}\n"
        f"Sample  : {sample_id}\n\n"
        f"{model_family} produced {N} prediction(s):\n{pred_desc_lines}\n\n"
        f"GT contains {M} region(s):\n{gt_desc_lines}\n\n"
        f"Panoptic IoU  : {match_info['panoptic_iou']:.4f}\n"
        f"Panoptic Dice : {match_info['panoptic_dice']:.4f}\n"
        f"Matched mIoU  : {match_info['matched_mean_iou']:.4f}\n"
        f"ARI           : {match_info.get('ari', float('nan')):.4f}\n"
        f"Assignment    : {match_info['assignment']}\n"
        f"BG coverage   : {(argmax_map == N).mean():.3f}"
    )
    axes[0, 3].text(
        0.01, 0.98, info,
        transform=axes[0, 3].transAxes,
        fontsize=8.5, va="top", family="monospace",
    )

    pred_to_gt = {int(r): int(c) for r, c in match_info["assignment"]}

    for i in range(N):
        ax_h, ax_p, ax_m, ax_g = axes[1 + i]

        im_h = ax_h.imshow(logits[i], cmap="magma")
        ax_h.set_title(f"{logit_title} — {pred_row_header} {i}")
        plt.colorbar(im_h, ax=ax_h, fraction=0.046, pad=0.04)

        pred_mask = (argmax_map == i).astype(np.float32)
        ax_p.imshow(pred_mask, cmap="Greens", vmin=0, vmax=1)
        ax_p.set_title(f"Pred mask {i}  cov={pred_mask.mean():.2f}")

        if i in pred_to_gt:
            gt_j = pred_to_gt[i]
            iou_ij = match_info["iou_matrix"][i][gt_j]
            gt_mask = gt_masks[gt_j]
            ax_m.imshow(gt_mask, cmap="Blues", vmin=0, vmax=1)
            ax_m.set_title(f"Matched GT [{gt_j}]  IoU={iou_ij:.3f}")
            ax_g.imshow(img_rs)
            ax_g.contour(pred_mask, levels=[0.5], colors="red", linewidths=1.5)
            ax_g.contour(gt_mask,  levels=[0.5], colors="cyan", linewidths=1.5)
            ax_g.set_title(
                f"Overlay — pred (red) vs GT (cyan)\n"
                f"{(descs[i] if i < len(descs) else '')[:60]}"
            )
        else:
            ax_m.axis("off")
            ax_m.text(0.5, 0.5, "(unmatched)",
                      transform=ax_m.transAxes, ha="center", va="center")
            ax_g.imshow(img_rs)
            ax_g.contour(pred_mask, levels=[0.5], colors="red", linewidths=1.5)
            ax_g.set_title(
                f"Overlay — pred (red) only\n"
                f"{(descs[i] if i < len(descs) else '')[:60]}"
            )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    legend_elems = [
        Patch(facecolor=PALETTE[k] / 255, label=f"pred {k}")
        for k in range(min(N, 6))
    ] + [
        Patch(facecolor=PALETTE[-1] / 255, label="background (dustbin)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center",
               ncol=min(7, N + 1), bbox_to_anchor=(0.5, 0.0), fontsize=9)

    fig.suptitle(
        f"{model_family} — {dataset_name} — {sample_id}  (N={N}, M={M})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
