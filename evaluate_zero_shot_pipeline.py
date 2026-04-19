"""
Qwen2SAM_ZS — Zero-Shot VLM → SAM3 Per-Sample-K Texture Segmentation Baseline.

The target region count K is set PER SAMPLE from the GT:
  RWTD              → K = 2 always.
  ADE20k_DeTexture  → K = number of GT textures for that image (1..6).

Pipeline (no training, no custom bridges):
  Stage 1  Qwen3-VL-8B is prompted for EXACTLY K distinct surface regions and
           emits K phrases TEXTURE_1 … TEXTURE_K.
  Stage 2  Each description is encoded by SAM3's native VETextEncoder and
           passed through the Multimodal Decoder + Semantic Seg Head →
           K continuous per-pixel logit heatmaps.
  Stage 3  Stack (K+1, H, W) logits — K textures plus a static "dustbin"
           (background) channel — Softmax, Argmax → discrete (K+1)-class map.
  Stage 4  Hungarian matching (K × M) via scipy.linear_sum_assignment on the
           IoU cost matrix. Since we ask Qwen for exactly K = K_GT, K == M and
           the matching is a square permutation.

Datasets supported out-of-the-box:
  rwtd              → /home/aviad/datasets/RWTD/metadata.json
  rwtd_phase1       → /home/aviad/RWTD/metadata_phase1.json             (legacy)
  ade20k_detexture  → /home/aviad/datasets/ADE20k_DeTexture/metadata.json
  ade20k_textured   → /home/aviad/datasets/ADE20K_textured_images/metadata.json
  custom            → pass --metadata <path> for a unified-schema file.

Run:
  python evaluate_zero_shot_pipeline.py --dataset rwtd
  python evaluate_zero_shot_pipeline.py --dataset ade20k_detexture --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# --------------------------------------------------------------------------- #
# Path setup                                                                   #
# --------------------------------------------------------------------------- #
# SAM3 repo location. Override with the SAM3_ROOT env var if you clone the
# repo somewhere other than ~/sam3. The directory must contain the `sam3`
# python package with `model_builder.py`.
import os as _os

_SAM3_ROOT = Path(_os.environ.get("SAM3_ROOT", "/home/aviad/sam3"))
if str(_SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM3_ROOT))


# --------------------------------------------------------------------------- #
# Prompts                                                                      #
# --------------------------------------------------------------------------- #

MAX_TEXTURES = 10  # absolute upper bound on the regex parser

SYSTEM_PROMPT = (
    "You analyze surface textures in images. Always respond in the exact "
    "format requested, with no extra text."
)


def build_user_prompt(k: int) -> str:
    """Build the Qwen user prompt parameterized by the exact number of regions K."""
    if k == 2:
        # Preserve the original user-specified RWTD wording for K=2 samples.
        return (
            "This image contains exactly TWO main visually distinct regions "
            "separated by a boundary (for example, a prominent foreground "
            "object and its background, or two contrasting materials).\n\n"
            "Write a single, highly descriptive phrase (approximately 10-15 "
            "words) for each of the two regions. Include the following "
            "precise information:\n"
            "1. Semantic Name: A natural, common name for the material or object.\n"
            "2. Distinct Visual Features: The core visual attributes like "
            "color, pattern, or texture that strongly contrast with the other "
            "region.\n"
            "3. Spatial Context: A brief note on its general position "
            "(e.g., 'foreground', 'background', 'top-left').\n\n"
            "IMPORTANT: Describe the ENTIRE region as a collective group, NOT "
            "individual objects within it. Think of each region as a "
            "surface/area, not as a single object.\n\n"
            "Format your response exactly like this:\n"
            "TEXTURE_1: Texture of <description>\n"
            "TEXTURE_2: Texture of <description>"
        )

    # Variable-K prompt (ADE20k_DeTexture and friends).
    k_word = {
        1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR",
        5: "FIVE", 6: "SIX", 7: "SEVEN", 8: "EIGHT",
    }.get(k, str(k))
    format_lines = "\n".join(
        f"TEXTURE_{i}: Texture of <description>" for i in range(1, k + 1)
    )
    return (
        f"This image contains EXACTLY {k_word} ({k}) visually distinct "
        "surface regions separated by boundaries (e.g. sky, water, grass, "
        "foliage, road, sand, building, stone, fabric).\n\n"
        f"Identify all {k} regions. For each, write a single highly "
        "descriptive phrase (approximately 10-15 words) containing:\n"
        "1. Semantic Name: a common name for the material or object class.\n"
        "2. Distinct Visual Features: core attributes — color, pattern, "
        "texture — that contrast with the other regions.\n"
        "3. Spatial Context: a brief note on position (e.g. 'upper "
        "background', 'lower foreground', 'center').\n\n"
        "IMPORTANT RULES:\n"
        "• Describe each region as a collective surface/area, NOT individual "
        "objects within it.\n"
        f"• Output EXACTLY {k} TEXTURE_i lines, no more, no less. Use "
        "consecutive indexing starting at 1.\n"
        "• Each phrase must be different and describe a visually distinct "
        "region.\n\n"
        "Format your response exactly like this:\n"
        f"{format_lines}"
    )


# --------------------------------------------------------------------------- #
# Dataset registry                                                             #
# --------------------------------------------------------------------------- #

DATASET_REGISTRY = {
    "rwtd": {
        "metadata": "/home/aviad/datasets/RWTD/metadata.json",
        "schema": "unified",
    },
    "rwtd_phase1": {
        "metadata": "/home/aviad/RWTD/metadata_phase1.json",
        "schema": "phase1",
    },
    "ade20k_detexture": {
        "metadata": "/home/aviad/datasets/ADE20k_DeTexture/metadata.json",
        "schema": "unified",
    },
    "ade20k_textured": {
        "metadata": "/home/aviad/datasets/ADE20K_textured_images/metadata.json",
        "schema": "unified",
    },
}


def _load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"mask not found: {path}")
    return (m.astype(np.float32) / 255.0 > 0.5).astype(np.float32)


def load_samples(
    schema: str,
    metadata_path: str,
    limit: int | None = None,
    wanted_ids: set | None = None,
) -> List[dict]:
    """Return a list of unified samples with variable-length gt_masks."""
    with open(metadata_path) as f:
        raw = json.load(f)

    samples = []
    for entry in raw:
        if schema == "phase1":
            sid = entry.get("crop_name", Path(entry["image_path"]).stem)
            if wanted_ids and sid not in wanted_ids:
                continue
            samples.append({
                "id": sid,
                "image_path": entry["image_path"],
                "gt_masks": [entry["mask_a_path"], entry["mask_b_path"]],
                "gt_descs": [
                    entry.get("texture_a", ""),
                    entry.get("texture_b", ""),
                ],
            })
        elif schema == "unified":
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
        else:
            raise ValueError(f"unknown schema: {schema}")

        if limit is not None and len(samples) >= limit:
            break
    return samples


def load_gt_masks(mask_paths: List[str], min_area_frac: float = 0.0):
    """
    Load all GT masks, optionally dropping those below `min_area_frac` of the
    image. Empty masks are always skipped.
    """
    masks, keep_idx = [], []
    for i, p in enumerate(mask_paths):
        try:
            m = _load_mask(p)
        except FileNotFoundError:
            continue
        frac = float(m.mean())
        if frac <= 0.0:
            continue
        if frac < min_area_frac:
            continue
        masks.append(m)
        keep_idx.append(i)
    return masks, keep_idx


# --------------------------------------------------------------------------- #
# SAM3 preprocessing                                                           #
# --------------------------------------------------------------------------- #

SAM3_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SAM3_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SAM3_SIZE = 1008


def preprocess_image_for_sam3(image_rgb: np.ndarray, size: int = SAM3_SIZE) -> torch.Tensor:
    if image_rgb.shape[0] != size or image_rgb.shape[1] != size:
        image_rgb = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - SAM3_MEAN) / SAM3_STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)


# --------------------------------------------------------------------------- #
# Stage 1 — Qwen3-VL-8B Description Generation (variable N)                    #
# --------------------------------------------------------------------------- #

def load_qwen_vlm(device: torch.device, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    print(f"[Qwen] loading {model_name}...")
    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(model_name, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"[Qwen] ready on {device}")
    return model, processor


@torch.no_grad()
def generate_descriptions(
    qwen, processor, image_pil: Image.Image, device: torch.device,
    k: int, max_new_tokens: int = 700,
) -> str:
    """
    Qwen3-VL inference via the official `qwen_vl_utils.process_vision_info`
    path. On transformers >= 5.x, the bare `{"type": "image"}` placeholder
    pattern can desync the image-grid-thw token count with the actual vision
    tower output and crash CUDA with `vectorized_gather_kernel` OOB.
    Embedding the PIL image into the message and letting `process_vision_info`
    handle it keeps the chat-template expansion and the image features in
    lock-step.
    """
    from qwen_vl_utils import process_vision_info

    user_prompt = build_user_prompt(k)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_pil},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs.pop("token_type_ids", None)
    inputs = {key: v.to(device) for key, v in inputs.items()}
    output_ids = qwen.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
    )
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, input_len:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True)


def parse_descriptions(raw_text: str, max_n: int = MAX_TEXTURES) -> List[str]:
    """Extract TEXTURE_i lines in order, stopping at the first missing index."""
    descs: List[str] = []
    for i in range(1, max_n + 1):
        m = re.search(rf"TEXTURE_{i}\s*:\s*(.+?)(?:\n|$)", raw_text, re.IGNORECASE)
        if not m:
            break
        desc = m.group(1).strip()
        if not desc:
            break
        descs.append(desc)
    return descs


# --------------------------------------------------------------------------- #
# Stage 2 — Native SAM3 Semantic Segmentation                                  #
# --------------------------------------------------------------------------- #

def load_sam3(device: torch.device):
    import sam3 as sam3_module
    from sam3.model_builder import build_sam3_image_model

    bpe_path = str(
        Path(sam3_module.__path__[0]) / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    )
    print("[SAM3] building image model from HuggingFace weights...")
    sam3 = build_sam3_image_model(
        bpe_path=bpe_path,
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=True,
        enable_segmentation=True,
        device=str(device),
    )
    sam3.eval()
    for p in sam3.parameters():
        p.requires_grad = False
    print("[SAM3] ready (frozen)")
    return sam3


def _get_img_feats(sam3, backbone_out: dict, img_ids: torch.Tensor):
    n_levels = sam3.num_feature_levels
    vis_feats = backbone_out["backbone_fpn"][-n_levels:]
    vis_pos_enc = backbone_out["vision_pos_enc"][-n_levels:]
    vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
    img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
    img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
    return img_feats, img_pos_embeds, vis_feat_sizes


@torch.no_grad()
def run_sam3_semantic(sam3, backbone_out, prompt_embed, prompt_mask):
    B = prompt_embed.shape[0]
    device = prompt_embed.device
    img_ids = torch.arange(B, device=device)

    img_feats, img_pos_embeds, vis_feat_sizes = _get_img_feats(
        sam3, backbone_out, img_ids
    )

    prompt = prompt_embed.transpose(0, 1)  # (N, B, 256)
    prompt_pos = torch.zeros_like(prompt)

    memory_dict = sam3.transformer.encoder(
        src=[f.clone() for f in img_feats],
        src_key_padding_mask=None,
        src_pos=[p.clone() for p in img_pos_embeds],
        prompt=prompt,
        prompt_pos=prompt_pos,
        prompt_key_padding_mask=prompt_mask,
        feat_sizes=vis_feat_sizes,
    )
    encoder_hidden_states = memory_dict["memory"]

    seg_head = sam3.segmentation_head
    enc_hs = encoder_hidden_states
    if seg_head.cross_attend_prompt is not None:
        tgt2 = seg_head.cross_attn_norm(enc_hs)
        tgt2 = seg_head.cross_attend_prompt(
            query=tgt2, key=prompt, value=prompt,
            key_padding_mask=prompt_mask,
        )[0]
        enc_hs = tgt2 + enc_hs

    pixel_embed = seg_head._embed_pixels(
        backbone_feats=backbone_out["backbone_fpn"],
        image_ids=img_ids,
        encoder_hidden_states=enc_hs,
    )
    return seg_head.semantic_seg_head(pixel_embed)  # (B, 1, H, W) logits


@torch.no_grad()
def encode_text_prompt(sam3, description: str, device: torch.device):
    text_out = sam3.backbone.forward_text([description], device=device)
    prompt = text_out["language_features"].squeeze(1)
    mask = text_out["language_mask"]
    if mask.ndim == 3:
        mask = mask.squeeze(1)
    return prompt, mask


def sam3_heatmap_for_description(
    sam3, backbone_out: dict, description: str,
    gt_h: int, gt_w: int, device: torch.device,
) -> np.ndarray:
    prompt, mask = encode_text_prompt(sam3, description, device)
    sem_logits = run_sam3_semantic(sam3, backbone_out, prompt, mask)
    if sem_logits.ndim == 4:
        sem_logits = sem_logits[0, 0]
    elif sem_logits.ndim == 3:
        sem_logits = sem_logits[0]
    logit_t = F.interpolate(
        sem_logits[None, None].float(),
        size=(gt_h, gt_w),
        mode="bilinear", align_corners=False,
    ).squeeze()
    return logit_t.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Stage 3 — Softmax + Argmax with Dustbin Channel (N+1 classes)                #
# --------------------------------------------------------------------------- #

def resolve_conflicts_softmax_argmax(
    logits: np.ndarray, dustbin_logit: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    logits: (N, H, W) raw Semantic Seg Head logits (one per description).

    Stacks a dustbin channel at index N (static logit = `dustbin_logit`, 0.0
    corresponds to sigmoid p=0.5 — a pixel is sent to dustbin unless at
    least one texture channel has logit > 0).

    Returns:
        argmax_map: (H, W) int32 with values in {0..N-1, N=background}.
        probs:      (N+1, H, W) softmaxed probabilities.
    """
    if logits.ndim != 3:
        raise ValueError(f"expected (N,H,W) logits, got shape {logits.shape}")
    N, H, W = logits.shape
    dustbin = np.full((1, H, W), dustbin_logit, dtype=logits.dtype)
    stacked = np.concatenate([logits, dustbin], axis=0)  # (N+1, H, W)
    stacked = stacked - stacked.max(axis=0, keepdims=True)
    exp = np.exp(stacked)
    probs = exp / exp.sum(axis=0, keepdims=True)
    return probs.argmax(axis=0).astype(np.int32), probs


# --------------------------------------------------------------------------- #
# Stage 4 — K×M Hungarian Matching with IoU                                    #
# --------------------------------------------------------------------------- #

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


def hungarian_match(
    preds: List[np.ndarray], gts: List[np.ndarray],
) -> dict:
    """
    Optimal Hungarian assignment between K predicted and M GT masks.

    Uses scipy.optimize.linear_sum_assignment on (1 - IoU). Supports K ≠ M;
    unmatched predictions / GTs are counted as IoU = 0 (penalty for over- or
    under-prediction), so the reported metric is:

        panoptic_iou = sum_matched(IoU) / max(K, M)

    Returns dict with:
        iou_matrix: (K, M) raw IoU matrix
        assignment: list of (pred_idx, gt_idx) pairs (length = min(K, M))
        matched_ious, matched_dices: per-pair quality
        panoptic_iou, panoptic_dice: normalized by max(K, M)
        matched_mean_iou, matched_mean_dice: averaged only over matched pairs
        n_preds, n_gts
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
            "matched_ious": [],
            "matched_dices": [],
            "panoptic_iou": 0.0,
            "panoptic_dice": 0.0,
            "matched_mean_iou": 0.0,
            "matched_mean_dice": 0.0,
            "n_preds": K, "n_gts": M,
        }

    cost = 1.0 - iou_mat[:K, :M]
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_ious = [float(iou_mat[r, c]) for r, c in zip(row_ind, col_ind)]
    matched_dices = [float(dice_mat[r, c]) for r, c in zip(row_ind, col_ind)]

    denom = max(K, M)
    panoptic_iou = sum(matched_ious) / denom if denom > 0 else 0.0
    panoptic_dice = sum(matched_dices) / denom if denom > 0 else 0.0
    matched_mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    matched_mean_dice = float(np.mean(matched_dices)) if matched_dices else 0.0

    return {
        "iou_matrix": iou_mat[:K, :M].tolist(),
        "assignment": [(int(r), int(c)) for r, c in zip(row_ind, col_ind)],
        "matched_ious": matched_ious,
        "matched_dices": matched_dices,
        "panoptic_iou": float(panoptic_iou),
        "panoptic_dice": float(panoptic_dice),
        "matched_mean_iou": matched_mean_iou,
        "matched_mean_dice": matched_mean_dice,
        "n_preds": K, "n_gts": M,
    }


# --------------------------------------------------------------------------- #
# Visualization                                                                #
# --------------------------------------------------------------------------- #

# Distinct, visually separable colors for up to 6 textures + dustbin.
PALETTE = np.array([
    [230,  55,  55],   # 0 red
    [ 55, 120, 230],   # 1 blue
    [ 80, 200,  90],   # 2 green
    [240, 180,  40],   # 3 amber
    [180,  90, 220],   # 4 purple
    [255, 120, 180],   # 5 pink
    [ 90,  90,  90],   # N background (dustbin)
], dtype=np.uint8)


def _class_map_to_rgb(class_map: np.ndarray, n_tex: int) -> np.ndarray:
    palette = PALETTE.copy()
    # Class index `n_tex` is always the dustbin; shift last palette slot there.
    palette[n_tex] = PALETTE[-1]
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_tex + 1):
        rgb[class_map == c] = palette[c]
    return rgb


def visualize_sample(
    image_rgb: np.ndarray,
    gt_masks: List[np.ndarray],
    logits: np.ndarray,              # (N, H, W)
    probs: np.ndarray,                # (N+1, H, W)
    argmax_map: np.ndarray,           # (H, W)
    descs: List[str],
    gt_descs: List[str],
    match_info: dict,
    sample_id: str, dataset_name: str,
    save_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    N = logits.shape[0]
    M = len(gt_masks)
    H, W = argmax_map.shape
    img_rs = cv2.resize(image_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

    # GT class map: index by GT mask order (0..M-1), background last.
    gt_class = np.full((H, W), M, dtype=np.int32)  # M = background
    for j, m in enumerate(gt_masks):
        gt_class[m > 0.5] = j
    gt_rgb = _class_map_to_rgb(gt_class, M)
    gt_overlay = (0.45 * gt_rgb + 0.55 * img_rs).astype(np.uint8)

    pred_rgb = _class_map_to_rgb(argmax_map, N)
    pred_overlay = (0.55 * pred_rgb + 0.45 * img_rs).astype(np.uint8)

    # Row layout: top row (image, GT, Pred, info) + per-prediction rows.
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
        f"Pred overlay (N={N})  "
        f"pIoU={match_info['panoptic_iou']:.3f}"
    )

    axes[0, 3].axis("off")
    gt_desc_lines = "\n".join(
        f"  [{j}] {d[:78]}" for j, d in enumerate(gt_descs[:M])
    ) or "  (no GT descriptions)"
    pred_desc_lines = "\n".join(
        f"  [{i}] {d[:78]}" for i, d in enumerate(descs[:N])
    ) or "  (no predictions)"
    info = (
        f"Dataset : {dataset_name}\n"
        f"Sample  : {sample_id}\n\n"
        f"Qwen predicted {N} region(s):\n{pred_desc_lines}\n\n"
        f"GT contains {M} region(s):\n{gt_desc_lines}\n\n"
        f"Panoptic IoU  : {match_info['panoptic_iou']:.4f}\n"
        f"Panoptic Dice : {match_info['panoptic_dice']:.4f}\n"
        f"Matched mIoU  : {match_info['matched_mean_iou']:.4f}\n"
        f"Assignment    : {match_info['assignment']}\n"
        f"BG coverage   : {(argmax_map == N).mean():.3f}"
    )
    axes[0, 3].text(
        0.01, 0.98, info,
        transform=axes[0, 3].transAxes,
        fontsize=8.5, va="top", family="monospace",
    )

    # Map pred → matched GT index for per-row display
    pred_to_gt = {int(r): int(c) for r, c in match_info["assignment"]}

    for i in range(N):
        ax_h = axes[1 + i, 0]
        ax_p = axes[1 + i, 1]
        ax_m = axes[1 + i, 2]
        ax_g = axes[1 + i, 3]

        im_h = ax_h.imshow(logits[i], cmap="magma")
        ax_h.set_title(f"SAM3 logits — pred {i}")
        plt.colorbar(im_h, ax=ax_h, fraction=0.046, pad=0.04)

        pred_mask = (argmax_map == i).astype(np.float32)
        ax_p.imshow(pred_mask, cmap="Greens", vmin=0, vmax=1)
        ax_p.set_title(
            f"Pred mask {i}  cov={pred_mask.mean():.2f}"
        )

        if i in pred_to_gt:
            gt_j = pred_to_gt[i]
            iou_ij = match_info["iou_matrix"][i][gt_j]
            gt_mask = gt_masks[gt_j]
            ax_m.imshow(gt_mask, cmap="Blues", vmin=0, vmax=1)
            ax_m.set_title(f"Matched GT [{gt_j}]  IoU={iou_ij:.3f}")
            ax_g.imshow(img_rs)
            ax_g.contour(pred_mask, levels=[0.5], colors="red", linewidths=1.5)
            ax_g.contour(gt_mask, levels=[0.5], colors="cyan", linewidths=1.5)
            ax_g.set_title(
                f"Overlay — pred (red) vs GT (cyan)\n"
                f"{descs[i][:60]}"
            )
        else:
            ax_m.axis("off")
            ax_m.text(0.5, 0.5, "(unmatched)",
                      transform=ax_m.transAxes, ha="center", va="center")
            ax_g.imshow(img_rs)
            ax_g.contour(pred_mask, levels=[0.5], colors="red", linewidths=1.5)
            ax_g.set_title(
                f"Overlay — pred (red) only\n{descs[i][:60]}"
            )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend
    legend_elems = [
        Patch(facecolor=PALETTE[k] / 255, label=f"pred {k}")
        for k in range(min(N, 6))
    ] + [
        Patch(facecolor=PALETTE[-1] / 255, label="background (dustbin)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center",
               ncol=min(7, N + 1), bbox_to_anchor=(0.5, 0.0), fontsize=9)

    fig.suptitle(
        f"Qwen2SAM_ZS — {dataset_name} — {sample_id}  (N={N}, M={M})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-sample evaluation                                                        #
# --------------------------------------------------------------------------- #

def evaluate_sample(
    sam3, qwen, qwen_proc,
    sample: dict,
    image_size: int,
    dustbin_logit: float,
    device: torch.device,
    vis_dir: Path | None,
    dataset_name: str,
    max_textures: int = MAX_TEXTURES,
    min_gt_area_frac: float = 0.0,
) -> dict:
    sid = sample["id"]
    image_bgr = cv2.imread(sample["image_path"])
    if image_bgr is None:
        return {"id": sid, "status": "image_read_failed"}
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # --- GT masks (all, not just top-2) ---------------------------------- #
    gt_masks, kept_idx = load_gt_masks(
        sample["gt_masks"], min_area_frac=min_gt_area_frac,
    )
    gt_descs = [sample["gt_descs"][i] for i in kept_idx]
    if len(gt_masks) == 0:
        return {"id": sid, "status": "no_gt_masks"}
    gt_h, gt_w = gt_masks[0].shape
    K = len(gt_masks)  # per-sample target count

    # --- Stage 1: Qwen descriptions (exactly K) -------------------------- #
    raw = generate_descriptions(qwen, qwen_proc, image_pil, device, k=K)
    descs = parse_descriptions(raw, max_n=max(max_textures, K))
    if len(descs) == 0:
        return {"id": sid, "status": "qwen_parse_failed", "raw_qwen": raw,
                "n_gt": K}

    # Enforce K: truncate over-production, pad under-production with a
    # generic fallback so Hungarian is K×M square and the final class map
    # carries exactly K + 1 channels (matches the visualization).
    if len(descs) > K:
        descs = descs[:K]
    elif len(descs) < K:
        missing = K - len(descs)
        descs = descs + [f"uniform surface region {i+1}" for i in range(missing)]

    # --- Stage 2: K SAM3 semantic heatmaps ------------------------------- #
    sam_img = preprocess_image_for_sam3(image_rgb, image_size).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        backbone_out = sam3.backbone.forward_image(sam_img)
        backbone_out["img_batch_all_stages"] = sam_img
        logit_list = [
            sam3_heatmap_for_description(sam3, backbone_out, d, gt_h, gt_w, device)
            for d in descs
        ]
    logits = np.stack(logit_list, axis=0)  # (N, H, W)

    # --- Stage 3: Softmax + Argmax with dustbin -------------------------- #
    argmax_map, probs = resolve_conflicts_softmax_argmax(
        logits, dustbin_logit=dustbin_logit
    )
    N = logits.shape[0]
    pred_masks = [(argmax_map == i).astype(np.float32) for i in range(N)]

    # --- Stage 4: Hungarian matching (K × M) ----------------------------- #
    match = hungarian_match(pred_masks, gt_masks)

    # --- Visualization --------------------------------------------------- #
    if vis_dir is not None:
        visualize_sample(
            image_rgb=image_rgb,
            gt_masks=gt_masks,
            logits=logits, probs=probs, argmax_map=argmax_map,
            descs=descs, gt_descs=gt_descs,
            match_info=match,
            sample_id=str(sid), dataset_name=dataset_name,
            save_path=vis_dir / f"{sid}.png",
        )

    return {
        "id": sid,
        "status": "ok",
        "n_pred": N,
        "n_gt": len(gt_masks),
        "descs": descs,
        "gt_descs": gt_descs,
        "assignment": match["assignment"],
        "matched_ious": match["matched_ious"],
        "matched_dices": match["matched_dices"],
        "panoptic_iou": match["panoptic_iou"],
        "panoptic_dice": match["panoptic_dice"],
        "matched_mean_iou": match["matched_mean_iou"],
        "matched_mean_dice": match["matched_mean_dice"],
        "bg_coverage": float((argmax_map == N).mean()),
    }


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Qwen2SAM_ZS — Zero-Shot VLM→SAM3 baseline (variable N textures)"
    )
    parser.add_argument("--dataset", type=str, default="rwtd",
                        choices=list(DATASET_REGISTRY.keys()) + ["custom"])
    parser.add_argument("--metadata", type=str, default=None,
                        help="Override metadata.json path (required for --dataset custom).")
    parser.add_argument("--schema", type=str, default=None,
                        choices=["unified", "phase1"])
    parser.add_argument("--output_dir", type=str,
                        default="/home/aviad/Qwen2SAM_ZS/eval_results")
    parser.add_argument("--image_size", type=int, default=SAM3_SIZE)
    parser.add_argument("--dustbin_logit", type=float, default=0.0,
                        help="Static logit for dustbin channel (0.0 ≡ 0.5 sigmoid prob).")
    parser.add_argument("--max_textures", type=int, default=MAX_TEXTURES,
                        help="Upper bound on regions described by Qwen (default 6).")
    parser.add_argument("--min_gt_area_frac", type=float, default=0.0,
                        help="Drop GT masks occupying less than this fraction of the image.")
    parser.add_argument("--samples", type=str, default=None,
                        help="Comma-separated ids to evaluate.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qwen_model", type=str,
                        default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--no_vis", action="store_true",
                        help="Disable per-sample PNG visualizations.")
    parser.add_argument("--vis_every", type=int, default=1,
                        help="Write a visualization every N samples (default 1 = all).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve dataset
    if args.dataset == "custom":
        if not args.metadata:
            parser.error("--metadata is required when --dataset custom.")
        metadata_path = args.metadata
        schema = args.schema or "unified"
        dataset_name = "custom"
    else:
        spec = DATASET_REGISTRY[args.dataset]
        metadata_path = args.metadata or spec["metadata"]
        schema = args.schema or spec["schema"]
        dataset_name = args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = None if args.no_vis else (output_dir / "vis")
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.samples.split(",")) if args.samples else None
    samples = load_samples(schema, metadata_path, limit=args.limit, wanted_ids=wanted)
    print(f"[data] dataset={dataset_name}  schema={schema}  "
          f"n={len(samples)}  metadata={metadata_path}")

    sam3 = load_sam3(device)
    qwen, qwen_proc = load_qwen_vlm(device, model_name=args.qwen_model)

    results = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        sid = sample["id"]
        should_vis = (vis_dir is not None) and (i % args.vis_every == 0)
        try:
            res = evaluate_sample(
                sam3, qwen, qwen_proc, sample,
                image_size=args.image_size,
                dustbin_logit=args.dustbin_logit,
                device=device,
                vis_dir=(vis_dir if should_vis else None),
                dataset_name=dataset_name,
                max_textures=args.max_textures,
                min_gt_area_frac=args.min_gt_area_frac,
            )
        except Exception as e:  # noqa: BLE001
            res = {"id": sid, "status": "exception", "error": repr(e)}

        results.append(res)
        if res.get("status") == "ok":
            print(f"  [{i+1:4d}/{len(samples)}] {str(sid):>22s}  "
                  f"N={res['n_pred']} M={res['n_gt']}  "
                  f"pIoU={res['panoptic_iou']:.4f}  "
                  f"mIoU={res['matched_mean_iou']:.4f}  "
                  f"bg={res['bg_coverage']:.2f}")
        else:
            print(f"  [{i+1:4d}/{len(samples)}] {str(sid):>22s}  "
                  f"SKIP status={res.get('status')}")

    elapsed = time.time() - t0

    # ---------------- Summary ---------------- #
    ok = [r for r in results if r.get("status") == "ok"]
    summary = {
        "dataset": dataset_name,
        "schema": schema,
        "metadata_path": str(metadata_path),
        "n_total": len(results),
        "n_ok": len(ok),
        "elapsed_seconds": elapsed,
        "dustbin_logit": args.dustbin_logit,
        "max_textures": args.max_textures,
        "image_size": args.image_size,
        "qwen_model": args.qwen_model,
    }
    if ok:
        summary.update({
            "panoptic_iou": float(np.mean([r["panoptic_iou"] for r in ok])),
            "panoptic_dice": float(np.mean([r["panoptic_dice"] for r in ok])),
            "matched_mean_iou": float(np.mean([r["matched_mean_iou"] for r in ok])),
            "matched_mean_dice": float(np.mean([r["matched_mean_dice"] for r in ok])),
            "mean_n_pred": float(np.mean([r["n_pred"] for r in ok])),
            "mean_n_gt": float(np.mean([r["n_gt"] for r in ok])),
            "mean_bg_coverage": float(np.mean([r["bg_coverage"] for r in ok])),
        })
        # Per-N breakdown (by number of GT regions)
        by_m: dict = {}
        for r in ok:
            by_m.setdefault(r["n_gt"], []).append(r["panoptic_iou"])
        summary["panoptic_iou_by_n_gt"] = {
            str(k): float(np.mean(v)) for k, v in sorted(by_m.items())
        }

    out_path = output_dir / "zero_shot_results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "samples": results}, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"  Qwen2SAM_ZS [{dataset_name}] — "
          f"{summary['n_ok']}/{summary['n_total']} ok in {elapsed:.1f}s")
    print("=" * 80)
    if ok:
        print(f"  Panoptic IoU  : {summary['panoptic_iou']:.4f}   "
              f"Panoptic Dice : {summary['panoptic_dice']:.4f}")
        print(f"  Matched mIoU  : {summary['matched_mean_iou']:.4f}   "
              f"Matched mDice : {summary['matched_mean_dice']:.4f}")
        print(f"  <N_pred>      : {summary['mean_n_pred']:.2f}   "
              f"<N_gt>        : {summary['mean_n_gt']:.2f}   "
              f"<bg>          : {summary['mean_bg_coverage']*100:.2f}%")
        print(f"  pIoU by #GT   : {summary['panoptic_iou_by_n_gt']}")
    print(f"  wrote {out_path}")
    if vis_dir is not None:
        print(f"  vis   {vis_dir}/")


if __name__ == "__main__":
    main()
