#!/usr/bin/env python3
"""Family 1 — Text-to-Mask via SAM3's native text encoder.

Pipeline per sample (K = number of GT masks):
  1. Read the per-dataset text list from datasets_config.yaml →
     prompts.sam3_vanilla. Two supported modes:
       static : prompts.sam3_vanilla.texts is a list of per-class phrases
                (e.g. ["foreground texture", "background texture"]).
                If len != K_GT we truncate or repeat the last to match.
       repeat : prompts.sam3_vanilla.text is one generic phrase repeated
                K_GT times (used when no class vocabulary exists, e.g. ADE20K).
  2. Each phrase is encoded by SAM3's VE text encoder and passed through the
     fusion encoder + Semantic Seg Head, producing one heatmap per phrase.
  3. Stack the K heatmaps and score via metrics_utils (same dustbin +
     Hungarian + ARI + Coverage used by every other family).

No mask-proposal decoder is involved — mirroring the grounded-SAM3 script's
strict rule.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_SUITE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SUITE_ROOT))

_SAM3_ROOT = Path("/home/aviad/sam3")
if str(_SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM3_ROOT))

from metrics_utils import (                                           # noqa: E402
    compute_sample_metrics, aggregate_run, masks_to_logits,
)
from data_utils import (                                              # noqa: E402
    load_samples, load_gt_masks, load_image_rgb,
    preprocess_image_for_sam3, SAM3_SIZE,
)
from viz_utils import visualize_sample                                # noqa: E402


# --------------------------------------------------------------------- #
# SAM3 loading + text → semantic heatmap                                  #
# --------------------------------------------------------------------- #

def load_sam3(device: torch.device):
    import sam3 as sam3_module
    from sam3.model_builder import build_sam3_image_model
    bpe_path = str(
        Path(sam3_module.__path__[0]) / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    )
    print("[SAM3] building image model from HuggingFace weights...", flush=True)
    sam3 = build_sam3_image_model(
        bpe_path=bpe_path, eval_mode=True, checkpoint_path=None,
        load_from_HF=True, enable_segmentation=True, device=str(device),
    )
    sam3.eval()
    for p in sam3.parameters():
        p.requires_grad = False
    print("[SAM3] ready (frozen)", flush=True)
    return sam3


def _get_img_feats(sam3, backbone_out, img_ids):
    n_levels = sam3.num_feature_levels
    vis_feats = backbone_out["backbone_fpn"][-n_levels:]
    vis_pos_enc = backbone_out["vision_pos_enc"][-n_levels:]
    vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
    img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
    img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
    return img_feats, img_pos_embeds, vis_feat_sizes


@torch.no_grad()
def encode_text_prompt(sam3, description: str, device: torch.device):
    # language_features: (N_tokens, B, 256)   language_mask: (B, N_tokens)
    out = sam3.backbone.forward_text([description], device=device)
    prompt = out["language_features"]
    mask = out["language_mask"]
    if mask.ndim == 3:
        mask = mask.squeeze(1)
    return prompt, mask


@torch.no_grad()
def sam3_text_to_semantic_logits(
    sam3, backbone_out, description: str,
    gt_h: int, gt_w: int, device: torch.device,
) -> np.ndarray:
    """One description → (gt_h, gt_w) semantic-seg-head logits."""
    prompt_embed, prompt_mask = encode_text_prompt(sam3, description, device)
    B = prompt_embed.shape[1]
    img_ids = torch.arange(B, device=device)

    img_feats, img_pos_embeds, vis_feat_sizes = _get_img_feats(
        sam3, backbone_out, img_ids,
    )
    prompt = prompt_embed       # (N, B, 256)
    prompt_pos = torch.zeros_like(prompt)

    memory_dict = sam3.transformer.encoder(
        src=[f.clone() for f in img_feats],
        src_key_padding_mask=None,
        src_pos=[p.clone() for p in img_pos_embeds],
        prompt=prompt, prompt_pos=prompt_pos,
        prompt_key_padding_mask=prompt_mask,
        feat_sizes=vis_feat_sizes,
    )
    enc_hs = memory_dict["memory"]

    seg_head = sam3.segmentation_head
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
    sem_logits = seg_head.semantic_seg_head(pixel_embed)
    if sem_logits.ndim == 4:
        sem_logits = sem_logits[0, 0]
    elif sem_logits.ndim == 3:
        sem_logits = sem_logits[0]

    sem_logits = F.interpolate(
        sem_logits[None, None].float(),
        size=(gt_h, gt_w), mode="bilinear", align_corners=False,
    ).squeeze()
    return sem_logits.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------- #
# Proposal-mode: SAM3's mask-proposal decoder with Top-K scoring          #
# --------------------------------------------------------------------- #

@torch.no_grad()
def sam3_proposal_top_k(
    sam3, image_pil, text: str, k_target: int,
    image_size: int, device: torch.device,
):
    """Run SAM3's native text-grounded mask-proposal decoder and return the
    top-K masks by descending confidence.

    Uses the official Sam3Processor wrapper so we go through forward_grounding
    (which returns {pred_masks, pred_logits, presence_logit_dec}) rather than
    the semantic-seg head. This is how SAM3 is *designed* to produce multiple
    masks from a single text query — right mode for variable-K datasets
    (ADE20K) where a generic phrase like "texture regions" has no class
    vocabulary.

    Returns: list of (mask_HW_float01_on_orig_image, score) tuples, score-desc.
    """
    from sam3.model.sam3_image_processor import Sam3Processor
    proc = Sam3Processor(
        model=sam3, resolution=image_size, device=str(device),
        confidence_threshold=0.0,   # keep everything; we pick top-K ourselves
    )
    state = proc.set_image(image_pil)
    state = proc.set_text_prompt(text, state)

    scores = state["scores"].detach().cpu()                   # (N,)
    masks_logits = state["masks_logits"].detach().cpu()       # (N, 1, H, W) soft probs
    if masks_logits.ndim == 4:
        masks_logits = masks_logits.squeeze(1)                # (N, H, W)
    n = int(scores.numel())
    if n == 0:
        return []
    order = torch.argsort(scores, descending=True)
    keep = min(n, k_target)
    idx = order[:keep]
    return [
        (masks_logits[i].numpy().astype(np.float32), float(scores[i]))
        for i in idx.tolist()
    ]


# --------------------------------------------------------------------- #
# Per-dataset prompt resolution                                           #
# --------------------------------------------------------------------- #

def resolve_texts(prompt_block: dict, k_gt: int) -> List[str]:
    """Produce exactly K_GT description strings for the semantic-head path.

    static mode: take prompt_block['texts']; if shorter than K_GT pad by
    repeating the last, if longer truncate.
    repeat mode: prompt_block['text'] repeated K_GT times (legacy — prefer
                 'proposal' mode instead, since repeating the same text K
                 times gives K identical heatmaps).
    proposal mode: not used here — see evaluate_sample().
    """
    mode = prompt_block["mode"]
    if mode == "static":
        texts = list(prompt_block["texts"])
        if len(texts) >= k_gt:
            return texts[:k_gt]
        return texts + [texts[-1]] * (k_gt - len(texts))
    if mode == "repeat":
        return [prompt_block["text"]] * k_gt
    raise ValueError(
        f"resolve_texts does not handle mode={mode!r}; 'proposal' is routed "
        "separately in evaluate_sample()."
    )


# --------------------------------------------------------------------- #
# Per-sample eval                                                         #
# --------------------------------------------------------------------- #

def evaluate_sample(
    sam3, sample: dict, prompt_block: dict,
    dustbin_logit: float, missing_logit: float,
    image_size: int, device: torch.device,
    vis_dir: Path | None, dataset_name: str,
) -> dict:
    from PIL import Image

    sid = sample["id"]
    try:
        image_rgb = load_image_rgb(sample["image_path"])
    except FileNotFoundError:
        return {"id": sid, "status": "image_read_failed"}

    gt_masks, kept_idx = load_gt_masks(sample["gt_masks"])
    if len(gt_masks) == 0:
        return {"id": sid, "status": "no_gt_masks"}
    gt_descs = [sample["gt_descs"][i] for i in kept_idx]
    gt_h, gt_w = gt_masks[0].shape
    K = len(gt_masks)

    mode = prompt_block["mode"]
    n_padded = 0

    if mode == "proposal":
        # Variable-K path: one text query → top-K mask proposals from SAM3.
        text = prompt_block["text"]
        image_pil = Image.fromarray(image_rgb)
        ranked = sam3_proposal_top_k(
            sam3, image_pil, text, k_target=K,
            image_size=image_size, device=device,
        )
        masks_only = [m for m, _ in ranked]
        scores_only = [s for _, s in ranked]
        logits = masks_to_logits(
            masks_only, target_hw=(gt_h, gt_w),
            missing_logit=missing_logit, k_target=K,
        )
        n_padded = K - len(ranked)
        descs = [f"proposal score={s:.3f}" for s in scores_only] + \
                ["(padded: SAM3 returned < K proposals)"] * n_padded
        texts = [text] * K   # for JSON traceability
        logit_title = "SAM3 mask-proposal soft probs"
        pred_row_header = "proposal"
    else:
        # static / repeat: per-class (or repeated-class) text → semantic head.
        texts = resolve_texts(prompt_block, K)
        sam_img = preprocess_image_for_sam3(image_rgb, image_size).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
            backbone_out = sam3.backbone.forward_image(sam_img)
            backbone_out["img_batch_all_stages"] = sam_img
            logit_list = [
                sam3_text_to_semantic_logits(
                    sam3, backbone_out, t, gt_h, gt_w, device,
                )
                for t in texts
            ]
        logits = np.stack(logit_list, axis=0)
        descs = list(texts)
        logit_title = "SAM3 text logits"
        pred_row_header = "text"

    metrics = compute_sample_metrics(logits, gt_masks, dustbin_logit=dustbin_logit)

    if vis_dir is not None:
        visualize_sample(
            image_rgb=image_rgb, gt_masks=gt_masks,
            logits=logits, argmax_map=metrics["argmax_map"],
            descs=descs, gt_descs=gt_descs,
            match_info=metrics,
            sample_id=str(sid), dataset_name=dataset_name,
            save_path=vis_dir / f"{sid}.png",
            model_family=f"sam3_vanilla[{mode}]",
            logit_title=logit_title,
            pred_row_header=pred_row_header,
        )

    return {
        "id": sid, "status": "ok",
        "prompt_mode": mode,
        "n_padded": n_padded,
        "texts": texts, "n_pred": metrics["n_pred"], "n_gt": metrics["n_gt"],
        "panoptic_iou": metrics["panoptic_iou"],
        "panoptic_dice": metrics["panoptic_dice"],
        "matched_mean_iou": metrics["matched_mean_iou"],
        "matched_mean_dice": metrics["matched_mean_dice"],
        "assignment": metrics["assignment"],
        "ari": metrics["ari"],
        "bg_coverage": metrics["bg_coverage"],
    }


# --------------------------------------------------------------------- #
# Main                                                                    #
# --------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=_SUITE_ROOT / "datasets_config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Dataset-level dir. Writes zero_shot_results.json "
                             "and vis/<sample>.png inside.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip PNG visualizations.")
    parser.add_argument("--vis-every", type=int, default=1,
                        help="Write a visualization every N samples.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["datasets"][args.dataset]
    prompt_block = ds_cfg["prompts"]["sam3_vanilla"]
    dustbin_logit = float(cfg["shared"]["dustbin_logit"])
    missing_logit = float(cfg["padding"]["missing_logit"])
    image_size = int(cfg["shared"]["image_size"])

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = None if args.no_vis else (out_dir / "vis")
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "zero_shot_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_samples(ds_cfg["metadata"], limit=args.limit)
    print(f"[data] {args.dataset} n={len(samples)}  "
          f"mode={prompt_block['mode']}", flush=True)

    sam3 = load_sam3(device)

    results = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        sid = sample["id"]
        should_vis = (vis_dir is not None) and (i % args.vis_every == 0)
        try:
            res = evaluate_sample(
                sam3, sample, prompt_block,
                dustbin_logit, missing_logit, image_size, device,
                vis_dir=(vis_dir if should_vis else None),
                dataset_name=args.dataset,
            )
        except Exception as e:      # noqa: BLE001
            res = {"id": sid, "status": "exception", "error": repr(e)}
        results.append(res)

        if res.get("status") == "ok":
            print(f"  [{i+1:4d}/{len(samples)}] {str(sid):>22s}  "
                  f"K={res['n_gt']}  "
                  f"pIoU={res['panoptic_iou']:.4f}  "
                  f"mIoU={res['matched_mean_iou']:.4f}  "
                  f"ARI={res['ari']:.4f}  "
                  f"bg={res['bg_coverage']:.2f}")
        else:
            print(f"  [{i+1:4d}/{len(samples)}] {str(sid):>22s}  "
                  f"SKIP status={res.get('status')}")

    elapsed = time.time() - t0
    summary = aggregate_run(results)
    summary.update({
        "model_family": "sam3_vanilla",
        "dataset": args.dataset,
        "metadata_path": ds_cfg["metadata"],
        "prompt_mode": prompt_block["mode"],
        "dustbin_logit": dustbin_logit,
        "image_size": image_size,
        "elapsed_seconds": elapsed,
    })

    json_path.write_text(json.dumps(
        {"summary": summary, "samples": results},
        indent=2, default=str,
    ))

    print("\n" + "=" * 72)
    print(f"  sam3_vanilla [{args.dataset}] — "
          f"{summary.get('n_ok', 0)}/{summary.get('n_total', 0)} ok "
          f"in {elapsed:.1f}s")
    if summary.get("n_ok", 0):
        print(f"  pIoU  = {summary['panoptic_iou']:.4f}   "
              f"mIoU = {summary['matched_mean_iou']:.4f}   "
              f"ARI  = {summary['mean_ari']:.4f}   "
              f"bg   = {summary['mean_bg_coverage']:.3f}")
    print(f"  wrote {json_path}")
    if vis_dir is not None:
        print(f"  vis   {vis_dir}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
