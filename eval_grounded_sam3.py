#!/usr/bin/env python3
"""Family 2 — BBox-to-Mask via Grounding DINO → SAM3.

Pipeline per sample (K = number of GT masks):
  1. Grounding DINO (HF `IDEA-Research/grounding-dino-tiny`) consumes one
     dot-separated text string (from datasets_config.yaml →
     prompts.grounded_sam3.text) and returns boxes + per-box scores.
  2. Keep the top-K boxes by confidence. If DINO returned fewer, we pad the
     missing heatmaps with a huge-negative logit so Softmax routes them to
     the dustbin and the Hungarian matcher charges 0 IoU (failure penalty).
  3. Each kept box is fed SEQUENTIALLY into SAM3's geometry-encoder → fusion
     encoder → Semantic Seg Head to produce one heatmap. We deliberately
     SKIP the mask-proposal decoder (no forward_grounding).
  4. Stack the K heatmaps, run metrics_utils.compute_sample_metrics
     (softmax + dustbin, Hungarian, ARI, Coverage).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_SUITE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SUITE_ROOT))

_SAM3_ROOT = Path("/home/aviad/sam3")
if str(_SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM3_ROOT))

from metrics_utils import compute_sample_metrics, aggregate_run       # noqa: E402
from data_utils import (                                             # noqa: E402
    load_samples, load_gt_masks, load_image_rgb,
    preprocess_image_for_sam3, SAM3_SIZE,
)
from viz_utils import visualize_sample                               # noqa: E402


# --------------------------------------------------------------------- #
# Grounding DINO                                                          #
# --------------------------------------------------------------------- #

def load_grounding_dino(device: torch.device,
                        model_name: str = "IDEA-Research/grounding-dino-tiny"):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    print(f"[DINO] loading {model_name}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(model_name)
        .to(device).eval()
    )
    print(f"[DINO] ready on {device}", flush=True)
    return model, processor


@torch.no_grad()
def detect_boxes(
    dino_model, dino_proc, image_pil, text: str, device: torch.device,
    box_threshold: float = 0.2, text_threshold: float = 0.2,
):
    """Run DINO zero-shot detection. Returns (boxes_xyxy, scores) tensors on
    CPU, both sorted by descending score."""
    inputs = dino_proc(images=image_pil, text=[text], return_tensors="pt").to(device)
    outputs = dino_model(**inputs)
    result = dino_proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]],   # (H, W) in pixels
    )[0]
    boxes = result["boxes"].detach().cpu()        # (N, 4) xyxy in pixels
    scores = result["scores"].detach().cpu()      # (N,)
    if scores.numel() > 0:
        order = torch.argsort(scores, descending=True)
        boxes = boxes[order]
        scores = scores[order]
    return boxes, scores


def xyxy_to_cxcywh_norm(box_xyxy: torch.Tensor, H: int, W: int) -> tuple:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return cx, cy, w, h


# --------------------------------------------------------------------- #
# SAM3 box-prompt → semantic heatmap (bypasses mask-proposal decoder)    #
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
def sam3_box_to_semantic_logits(
    sam3, backbone_out, box_cxcywh_norm: tuple,
    gt_h: int, gt_w: int, device: torch.device,
) -> np.ndarray:
    """One box → (gt_h, gt_w) semantic-seg-head logits.

    Bypasses sam3.forward_grounding() entirely: constructs a Prompt with only
    the box, runs geometry_encoder → fusion encoder → segmentation_head.
    """
    from sam3.model.geometry_encoders import Prompt

    B = 1
    img_ids = torch.arange(B, device=device)
    img_feats, img_pos_embeds, vis_feat_sizes = _get_img_feats(
        sam3, backbone_out, img_ids,
    )

    box_t = torch.tensor(box_cxcywh_norm, device=device,
                         dtype=torch.float32).view(1, 1, 4)
    box_labels = torch.ones(1, 1, device=device, dtype=torch.bool)
    geo_prompt = Prompt(box_embeddings=box_t, box_labels=box_labels)

    geo_feats, geo_masks = sam3.geometry_encoder(
        geo_prompt=geo_prompt,
        img_feats=img_feats, img_sizes=vis_feat_sizes,
        img_pos_embeds=img_pos_embeds,
    )

    # No visual-exemplar mask; pass zero-length tensors just like _encode_prompt.
    visual_prompt_embed = torch.zeros(
        (0, *geo_feats.shape[1:]), device=geo_feats.device, dtype=geo_feats.dtype,
    )
    visual_prompt_mask = torch.zeros(
        (*geo_masks.shape[:-1], 0),
        device=geo_masks.device, dtype=geo_masks.dtype,
    )
    prompt = torch.cat([geo_feats, visual_prompt_embed], dim=0)      # (N, B, 256)
    prompt_mask = torch.cat([geo_masks, visual_prompt_mask], dim=1)  # (B, N)
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
    sem_logits = seg_head.semantic_seg_head(pixel_embed)    # (B, 1, H, W)
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
# Per-sample evaluation                                                  #
# --------------------------------------------------------------------- #

def evaluate_sample(
    sam3, dino_model, dino_proc,
    sample: dict, dino_text: str,
    dustbin_logit: float, missing_logit: float, image_size: int,
    device: torch.device,
    vis_dir: Path | None, dataset_name: str,
) -> dict:
    from PIL import Image

    sid = sample["id"]
    try:
        image_rgb = load_image_rgb(sample["image_path"])
    except FileNotFoundError:
        return {"id": sid, "status": "image_read_failed"}
    image_pil = Image.fromarray(image_rgb)

    gt_masks, kept_idx = load_gt_masks(sample["gt_masks"])
    if len(gt_masks) == 0:
        return {"id": sid, "status": "no_gt_masks"}
    gt_descs = [sample["gt_descs"][i] for i in kept_idx]
    gt_h, gt_w = gt_masks[0].shape
    K = len(gt_masks)

    # Stage 1 — DINO detection at original image resolution.
    boxes_xyxy, scores = detect_boxes(dino_model, dino_proc,
                                      image_pil, dino_text, device)
    n_det = int(boxes_xyxy.shape[0])
    keep_k = min(n_det, K)
    kept_boxes = boxes_xyxy[:keep_k]
    kept_scores = scores[:keep_k]

    # Stage 2 — SAM3 forward once on the 1008 tensor, reused for all boxes.
    sam_img = preprocess_image_for_sam3(image_rgb, image_size).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
        backbone_out = sam3.backbone.forward_image(sam_img)
        backbone_out["img_batch_all_stages"] = sam_img

        # Stage 3 — box → semantic heatmap, sequential (one box at a time).
        H, W = image_pil.size[1], image_pil.size[0]
        logit_list = []
        box_descs = []
        for b, s in zip(kept_boxes, kept_scores):
            cx, cy, w, h = xyxy_to_cxcywh_norm(b, H, W)
            logit = sam3_box_to_semantic_logits(
                sam3, backbone_out, (cx, cy, w, h), gt_h, gt_w, device,
            )
            logit_list.append(logit)
            x1, y1, x2, y2 = [int(v) for v in b.tolist()]
            box_descs.append(
                f"box [{x1},{y1},{x2},{y2}] score={float(s):.3f}"
            )

    # Stage 3b — pad missing heatmaps with strongly-negative logits so the
    # Softmax sends them to the dustbin and Hungarian charges 0 IoU.
    for _ in range(K - len(logit_list)):
        logit_list.append(np.full((gt_h, gt_w), missing_logit, dtype=np.float32))
        box_descs.append("(padded: DINO returned < K)")
    logits = np.stack(logit_list, axis=0)   # (K, gt_h, gt_w)

    # Stage 4 — scoring (shared scorer).
    metrics = compute_sample_metrics(logits, gt_masks, dustbin_logit=dustbin_logit)

    if vis_dir is not None:
        visualize_sample(
            image_rgb=image_rgb, gt_masks=gt_masks,
            logits=logits, argmax_map=metrics["argmax_map"],
            descs=box_descs, gt_descs=gt_descs,
            match_info=metrics,
            sample_id=str(sid), dataset_name=dataset_name,
            save_path=vis_dir / f"{sid}.png",
            model_family="grounded_sam3",
            logit_title="SAM3 box-prompt logits",
            pred_row_header="box",
        )

    return {
        "id": sid,
        "status": "ok",
        "n_detected_boxes": n_det,
        "n_used_boxes": keep_k,
        "n_padded": K - keep_k,
        "kept_scores": [float(s) for s in kept_scores.tolist()],
        "kept_boxes_xyxy": [list(map(float, b.tolist())) for b in kept_boxes],
        "n_pred": metrics["n_pred"],
        "n_gt": metrics["n_gt"],
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
    parser.add_argument("--vis-every", type=int, default=1)
    parser.add_argument("--dino_model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box_threshold", type=float, default=0.2)
    parser.add_argument("--text_threshold", type=float, default=0.2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.dataset not in cfg["datasets"]:
        raise SystemExit(f"unknown dataset: {args.dataset}")
    ds_cfg = cfg["datasets"][args.dataset]
    if "grounded_sam3" not in ds_cfg["prompts"]:
        raise SystemExit(
            f"{args.dataset} has no grounded_sam3 prompt block in the config.")

    dino_text = ds_cfg["prompts"]["grounded_sam3"]["text"]
    missing_logit = float(cfg["padding"]["missing_logit"])
    dustbin_logit = float(cfg["shared"]["dustbin_logit"])
    image_size = int(cfg["shared"]["image_size"])

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = None if args.no_vis else (out_dir / "vis")
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "zero_shot_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_samples(ds_cfg["metadata"], limit=args.limit)
    print(f"[data] {args.dataset} n={len(samples)}  text={dino_text!r}", flush=True)

    dino_model, dino_proc = load_grounding_dino(device, args.dino_model)
    sam3 = load_sam3(device)

    results = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        sid = sample["id"]
        should_vis = (vis_dir is not None) and (i % args.vis_every == 0)
        try:
            res = evaluate_sample(
                sam3, dino_model, dino_proc, sample, dino_text,
                dustbin_logit, missing_logit, image_size, device,
                vis_dir=(vis_dir if should_vis else None),
                dataset_name=args.dataset,
            )
        except Exception as e:       # noqa: BLE001
            res = {"id": sid, "status": "exception", "error": repr(e)}
        results.append(res)

        if res.get("status") == "ok":
            print(f"  [{i+1:4d}/{len(samples)}] {str(sid):>22s}  "
                  f"det={res['n_detected_boxes']:<2d}  "
                  f"K={res['n_gt']}  "
                  f"pad={res['n_padded']}  "
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
        "model_family": "grounded_sam3",
        "dataset": args.dataset,
        "metadata_path": ds_cfg["metadata"],
        "dino_text": dino_text,
        "dino_model": args.dino_model,
        "dustbin_logit": dustbin_logit,
        "missing_logit": missing_logit,
        "image_size": image_size,
        "elapsed_seconds": elapsed,
    })

    json_path.write_text(json.dumps(
        {"summary": summary, "samples": results},
        indent=2, default=str,
    ))

    print("\n" + "=" * 72)
    print(f"  grounded_sam3 [{args.dataset}] — "
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
