#!/usr/bin/env python3
"""Family 3 — End-to-end VLM that produces binary masks directly.

Pipeline per sample (K = number of GT masks):
  1. Read the per-dataset conversational prompt from datasets_config.yaml →
     prompts.vlm_end2end.text (mode: single).
  2. Call the chosen VLM backend's ``generate_masks`` method to produce up
     to K_GT binary masks + per-mask scores.
  3. Keep top-K_GT by score. If the model returned fewer, pad with a
     strongly-negative logit plane (per the shared padding policy) so the
     Softmax routes the phantom channel to dustbin and Hungarian charges
     0 IoU.
  4. Convert masks → per-channel logits (signed log-odds form that matches
     what metrics_utils.resolve_conflicts_softmax_argmax expects) and score.

Status: the script is complete EXCEPT the VLM backend itself. Neither
LISA nor Sa2VA is installed in the shared texture_boundary env. Two
hand-off points are marked with `# TODO(backend)`:
  - MockMaskVLM: a harmless reference implementation that emits K_GT full-
    image masks with score 1.0 so the scaffolding can be smoke-tested
    without any extra weights. Passes the Hungarian matcher but produces
    trivially bad numbers — replace before reporting.
  - Sa2VA / LISA adapters: each needs (a) model/processor load, (b) a
    generate_masks() that returns List[Tuple[np.ndarray, float]]. Nothing
    else changes; plug your model in and the rest of the pipeline (prompt,
    padding, scoring, output JSON) Just Works.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import yaml

_SUITE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SUITE_ROOT))

from metrics_utils import (                                          # noqa: E402
    compute_sample_metrics, aggregate_run, masks_to_logits,
)
from data_utils import load_samples, load_gt_masks, load_image_rgb   # noqa: E402
from viz_utils import visualize_sample                               # noqa: E402


# --------------------------------------------------------------------- #
# VLM backend interface                                                  #
# --------------------------------------------------------------------- #

class MaskVLM(ABC):
    """Contract every mask-producing VLM backend must fulfil."""

    name: str

    @abstractmethod
    def generate_masks(
        self, image_rgb: np.ndarray, prompt: str, k_max: int,
    ) -> List[Tuple[np.ndarray, float]]:
        """Return up to k_max (mask_HW_float01, score) tuples, score-ordered.

        mask_HW_float01: (H, W) float in [0,1] at the input image resolution.
        score: per-mask confidence used to truncate to top-K and break ties.
        """


class MockMaskVLM(MaskVLM):
    """Reference implementation for scaffolding smoke tests.

    Returns K full-image masks with descending scores. Will score poorly;
    it's just a null-hypothesis that exercises the full code path.
    """

    name = "mock"

    def generate_masks(self, image_rgb, prompt, k_max):
        H, W = image_rgb.shape[:2]
        out = []
        for i in range(k_max):
            mask = np.ones((H, W), dtype=np.float32) * (1.0 - 0.1 * i)
            out.append((mask, 1.0 - 0.01 * i))
        return out


class Sa2VAMaskVLM(MaskVLM):
    """ByteDance Sa2VA-4B (InternVL-based conversational VLM with [SEG]).

    Uses `model.predict_forward(image=PIL, text=str, tokenizer=tok)` which
    returns {'prediction': str, 'prediction_masks': [np.ndarray(1,H,W), ...]}
    — one mask per [SEG] token emitted in the model's answer.

    Scoring: Sa2VA doesn't expose per-[SEG] confidences, so we rank masks by
    their area (more confident grounding → more non-zero pixels). This is a
    stable, reproducible proxy that's consistent across samples.
    """

    name = "sa2va"

    def __init__(self, device, model_path: str = "ByteDance/Sa2VA-4B"):
        # Sa2VA's modeling files declare `import flash_attn`; HuggingFace's
        # static check_imports fails without the package even though we set
        # use_flash_attn=False. The env ships a dummy flash_attn stub
        # package (installed once via pip) that satisfies the import scan
        # without providing real kernels. At runtime Sa2VA never calls into
        # it because its flash paths are gated by `has_flash_attn` checks.
        from transformers import AutoTokenizer, AutoModel
        from transformers.modeling_utils import PreTrainedModel
        # transformers 5.x expects every subclass to populate
        # `self.all_tied_weights_keys` in __init__ (via get_expanded_tied_weights_keys).
        # Sa2VA's custom Sa2VAChatModel predates that convention, so we
        # supply an empty dict as a class-level fallback. Missing tied-
        # weight keys just means transformers won't try to re-tie anything
        # — harmless for inference-only usage.
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}
        print(f"[Sa2VA] loading {model_path}...", flush=True)
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,           # flash_attn not in env
            trust_remote_code=True,
        ).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False,
        )
        print(f"[Sa2VA] ready on {device}", flush=True)

    @torch.no_grad()
    def generate_masks(self, image_rgb, prompt, k_max):
        from PIL import Image
        image_pil = Image.fromarray(image_rgb)

        # Sa2VA expects a literal "<image>" token somewhere in text. Prepend
        # if the caller's prompt forgot it.
        text = prompt if "<image>" in prompt else f"<image>{prompt}"

        out = self.model.predict_forward(
            image=image_pil, text=text, past_text="",
            mask_prompts=None, tokenizer=self.tokenizer,
        )
        raw_masks = out.get("prediction_masks") or []

        ranked = []
        H, W = image_rgb.shape[:2]
        for m in raw_masks:
            if m is None:
                continue
            m = np.asarray(m).astype(np.float32)
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            # Sa2VA masks usually already at original resolution; if not,
            # metrics_utils.masks_to_logits will resize — but normalise here
            # for deterministic shape on the score proxy.
            if m.shape != (H, W):
                import cv2
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
            m = (m > 0.5).astype(np.float32)      # binarise
            area = float(m.sum())
            if area > 0:
                ranked.append((m, area))

        if not ranked:
            return []

        ranked.sort(key=lambda mk: mk[1], reverse=True)
        # Normalise scores to [0,1] by dividing by the top area, so the
        # numbers are comparable across samples even if masks aren't.
        top_area = ranked[0][1]
        ranked = [(m, a / top_area) for m, a in ranked]
        return ranked[:k_max]


# ---- Backend registry ------------------------------------------------- #
# Register real adapters here. `mock` is kept for harness debugging only
# (run via `python eval_vlm_end2end.py --backend mock ...`) — it's not in
# the master_runner's default model list.

BACKENDS: dict = {
    "mock":  MockMaskVLM,
    "sa2va": Sa2VAMaskVLM,
    # "lisa": LISAMaskVLM,  # pending: no HF-native path verified in this env
}


def load_backend(name: str, device: torch.device) -> MaskVLM:
    if name not in BACKENDS:
        raise SystemExit(
            f"unknown --backend '{name}'. "
            f"Known: {list(BACKENDS)}. Plug new adapters into BACKENDS.")
    cls = BACKENDS[name]
    try:
        return cls(device=device)     # adapters that need device
    except TypeError:
        return cls()                  # MockMaskVLM takes no args


# mask → logit conversion lives in metrics_utils.masks_to_logits so every
# mask-producing family (vlm_end2end, sam3_vanilla proposal, ...) uses the
# same stack formula.


# --------------------------------------------------------------------- #
# Per-sample eval                                                         #
# --------------------------------------------------------------------- #

def evaluate_sample(
    vlm: MaskVLM, sample: dict, prompt: str,
    dustbin_logit: float, missing_logit: float,
    vis_dir: Path | None, dataset_name: str,
) -> dict:
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

    ranked = vlm.generate_masks(image_rgb, prompt, k_max=K)
    ranked.sort(key=lambda mk: mk[1], reverse=True)
    kept = ranked[:K]
    masks_only = [m for m, _ in kept]
    scores_only = [float(s) for _, s in kept]

    logits = masks_to_logits(
        masks_only, target_hw=(gt_h, gt_w),
        missing_logit=missing_logit, k_target=K,
    )
    metrics = compute_sample_metrics(logits, gt_masks, dustbin_logit=dustbin_logit)

    if vis_dir is not None:
        mask_descs = [f"mask score={s:.3f}" for s in scores_only]
        mask_descs += ["(padded: VLM returned < K)"] * (K - len(kept))
        visualize_sample(
            image_rgb=image_rgb, gt_masks=gt_masks,
            logits=logits, argmax_map=metrics["argmax_map"],
            descs=mask_descs, gt_descs=gt_descs,
            match_info=metrics,
            sample_id=str(sid), dataset_name=dataset_name,
            save_path=vis_dir / f"{sid}.png",
            model_family=f"vlm_end2end[{vlm.name}]",
            logit_title="VLM mask logits",
            pred_row_header="mask",
        )

    return {
        "id": sid, "status": "ok",
        "backend": vlm.name, "prompt": prompt,
        "n_vlm_masks": len(ranked),
        "n_used_masks": len(kept),
        "n_padded": K - len(kept),
        "vlm_scores": scores_only,
        "n_pred": metrics["n_pred"], "n_gt": metrics["n_gt"],
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
    parser.add_argument("--backend", type=str, default="mock",
                        help="VLM backend name (mock | sa2va | lisa | ...).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ds_cfg = cfg["datasets"][args.dataset]
    # One conversational-prompt block is shared by every end-to-end VLM
    # backend (lisa, sa2va, mock, ...). If a future adapter needs its own
    # per-backend prompt, look up `prompts.<backend>` first then fall back
    # to `prompts.vlm_end2end`.
    prompt_block = (
        ds_cfg["prompts"].get(args.backend)
        or ds_cfg["prompts"]["vlm_end2end"]
    )
    if prompt_block["mode"] != "single":
        raise SystemExit(
            f"vlm_end2end expects mode='single', got {prompt_block['mode']!r}")
    prompt = prompt_block["text"]
    missing_logit = float(cfg["padding"]["missing_logit"])
    dustbin_logit = float(cfg["shared"]["dustbin_logit"])

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = None if args.no_vis else (out_dir / "vis")
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "zero_shot_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_samples(ds_cfg["metadata"], limit=args.limit)
    print(f"[data] {args.dataset} n={len(samples)}  backend={args.backend}",
          flush=True)
    print(f"[prompt] {prompt!r}", flush=True)

    vlm = load_backend(args.backend, device)

    results = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        sid = sample["id"]
        should_vis = (vis_dir is not None) and (i % args.vis_every == 0)
        try:
            res = evaluate_sample(
                vlm, sample, prompt, dustbin_logit, missing_logit,
                vis_dir=(vis_dir if should_vis else None),
                dataset_name=args.dataset,
            )
        except Exception as e:      # noqa: BLE001
            res = {"id": sid, "status": "exception", "error": repr(e)}
        results.append(res)

        if res.get("status") == "ok":
            print(f"  [{i+1:4d}/{len(samples)}] {str(sid):>22s}  "
                  f"K={res['n_gt']}  "
                  f"vlm={res['n_vlm_masks']}  pad={res['n_padded']}  "
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
        "model_family": "vlm_end2end",
        "backend": args.backend,
        "dataset": args.dataset,
        "metadata_path": ds_cfg["metadata"],
        "prompt": prompt,
        "dustbin_logit": dustbin_logit,
        "missing_logit": missing_logit,
        "elapsed_seconds": elapsed,
    })

    json_path.write_text(json.dumps(
        {"summary": summary, "samples": results},
        indent=2, default=str,
    ))

    print("\n" + "=" * 72)
    print(f"  vlm_end2end[{args.backend}] [{args.dataset}] — "
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
