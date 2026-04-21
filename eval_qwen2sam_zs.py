#!/usr/bin/env python3
"""Family 4 — qwen2sam_zs (the Qwen → SAM3 text pipeline of the Detecture paper).

Thin shim that invokes evaluate_zero_shot_pipeline.py (same directory) and
republishes its outputs into the eval suite's per-model / per-dataset tree:

    <output-dir>/zero_shot_results.json
    <output-dir>/vis/<sample>.png      (unless --no-vis)

Format matches sam3_vanilla / grounded_sam3 / vlm_end2end so side-by-side
comparison is uniform.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


_SUITE_ROOT = Path(__file__).resolve().parent
QWEN2SAM_ZS_SCRIPT = _SUITE_ROOT / "evaluate_zero_shot_pipeline.py"

# eval_suite dataset name → existing pipeline's --dataset key (lower-case).
DATASET_NAME_MAP = {
    "CAID":             "caid",
    "RWTD":             "rwtd",
    "STLD":             "stld",
    "ADE20k_Detecture": "ade20k_detecture",
    "ADE20K_textured":  "ade20k_textured",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=_SUITE_ROOT / "datasets_config.yaml")
    parser.add_argument("--dataset", required=True,
                        help="eval_suite dataset name (e.g. RWTD)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Dataset-level dir to publish into.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip PNG visualizations.")
    args = parser.parse_args()

    if not QWEN2SAM_ZS_SCRIPT.exists():
        raise SystemExit(f"pipeline script not found: {QWEN2SAM_ZS_SCRIPT}")
    if args.dataset not in DATASET_NAME_MAP:
        raise SystemExit(f"unmapped dataset: {args.dataset}. "
                         f"Known: {list(DATASET_NAME_MAP)}")
    qzs_key = DATASET_NAME_MAP[args.dataset]

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.dataset not in cfg["datasets"]:
        raise SystemExit(f"dataset missing from config: {args.dataset}")

    # The existing pipeline writes to <output_dir>/<dataset_lower>/
    # {zero_shot_results.json, vis/*.png}. Give it a scratch dir we can
    # then re-publish from.
    scratch_dir = _SUITE_ROOT / "results" / "qwen2sam_zs" / "_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(QWEN2SAM_ZS_SCRIPT),
        "--dataset", qzs_key,
        "--output_dir", str(scratch_dir),
    ]
    if args.no_vis:
        cmd += ["--no_vis"]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]

    print(f"[qwen2sam_zs shim] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=False)
    if proc.returncode != 0:
        raise SystemExit(f"pipeline failed (rc={proc.returncode})")

    src_dir = scratch_dir / qzs_key
    src_json = src_dir / "zero_shot_results.json"
    src_vis = src_dir / "vis"
    if not src_json.exists():
        raise SystemExit(f"expected output missing: {src_json}")

    # Publish JSON (tag summary with normalized model/dataset names).
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(src_json.read_text())
    payload.setdefault("summary", {})["model_family"] = "qwen2sam_zs"
    payload["summary"]["dataset"] = args.dataset
    (args.output_dir / "zero_shot_results.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )

    # Publish visualizations.
    if not args.no_vis and src_vis.exists():
        dst_vis = args.output_dir / "vis"
        if dst_vis.exists():
            shutil.rmtree(dst_vis)
        shutil.copytree(src_vis, dst_vis)
        print(f"  vis   {dst_vis}/")

    print(f"  wrote {args.output_dir / 'zero_shot_results.json'}")


if __name__ == "__main__":
    main()
