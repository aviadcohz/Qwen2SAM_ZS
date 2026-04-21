#!/usr/bin/env python3
"""Orchestrate every (model_family × dataset) evaluation pair.

The runner does NOT touch a GPU itself. For each (model, dataset) combo it
invokes the model's eval_*.py as a fresh subprocess, so each family can
own its own imports / GPU allocation without polluting the parent process.

Typical usage:

    # Run everything (all models × all datasets)
    python master_runner.py

    # One model, all datasets
    python master_runner.py --model sam3_vanilla

    # One dataset, all models
    python master_runner.py --dataset RWTD

    # One specific cell, quick debug
    python master_runner.py --model grounded_sam3 --dataset CAID --limit 5

Each eval_*.py MUST accept these flags and nothing else is expected here:
    --config <path>       path to datasets_config.yaml
    --dataset <name>      dataset key (CAID, RWTD, STLD, ADE20k_Detecture)
    --output-dir <path>   dataset-level dir — writes zero_shot_results.json
                          + vis/<sample>.png inside
    --limit <int>         optional sample cap for quick debugging
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


SUITE_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = SUITE_ROOT / "datasets_config.yaml"
RESULTS_ROOT = SUITE_ROOT / "results"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_targets(cfg: dict, model: str | None, dataset: str | None):
    models = cfg["models"]
    datasets = list(cfg["datasets"].keys())

    if model:
        models = [m for m in models if m["name"] == model]
        if not models:
            raise SystemExit(f"unknown --model '{model}'. "
                             f"Known: {[m['name'] for m in cfg['models']]}")
    if dataset:
        if dataset not in datasets:
            raise SystemExit(f"unknown --dataset '{dataset}'. "
                             f"Known: {datasets}")
        datasets = [dataset]

    return models, datasets


def run_one(
    model_entry: dict, dataset_name: str,
    config_path: Path, limit: int | None, dry_run: bool,
) -> dict:
    script_path = SUITE_ROOT / model_entry["script"]
    out_dir = RESULTS_ROOT / model_entry["name"] / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script_path),
        "--config", str(config_path),
        "--dataset", dataset_name,
        "--output-dir", str(out_dir),
    ]
    # Per-model options carried in the registry entry.
    if "backend" in model_entry:
        cmd += ["--backend", model_entry["backend"]]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    header = f"[{model_entry['name']:>15s} × {dataset_name:<20s}]"
    print(f"{header} $ {' '.join(cmd)}", flush=True)
    if dry_run:
        return {"status": "dry_run", "cmd": cmd, "output_dir": str(out_dir)}

    if not script_path.exists():
        print(f"{header} SKIP — eval script not found: {script_path}",
              flush=True)
        return {"status": "missing_script", "output_dir": str(out_dir)}

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    rc = proc.returncode
    status = "ok" if rc == 0 else "failed"
    print(f"{header} done rc={rc} ({elapsed:.1f}s) → {out_dir}",
          flush=True)
    return {
        "status": status, "returncode": rc,
        "elapsed_seconds": elapsed, "output_dir": str(out_dir),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model family.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run only this dataset.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap samples per run (debug).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--summary-out", type=Path,
                        default=RESULTS_ROOT / "runner_summary.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    models, datasets = resolve_targets(cfg, args.model, args.dataset)

    print(f"[runner] models:   {[m['name'] for m in models]}")
    print(f"[runner] datasets: {datasets}")
    print(f"[runner] results:  {RESULTS_ROOT}")

    runs = []
    for m in models:
        for d in datasets:
            res = run_one(m, d, args.config, args.limit, args.dry_run)
            runs.append({"model": m["name"], "dataset": d, **res})

    summary = {
        "n_runs": len(runs),
        "n_ok": sum(1 for r in runs if r["status"] == "ok"),
        "n_failed": sum(1 for r in runs if r["status"] == "failed"),
        "n_missing": sum(1 for r in runs if r["status"] == "missing_script"),
        "runs": runs,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print(f"  {summary['n_ok']} ok   "
          f"{summary['n_failed']} failed   "
          f"{summary['n_missing']} missing-script   "
          f"out of {summary['n_runs']} total")
    print(f"  summary: {args.summary_out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
