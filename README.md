# Qwen2SAM_Detecture_Benchmark

Zero-shot evaluation suite for the **Qwen2SAM_Detecture** paper. Compares
four model families on four texture-segmentation datasets under an
identical scoring stack (softmax + dustbin + Hungarian + ARI + fixed
`-1e4` padding for under-produced masks).

Evaluated families (one sibling row in `results/` each):

| Family | What it does |
| --- | --- |
| `sam3_vanilla` | SAM3 text encoder → Semantic Seg Head; proposal-mode fallback on variable-K datasets |
| `grounded_sam3` | Grounding DINO → SAM3 box-prompt → Semantic Seg Head |
| `sa2va` | ByteDance Sa2VA-4B, end-to-end VLM with interleaved `[SEG]` tokens |
| `qwen2sam_zs` | Qwen3-VL-8B → SAM3 text (the core **Qwen2SAM_Detecture** zero-shot pipeline) |

The region count `K` is set **per-sample** from the GT:

- **CAID** → K = 1 (water surface).
- **RWTD** → K = 2 always.
- **STLD** → K = 2 always (Brodatz foreground on Brodatz background).
- **ADE20k_Detecture** → K = number of GT textures for that image (1..6).

## Install

```bash
# 1. clone this repo
git clone https://github.com/<your-gh-user>/Qwen2SAM_Detecture_Benchmark.git
cd Qwen2SAM_Detecture_Benchmark

# 2. clone SAM3 somewhere and point to it
git clone https://github.com/facebookresearch/sam3.git ~/sam3
pip install -e ~/sam3
export SAM3_ROOT=~/sam3   # optional; defaults to /home/aviad/sam3

# 3. install python deps
pip install -r requirements.txt

# 4. (one-time) install a stub flash_attn so Sa2VA's custom code loads.
#    No real flash-attention kernels are compiled — Sa2VA runs with
#    use_flash_attn=False and we only satisfy HF's static import check.
pip install --no-deps /path/to/flash_attn_stub
```

HF weights download on first run:

- `Qwen/Qwen3-VL-8B-Instruct`
- `ByteDance/Sa2VA-4B`
- `IDEA-Research/grounding-dino-tiny`
- SAM3 (via `build_sam3_image_model(load_from_HF=True)`)

## Datasets

Paths and per-dataset prompts are centralised in
[`datasets_config.yaml`](./datasets_config.yaml). Defaults expect:

| Dataset key          | Metadata path                                              |
| -------------------- | ---------------------------------------------------------- |
| `CAID`               | `/home/aviad/datasets/CAID/metadata.json`                  |
| `RWTD`               | `/home/aviad/datasets/RWTD/metadata.json`                  |
| `STLD`               | `/home/aviad/datasets/STLD/metadata.json`                  |
| `ADE20k_Detecture`   | `/home/aviad/datasets/ADE20k_Detecture/metadata.json`      |

Unified-schema entry:

```json
{
  "image_path": "/abs/path/image.jpg",
  "id": "sample_id",
  "textures": [
    {"description": "…", "mask_path": "/abs/path/mask_0.png"},
    {"description": "…", "mask_path": "/abs/path/mask_1.png"}
  ]
}
```

## Run

```bash
# Run every model on every dataset in the config
python master_runner.py

# One model, all datasets
python master_runner.py --model qwen2sam_zs

# One dataset, all models (smoke test with 10 samples)
python master_runner.py --dataset RWTD --limit 10

# One specific cell
python master_runner.py --model grounded_sam3 --dataset CAID --limit 5
```

Individual scripts can also run standalone:

```bash
python eval_sam3_vanilla.py   --dataset RWTD --output-dir out/sv_rwtd
python eval_grounded_sam3.py  --dataset CAID --output-dir out/gs_caid
python eval_vlm_end2end.py    --dataset STLD --backend sa2va --output-dir out/sa2va_stld
python eval_qwen2sam_zs.py    --dataset ADE20k_Detecture --output-dir out/qzs_ade
```

## Outputs

Per `<model>/<dataset>/` directory:

- `zero_shot_results.json` — summary block (mIoU, pIoU, Dice, ARI, coverage)
  + per-sample records.
- `vis/<id>.png` — identical-layout figures across every family: top row
  (image / GT overlay / pred overlay / metrics banner) + per-prediction
  rows (logits / pred mask / matched GT / contour overlay).

## Aggregation

After a run:

```bash
# Markdown tables on stdout + CSV + LaTeX
python aggregate_results.py --csv results/summary.csv --latex results/summary.tex

# Pick a different metric for the LaTeX snippet
python aggregate_results.py --metric mean_ari --latex results/summary_ari.tex
```

The narrative discussion for the paper (four paragraphs, references
`Table~\ref{tab:miou}` and `Table~\ref{tab:ari}`) lives in
[`results/discussion.tex`](./results/discussion.tex) — include it with
`\input{results/discussion.tex}` in your main manuscript.

## Pipeline summary per family

### `sam3_vanilla`

1. Read per-dataset text list from
   `prompts.sam3_vanilla.texts` (or `.text` in repeat / proposal mode).
2. For `mode: static` — each text independently routed through
   `backbone.forward_text` → fusion encoder → Semantic Seg Head.
3. For `mode: proposal` (ADE20K) — one text → SAM3's mask-proposal decoder
   → top-K masks by confidence. Avoids the collapse that happens when the
   same text is applied K times to the semantic head.
4. Softmax over (K+1, H, W) with a static dustbin channel.

### `grounded_sam3`

1. `IDEA-Research/grounding-dino-tiny` consumes one dot-separated phrase
   string (`"foreground texture . background texture ."`).
2. Top-K boxes by confidence → SAM3 geometry encoder → Semantic Seg Head
   (NOT the mask-proposal decoder).
3. Missing boxes get a `-1e4` logit pad so the dustbin absorbs the
   phantom channel and Hungarian charges 0 IoU.

### `sa2va` (end-to-end VLM)

1. `ByteDance/Sa2VA-4B` called with the interleaved-segmentation caption
   prompt it was trained on (`"<image>Please give a brief description …
   with interleaved segmentation masks …"`).
2. Each `[SEG]` token emitted in the answer produces one mask in
   `prediction_masks`.
3. Masks ranked by area as a confidence proxy, top-K kept, rest padded.

### `qwen2sam_zs` (our pipeline)

1. Qwen3-VL-8B generates exactly `K = K_GT` texture descriptions per image.
2. Each description passes through SAM3's text encoder → Semantic Seg Head.
3. Same softmax + Hungarian + ARI scoring as above.

## Hungarian + padding policy

`compute_sample_metrics` (see [`metrics_utils.py`](./metrics_utils.py))
does softmax over K + 1 channels, argmax → discrete partition, then
`scipy.optimize.linear_sum_assignment` on `1 - IoU`. Final metrics:

```
panoptic_iou = sum_matched(IoU) / max(K_pred, M_gt)
```

Under-production is penalised: a baseline that returns `k < K_GT` masks
has the remaining `K_GT - k` channels padded with `-1e4`, so those
channels collapse to dustbin, producing empty pred masks that the
Hungarian matcher pairs to the unassigned GT masks with IoU = 0. ARI is
computed directly on the partition (permutation-invariant; no Hungarian
pre-alignment needed).
