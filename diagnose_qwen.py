"""
Minimal Qwen3-VL forward-pass diagnostic.

Prints the chat-template output, input_id ranges, vocab size, image_grid_thw,
and pixel_values shape. Then tries two generate() calls:

    (1) plain small prompt — a literal cat-describe hello-world.
    (2) the same prompt + the first RWTD image + our real build_user_prompt(2).

If (1) succeeds and (2) fails we know the issue is in OUR prompt / image path.
If BOTH fail, Qwen3-VL itself is mis-installed on 179 and the fix is in the env.

Run:
  cd ~/Qwen2SAM_Detecture_Benchmark
  python diagnose_qwen.py
  # optional: point at a specific image
  python diagnose_qwen.py --image /home/aviad/datasets/RWTD/images/1.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image


def banner(msg: str) -> None:
    print("\n" + "=" * 76 + f"\n  {msg}\n" + "=" * 76)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--image", default="/home/aviad/datasets/RWTD/images/1.jpg",
        help="Image used for test (2).",
    )
    args = parser.parse_args()

    banner("env")
    import transformers
    import numpy as np
    print(f"  transformers : {transformers.__version__}")
    print(f"  torch        : {torch.__version__}  cuda={torch.version.cuda}")
    print(f"  numpy        : {np.__version__}")

    banner("transformers Qwen symbols")
    qwen_syms = [s for s in dir(transformers) if "Qwen" in s]
    for s in qwen_syms:
        print(f"  {s}")

    banner("loading model + processor")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(args.model, torch_dtype=torch.bfloat16)
        .to("cuda").eval()
    )
    processor = AutoProcessor.from_pretrained(args.model)
    vocab_size = getattr(
        getattr(model.config, "text_config", model.config),
        "vocab_size", None,
    )
    print(f"  model class   : {type(model).__name__}")
    print(f"  processor cls : {type(processor).__name__}")
    print(f"  vocab_size    : {vocab_size}")
    tok = processor.tokenizer
    print(f"  tokenizer     : {type(tok).__name__}  len={len(tok)}")

    try:
        from qwen_vl_utils import process_vision_info
        have_qvlu = True
    except Exception as e:  # noqa: BLE001
        print(f"  qwen_vl_utils import FAILED: {e!r}")
        have_qvlu = False

    # ------------------------------------------------------------------ #
    # (1) Text-only hello-world
    # ------------------------------------------------------------------ #
    banner("TEST 1 — text-only")
    messages1 = [
        {"role": "user", "content": [{"type": "text", "text": "Say hi in 5 words."}]},
    ]
    text1 = processor.apply_chat_template(
        messages1, tokenize=False, add_generation_prompt=True,
    )
    print(f"  chat text ({len(text1)} chars):")
    print("  " + text1.replace("\n", "\n  "))
    inputs1 = processor(text=[text1], return_tensors="pt", padding=True)
    inputs1.pop("token_type_ids", None)
    inputs1 = {k: v.to("cuda") for k, v in inputs1.items()}
    print(f"  input_ids shape={tuple(inputs1['input_ids'].shape)}  "
          f"min={int(inputs1['input_ids'].min())}  "
          f"max={int(inputs1['input_ids'].max())}")
    try:
        with torch.no_grad():
            out1 = model.generate(**inputs1, max_new_tokens=20, do_sample=False)
        gen1 = processor.tokenizer.decode(
            out1[0, inputs1["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        print(f"  TEST 1 OK → {gen1!r}")
    except Exception as e:  # noqa: BLE001
        print(f"  TEST 1 FAILED → {e!r}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # (2) Image + our actual prompt
    # ------------------------------------------------------------------ #
    if not Path(args.image).exists():
        print(f"  skipping TEST 2 — image not found: {args.image}")
        return

    banner("TEST 2 — image + our build_user_prompt(2)")
    # Load our prompt builder
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_zero_shot_pipeline import SYSTEM_PROMPT, build_user_prompt
    image = Image.open(args.image).convert("RGB")
    print(f"  image : {args.image}  size={image.size}")

    messages2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": build_user_prompt(2)},
        ]},
    ]
    text2 = processor.apply_chat_template(
        messages2, tokenize=False, add_generation_prompt=True,
    )
    print(f"  chat text length={len(text2)} chars")
    # Show image-pad-token span
    print("  chat text tail (400 chars):")
    print("  " + text2[-400:].replace("\n", "\n  "))

    if have_qvlu:
        image_inputs, video_inputs = process_vision_info(messages2)
        inputs2 = processor(
            text=[text2], images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=True,
        )
    else:
        inputs2 = processor(
            text=[text2], images=[image],
            return_tensors="pt", padding=True,
        )
    inputs2.pop("token_type_ids", None)

    print(f"  input_ids shape={tuple(inputs2['input_ids'].shape)}  "
          f"min={int(inputs2['input_ids'].min())}  "
          f"max={int(inputs2['input_ids'].max())}")
    if vocab_size is not None and int(inputs2['input_ids'].max()) >= vocab_size:
        print(f"  !! input_ids.max() >= vocab_size ({vocab_size}) — "
              "embedding OOB almost certainly here")
    print(f"  image_grid_thw : {inputs2.get('image_grid_thw')}")
    print(f"  pixel_values   : "
          f"{tuple(inputs2['pixel_values'].shape) if 'pixel_values' in inputs2 else None}")

    # Count image-pad tokens in input_ids vs what image_grid_thw implies
    img_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    vid_pad_id = tok.convert_tokens_to_ids("<|video_pad|>")
    n_img_pad = int((inputs2["input_ids"] == img_pad_id).sum())
    print(f"  <|image_pad|> token id={img_pad_id}  count={n_img_pad}")
    if inputs2.get("image_grid_thw") is not None:
        thw = inputs2["image_grid_thw"][0].tolist()
        spatial_merge = getattr(model.config, "spatial_merge_size", None) or 2
        expected = int(thw[0] * thw[1] * thw[2] / (spatial_merge ** 2))
        print(f"  image_grid_thw={thw}  spatial_merge={spatial_merge}  "
              f"expected #pad tokens={expected}  actual={n_img_pad}")
        if expected != n_img_pad:
            print("  !! MISMATCH between image_grid_thw and #<|image_pad|> tokens — "
                  "this is the CUDA gather OOB source")

    inputs2 = {k: v.to("cuda") for k, v in inputs2.items()}
    try:
        with torch.no_grad():
            out2 = model.generate(**inputs2, max_new_tokens=40, do_sample=False)
        gen2 = processor.tokenizer.decode(
            out2[0, inputs2["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        print(f"\n  TEST 2 OK → {gen2[:200]!r}")
    except Exception as e:  # noqa: BLE001
        print(f"\n  TEST 2 FAILED → {e!r}")


if __name__ == "__main__":
    main()
