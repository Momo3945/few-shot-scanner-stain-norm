#!/usr/bin/env python3
"""
normalize_lizard.py -- SD1.5 + colour LoRA + ControlNet-Canny + LCM-LoRA img2img
normalisation over a flat directory of Lizard images (P2-08, Structural Safety,
step 5). Produces the "B" side of the G/A/B triangle: run hovernet_wrapper.py +
lizard_dice.py (with --against) on this script's output to compute Relative Dice
= Dice(B,G)/Dice(A,G).

Not a reuse of infer_colour_lora.py: Lizard images are single unpaired images
(no Aperio/Hamamatsu registration), and lizard_dice.py requires the normalised
output to match its source image's (H, W) EXACTLY (it loads Lizard's .mat ground
truth in the original image's pixel coordinates and skips any image where
pred_mask.shape != gt_mask.shape -- see lizard_dice.py:154). infer_colour_lora.py
writes independent scoring crops, not one reassembled full-resolution image, so
it can't produce that directly.

Approach: pad each image to a multiple of --crop with reflect padding, tile into
non-overlapping crop x crop blocks (deliberately NOT infer_colour_lora.py's
grid_offsets(), which allows overlap for independent crop sampling -- overlap
here would double-process pixels and create visible seams when reassembled),
run img2img + ControlNet-Canny conditioning per tile, write each result tile
directly into the output canvas (disjoint tiles, no blending needed), then crop
back to the original (H, W) and save as <out>/<source_stem>.png.

Known limitation, not engineered around: independent per-tile denoising can
leave faint seams at tile boundaries. Not addressed with overlap-blending --
the default strength (0.20) is ~1 real LCM step (near input-preserving, the
mildest setting anywhere in the ablation ladder per P1-09), so seam risk is
low, and this is a one-shot research eval, not a production pipeline.

Default config is the P1-09 best general-purpose operating point (1-step LCM,
strength 0.20): colour LoRA (a2h_r8) + ControlNet-Canny + LCM-LoRA, matching
slurm/infer_a4_lcm.slurm's A4 config.

Usage
-----
    python normalize_lizard.py \
        --images-dir lizard_heldout/images \
        --lora lora/a2h_r8/final \
        --out eval/lizard_normalised_images \
        --strength 0.20 --steps 8 --guidance 1.5 --limit 5

Dependencies: torch, diffusers, numpy, Pillow, opencv-python-headless.
Requires canny.py on the path (same folder).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Colour-LoRA + ControlNet + LCM img2img normalisation over a flat image directory.")
    ap.add_argument("--images-dir", required=True, help="Directory of source images.")
    ap.add_argument("--out", required=True, help="Output dir for normalised <stem>.png files.")
    ap.add_argument("--lora", default=None,
                    help="Path to trained colour-LoRA dir (contains pytorch_lora_weights.safetensors). "
                         "Omit to run without the colour LoRA.")
    ap.add_argument("--controlnet", default="lllyasviel/sd-controlnet-canny",
                    help="ControlNet repo id, resolved from HF_HOME cache. Empty string disables it.")
    ap.add_argument("--controlnet-scale", type=float, default=1.0)
    ap.add_argument("--lcm", action="store_true", default=True,
                    help="Attach the pretrained LCM-LoRA and switch to LCMScheduler (on by default -- "
                         "this script's whole point is the P1-09 best operating point, which is LCM).")
    ap.add_argument("--no-lcm", dest="lcm", action="store_false")
    ap.add_argument("--lcm-scale", type=float, default=1.0)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--strength", type=float, default=0.20)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all).")
    ap.add_argument("--online", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    from PIL import Image
    from diffusers import DDIMScheduler
    if args.lcm:
        from diffusers import LCMScheduler
    if args.controlnet:
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
    else:
        from diffusers import StableDiffusionImg2ImgPipeline

    if args.controlnet:
        from canny import extract_canny_control_image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("No CUDA device -- inference must run on a GPU node.")

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- pipeline: SD 1.5 img2img, optionally + colour LoRA, optionally + ControlNet,
    # optionally + LCM-LoRA (DDIM otherwise) -- mirrors infer_colour_lora.py's loading ----
    multi_lora = args.lcm and bool(args.lora)
    label = " + ".join(filter(None, [
        "LoRA" if args.lora else None, f"ControlNet({args.controlnet})" if args.controlnet else None,
        "LCM-LoRA" if args.lcm else None,
    ])) or "no adapters"
    print(f"Loading SD 1.5 img2img pipeline: {label} ...")
    if args.controlnet:
        controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            args.model, controlnet=controlnet, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False)
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.model, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = (LCMScheduler if args.lcm else DDIMScheduler).from_config(pipe.scheduler.config)
    active, weights = [], []
    if args.lora:
        lora_kwargs = {"weight_name": "pytorch_lora_weights.safetensors"}
        if multi_lora:
            lora_kwargs["adapter_name"] = "colour"
        pipe.load_lora_weights(args.lora, **lora_kwargs)
        active.append("colour"); weights.append(1.0)
    if args.lcm:
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5",
                                weight_name="pytorch_lora_weights.safetensors",
                                adapter_name="lcm" if multi_lora else None)
        if multi_lora:
            active.append("lcm"); weights.append(args.lcm_scale)
    if multi_lora:
        pipe.set_adapters(active, adapter_weights=weights)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    def pad_to_multiple(rgb: "np.ndarray", crop: int) -> "np.ndarray":
        h, w = rgb.shape[:2]
        ph = (-h) % crop
        pw = (-w) % crop
        if ph == 0 and pw == 0:
            return rgb
        mode = "reflect" if ph < h and pw < w else "edge"
        return np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode=mode)

    def run_tile(tile_rgb: "np.ndarray") -> "np.ndarray":
        tile_pil = Image.fromarray(tile_rgb)
        control_kwargs = {}
        if args.controlnet:
            control_kwargs = {
                "control_image": Image.fromarray(extract_canny_control_image(tile_rgb)),
                "controlnet_conditioning_scale": args.controlnet_scale,
            }
        gen = torch.Generator(device=device).manual_seed(args.seed)
        out = pipe(prompt=args.prompt, image=tile_pil, strength=args.strength,
                   num_inference_steps=args.steps, guidance_scale=args.guidance,
                   generator=gen, **control_kwargs).images[0]
        return np.asarray(out)

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    )
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    print(f"Normalising {len(image_paths)} images from {images_dir} "
          f"(strength={args.strength}, steps={args.steps}, crop={args.crop})")

    crop = args.crop
    n_ok = 0
    for p in image_paths:
        src = np.asarray(Image.open(p).convert("RGB"))
        h, w = src.shape[:2]
        padded = pad_to_multiple(src, crop)
        ph, pw = padded.shape[:2]

        canvas = np.zeros_like(padded)
        for y in range(0, ph, crop):
            for x in range(0, pw, crop):
                tile = padded[y:y + crop, x:x + crop]
                canvas[y:y + crop, x:x + crop] = run_tile(tile)

        result = canvas[:h, :w]
        if result.shape[:2] != (h, w):
            raise RuntimeError(f"{p.name}: stitched shape {result.shape[:2]} != source {(h, w)}")
        Image.fromarray(result).save(out_dir / f"{p.stem}.png")
        n_ok += 1
        print(f"  {p.stem}: {h}x{w} -> {(ph // crop) * (pw // crop)} tiles")

    print(f"\nNormalised {n_ok}/{len(image_paths)} images into {out_dir}")
    print("Next: sbatch infer_hovernet.slurm lizard_normalised <this out dir> then "
          "sbatch score_lizard.slurm lizard_normalised lizard_original")


if __name__ == "__main__":
    main()
