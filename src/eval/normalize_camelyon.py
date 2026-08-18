#!/usr/bin/env python3
"""
normalize_camelyon.py -- SD1.5 + colour LoRA + ControlNet-Canny + LCM-LoRA
img2img normalisation over CAMELYON17 patches (P2-10, multi-centre
generalisation). Produces the "after" (D_post) side of P2-10's pairwise
LAB-Wasserstein measurement -- score with score_camelyon_wasserstein.py after.

Unlike normalize_lizard.py (P2-08), this needs NO padding/tiling/stitching:
every CAMELYON17 patch is already exactly --crop-sized (512x512 by
construction, see camelyon_patches.py's extraction), so one patch = one
img2img call directly. Preserves the centre_<i>_patient_<id>/ subdirectory
structure between input and output so score_camelyon_wasserstein.py can group
identically on both raw and normalised sets.

Default config is the same P1-09 best general-purpose operating point used for
every other normalisation run this session (P2-07 round-trip, P2-08 Lizard):
colour LoRA (a2h_r8) + ControlNet-Canny + LCM-LoRA, strength 0.20, 8 steps,
guidance 1.5. Applied uniformly to all 5 centres -- the LoRA was trained on one
scanner pair (Aperio->Hamamatsu) and never saw any CAMELYON17 centre; whether
it still pulls unseen scanners toward a common colour regime is exactly what
D_post < D_pre (score_camelyon_wasserstein.py's --against) tests.

Usage
-----
    python normalize_camelyon.py \
        --images-dir camelyon17_patches \
        --lora lora/a2h_r8/final \
        --out eval/camelyon17_normalised \
        --strength 0.20 --steps 8 --guidance 1.5 --limit 10

Dependencies: torch, diffusers, numpy, Pillow, opencv-python-headless.
Requires canny.py on the path (same folder).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Colour-LoRA + ControlNet + LCM img2img normalisation over CAMELYON17 patches.")
    ap.add_argument("--images-dir", required=True,
                    help="Root containing centre_<0-4>_patient_<id>/*.png.")
    ap.add_argument("--out", required=True, help="Output dir for normalised patches (same subdir layout).")
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
    ap.add_argument("--crop", type=int, default=512, help="Expected patch size (sanity check only, no tiling).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="Max patches to process (0 = all).")
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

    # ---- pipeline: identical loading pattern to normalize_lizard.py/infer_colour_lora.py ----
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

    def run_patch(rgb):
        pil = Image.fromarray(rgb)
        control_kwargs = {}
        if args.controlnet:
            control_kwargs = {
                "control_image": Image.fromarray(extract_canny_control_image(rgb)),
                "controlnet_conditioning_scale": args.controlnet_scale,
            }
        gen = torch.Generator(device=device).manual_seed(args.seed)
        out = pipe(prompt=args.prompt, image=pil, strength=args.strength,
                   num_inference_steps=args.steps, guidance_scale=args.guidance,
                   generator=gen, **control_kwargs).images[0]
        return np.asarray(out)

    image_paths = sorted(images_dir.glob("centre_*_patient_*/*.png"))
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No centre_*_patient_*/*.png patches found under {images_dir}")

    print(f"Normalising {len(image_paths)} CAMELYON17 patches from {images_dir} "
          f"(strength={args.strength}, steps={args.steps})")

    n_ok = 0
    for p in image_paths:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        if rgb.shape[:2] != (args.crop, args.crop):
            print(f"  WARNING: {p} is {rgb.shape[:2]}, expected ({args.crop},{args.crop}) -- processing anyway.")
        result = run_patch(rgb)
        dest_dir = out_dir / p.parent.name  # centre_<i>_patient_<id>
        dest_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(dest_dir / p.name)
        n_ok += 1
        if n_ok % 100 == 0 or n_ok == len(image_paths):
            print(f"  {n_ok}/{len(image_paths)} done")

    print(f"\nNormalised {n_ok}/{len(image_paths)} patches into {out_dir}")
    print("Next: score with score_camelyon_wasserstein.py --against <D_pre pairwise.csv>")


if __name__ == "__main__":
    main()
