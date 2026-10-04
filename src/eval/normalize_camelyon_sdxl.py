#!/usr/bin/env python3
"""
normalize_camelyon_sdxl.py -- P3-09: SDXL counterpart to normalize_camelyon.py
(P2-10, which is hardcoded SD1.5 + plain Canny-ControlNet, no source
conditioning). Applies P3-07c's source-conditioned colour LoRA + fresh
6-channel ControlNet, img2img, over CAMELYON17 patches -- the multi-centre
generalisation test P2-10 ran on the plain config, now on the architecture
that produced this project's best SDXL result (P3-08, after fusion).
See tickets/PHASE3-TICKETS.md P3-09.

No pairing needed, same as normalize_camelyon.py: CAMELYON17 has no
Aperio/Hamamatsu labels at all, so each patch is translated using itself
as both the img2img target AND its own structural condition (source RGB +
Canny of that same patch) -- this is exactly what infer_colour_
translation_sdxl.py's --source-mode=correct already does for held-out
crops (it conditions on the SAME image being translated, never a genuinely
different "partner" image), so this script needs no registration, no
grid-tiling, no heldout_frames.csv -- simpler than that script, not a
cut-down version of it. Model-loading and control-tensor construction are
copied verbatim from infer_colour_translation_sdxl.py (same dtype/VAE/
ControlNet/LoRA wiring -- nothing about that changes for unpaired patches).

Direction: A2H forced (CAMELYON17 has no scanner-pair labels) -- mirrors
normalize_camelyon.py's own fixed-direction convention exactly.

Checkpoint: lora/a2h_cond_r8_sdxl_1024_overlap144/final (P3-07c) -- the
checkpoint behind P3-08's own result. Defaults (strength=0.50, steps=50,
guidance=2.0) match P3-07c's own established held-out operating point, no
new tuning.

Usage
-----
    # vae_only floor (no checkpoint needed, feeds fuse_camelyon_patches.py's A_V):
    python normalize_camelyon_sdxl.py --vae-only \
        --images-dir camelyon17_patches_1024 --out eval/camelyon17_sdxl_vae_only

    # full translation (feeds fuse_camelyon_patches.py's H_pred):
    python normalize_camelyon_sdxl.py \
        --lora lora/a2h_cond_r8_sdxl_1024_overlap144/final \
        --controlnet lora/a2h_cond_r8_sdxl_1024_overlap144/final/controlnet \
        --images-dir camelyon17_patches_1024 --out eval/camelyon17_sdxl_correct

Dependencies: torch, diffusers, numpy, Pillow, opencv-python-headless.
Requires canny.py on the path (same folder).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="P3-09: SDXL source-conditioned img2img normalisation over CAMELYON17 patches.")
    ap.add_argument("--images-dir", required=True,
                    help="Root containing centre_<0-4>_patient_<id>/*.png.")
    ap.add_argument("--out", required=True, help="Output dir for normalised patches (same subdir layout).")
    ap.add_argument("--lora", default=None,
                    help="Path to the trained SDXL LoRA checkpoint dir "
                         "(e.g. lora/a2h_cond_r8_sdxl_1024_overlap144/final). Required unless --vae-only.")
    ap.add_argument("--controlnet", default=None,
                    help="Path to the trained SDXL ControlNet dir (its sibling /controlnet). "
                         "Required unless --vae-only.")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--vae-only", action="store_true",
                    help="VAE-only floor control (feeds fuse_camelyon_patches.py's A_V): skip "
                         "UNet/ControlNet/LoRA entirely, VAE encode+decode only.")
    ap.add_argument("--controlnet-scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps, matches P3-07c's own default.")
    ap.add_argument("--strength", type=float, default=0.50,
                    help="img2img strength, matches P3-07c's own established operating point.")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--crop", type=int, default=1024, help="Expected patch size (sanity check only).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Single seed, matching normalize_camelyon.py's own convention -- P2-10's "
                         "generalisation test is a before/after pairwise-distance comparison, not a "
                         "per-crop metric needing multi-seed spread.")
    ap.add_argument("--limit", type=int, default=0, help="Max patches to process (0 = all).")
    ap.add_argument("--online", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.vae_only and (not args.lora or not args.controlnet):
        raise SystemExit("--lora and --controlnet are required unless --vae-only is set.")
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("No CUDA device -- inference must run on a GPU node.")

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Model: exact loading pattern from infer_colour_translation_sdxl.py --
    # nothing about this changes for unpaired patches, only how the loop
    # below supplies images (itself, not a registered-pair grid).
    # ------------------------------------------------------------------
    if args.vae_only:
        from diffusers import AutoencoderKL
        print("Loading SDXL VAE only (fp32 -- SDXL's official VAE NaNs under fp16) ...")
        vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32).to(device)
        vae.eval()
        scaling = vae.config.scaling_factor

        @torch.no_grad()
        def run_patch(rgb):
            torch.manual_seed(args.seed)
            arr = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
            arr = arr.unsqueeze(0).to(device, dtype=torch.float32)
            latents = vae.encode(arr).latent_dist.mode() * scaling
            decoded = vae.decode(latents / scaling).sample
            out = ((decoded[0].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            return out
    else:
        from diffusers import (AutoencoderKL, ControlNetModel, DDIMScheduler,
                               StableDiffusionXLControlNetImg2ImgPipeline)
        from canny import extract_canny_control_image
        print(f"Loading P3-07c SDXL pipeline: LoRA={args.lora}  ControlNet={args.controlnet} ...")
        vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
        controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            args.model, controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors",
                               adapter_name="colour")
        pipe.set_adapters(["colour"], adapter_weights=[1.0])
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        def build_control_tensor(rgb):
            canny = extract_canny_control_image(rgb)
            rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            canny_t = torch.from_numpy(canny.astype(np.float32) / 255.0).permute(2, 0, 1)
            return torch.cat([rgb_t, canny_t], dim=0).unsqueeze(0)

        def run_patch(rgb):
            # Self-conditioning: the same patch is both the img2img target
            # and its own structural condition -- exactly --source-mode=
            # correct's semantics in infer_colour_translation_sdxl.py.
            control_tensor = build_control_tensor(rgb)
            gen = torch.Generator(device=device).manual_seed(args.seed)
            out = pipe(prompt=args.prompt, image=Image.fromarray(rgb), strength=args.strength,
                       num_inference_steps=args.steps, guidance_scale=args.guidance,
                       control_image=control_tensor, controlnet_conditioning_scale=args.controlnet_scale,
                       generator=gen).images[0]
            return np.asarray(out)

    image_paths = sorted(images_dir.glob("centre_*_patient_*/*.png"))
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No centre_*_patient_*/*.png patches found under {images_dir}")

    mode_label = "vae_only" if args.vae_only else "correct"
    print(f"Normalising {len(image_paths)} CAMELYON17 patches from {images_dir} "
          f"(mode={mode_label}, strength={args.strength}, steps={args.steps})")

    n_ok = n_skipped = 0
    for p in image_paths:
        dest_dir = out_dir / p.parent.name
        dest_path = dest_dir / p.name
        if dest_path.exists():
            n_skipped += 1
            continue
        rgb = np.asarray(Image.open(p).convert("RGB"))
        if rgb.shape[:2] != (args.crop, args.crop):
            print(f"  WARNING: {p} is {rgb.shape[:2]}, expected ({args.crop},{args.crop}) -- processing anyway.")
        result = run_patch(rgb)
        dest_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(dest_path)
        n_ok += 1
        if n_ok % 100 == 0:
            print(f"  {n_ok + n_skipped}/{len(image_paths)} done ({n_skipped} skipped, already existed)")

    print(f"\nNormalised {n_ok}/{len(image_paths)} patches this run "
          f"({n_skipped} already existed, skipped -- resumable) into {out_dir}")
    print("Next (both modes done): fuse_camelyon_patches.py, then score_camelyon_wasserstein.py")


if __name__ == "__main__":
    main()
