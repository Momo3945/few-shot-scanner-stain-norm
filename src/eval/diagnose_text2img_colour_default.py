#!/usr/bin/env python3
"""
diagnose_text2img_colour_default.py -- the actual isolation test for the
backbone-colour-bias hypothesis that diagnose_direction_colour_bias.py's
`zero`-mode reuse failed to isolate (see tickets/PHASE3-TICKETS.md
P3-06b/P3-07 H2A's "D1" diagnostic -- that test found EVERY combo's zero-mode
output leans toward its own SOURCE domain, revealed to be an img2img
source-latent-persistence artefact, not a colour-free backbone default,
because `zero` mode still runs a real img2img pass from the source image's
noised latent).

This script removes that confound directly: pure text-to-image generation
(fixed prompt + trained colour LoRA only, NO ControlNet component loaded at
all, NO source image, starting from full random noise) -- there is no
source latent to leak, so whatever colour comes out reflects only the
frozen backbone + trained colour LoRA's own generative tendency, exactly
the quantity D1 was trying (and failing) to isolate.

Only the colour LoRA is loaded from each P1-10/P3-06/P3-07 checkpoint
(`pytorch_lora_weights.safetensors`, saved directly in the checkpoint dir,
separate from the ControlNet weights in its own `controlnet/` subfolder --
confirmed from how infer_colour_translation[_sdxl].py load these two
components independently). The ControlNet is never loaded here: with no
source image there is nothing for it to condition on, and this script's
entire point is to isolate the LoRA+backbone's behaviour without it.

Usage
-----
    python diagnose_text2img_colour_default.py --backbone sd15 \
        --lora /datasets/mhoosen/stain-norm/lora/a2h_cond_r8/best \
        --resolution 512 --out eval/diagnose_t2i/sd15_a2h --n-samples 24

    python diagnose_text2img_colour_default.py --backbone sdxl \
        --lora /datasets/mhoosen/stain-norm/lora/h2a_cond_r8_sdxl_1024/final \
        --resolution 1024 --out eval/diagnose_t2i/sdxl1024_h2a --n-samples 24

Dependencies: torch, diffusers, transformers, peft, safetensors, Pillow, numpy.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Pure text2img colour-default probe (colour LoRA only, no ControlNet, no source image).")
    ap.add_argument("--backbone", choices=["sd15", "sdxl"], required=True)
    ap.add_argument("--lora", required=True, help="Checkpoint dir (contains pytorch_lora_weights.safetensors).")
    ap.add_argument("--model", default=None,
                    help="HF repo id override. Default: stable-diffusion-v1-5/stable-diffusion-v1-5 (sd15) "
                         "or stabilityai/stable-diffusion-xl-base-1.0 (sdxl).")
    ap.add_argument("--resolution", type=int, default=512, help="512 for sd15/SDXL@512, 1024 for SDXL@1024.")
    ap.add_argument("--out", required=True, help="Output dir for generated crops.")
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--steps", type=int, default=50,
                    help="Matches this project's own infer_colour_translation[_sdxl].py default.")
    ap.add_argument("--guidance", type=float, default=2.0,
                    help="Matches this project's own infer_colour_translation[_sdxl].py default.")
    ap.add_argument("--n-samples", type=int, default=24, help="Matches the zero-mode ablation's own crop count.")
    ap.add_argument("--seed0", type=int, default=0, help="Samples use seeds seed0..seed0+n_samples-1.")
    ap.add_argument("--online", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    local_only = not args.online

    import torch
    from diffusers import AutoencoderKL, StableDiffusionPipeline, StableDiffusionXLPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("No CUDA device -- this probe must run on a GPU node.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backbone == "sd15":
        model = args.model or "stable-diffusion-v1-5/stable-diffusion-v1-5"
        print(f"Loading frozen SD1.5 text2img pipeline ({model}), LoRA-only (no ControlNet) ...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model, torch_dtype=torch.float16, safety_checker=None,
            requires_safety_checker=False, local_files_only=local_only)
    else:
        model = args.model or "stabilityai/stable-diffusion-xl-base-1.0"
        print(f"Loading frozen SDXL text2img pipeline ({model}), LoRA-only (no ControlNet) ...")
        # VAE forced to fp32 -- SDXL's official VAE NaNs under fp16 (same
        # precedent as every other SDXL script in this project).
        vae = AutoencoderKL.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32,
                                            local_files_only=local_only)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model, vae=vae, torch_dtype=torch.float16, local_files_only=local_only)

    # weight_name must be explicit: diffusers normally auto-detects it via a Hub
    # API call, unavailable under HF_HUB_OFFLINE=1 (set above deliberately).
    # NOTE: these checkpoints' ControlNet weights (in <lora>/controlnet/) are
    # deliberately NOT loaded -- this script's entire point is the LoRA +
    # frozen backbone's behaviour with no ControlNet/source-image at all.
    pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    print(f"Generating {args.n_samples} pure text2img samples at {args.resolution}x{args.resolution} "
          f"(steps={args.steps}, guidance={args.guidance}) ...")
    for i in range(args.n_samples):
        seed = args.seed0 + i
        gen = torch.Generator(device=device).manual_seed(seed)
        out = pipe(prompt=args.prompt, height=args.resolution, width=args.resolution,
                   num_inference_steps=args.steps, guidance_scale=args.guidance,
                   generator=gen).images[0]
        out_path = out_dir / f"t2i_seed{seed}.png"
        out.save(out_path)

    print(f"\nWrote {args.n_samples} text2img samples to {out_dir}")


if __name__ == "__main__":
    main()
