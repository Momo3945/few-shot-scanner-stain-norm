#!/usr/bin/env python3
"""
normalize_lizard_inversion_fusion.py -- P2-08b: run the FINAL SD1.5 pipeline (P1-10 source-conditioned
colour LoRA + ControlNet, P1-11 DDIM inversion, P1-16 F3 fusion) over the Lizard structural-safety images,
so Relative Dice (H3) can be measured on the configuration the paper reports, not only on the P1-09
operating point P2-08 used. See tickets/PHASE2-TICKETS.md P2-08b.

Writes two "B" sides for lizard_dice.py, from one GPU pass:
  <out-fused>/<stem>.png    F3 fusion (sigma=8, beta=0.50) of the stitched prediction onto the raw image
  <out-unfused>/<stem>.png  the stitched DDIM-inversion prediction itself (P1-11, no fusion)

Tiling mirrors normalize_lizard.py exactly (reflect-pad to a multiple of --crop, disjoint crop x crop
tiles, stitch, crop back to the source (H, W); lizard_dice.py requires identical shape). The diffusion core
(control tensor, CFG noise prediction, deterministic .mode() encode, DDIM inversion, DDIM reconstruction)
is copied from infer_colour_source_ddim_inversion.py, whose helpers are nested in its main() and cannot be
imported; that validated script is not edited. Fusion runs once on the STITCHED full image with the
validated fuse_f3 imported from fuse_source_detail.py, so the fusion step adds no tile seams.
A_V (the VAE-only reconstruction F3 subtracts) is the stock decoder's, matching P1-16.

Usage
-----
    python normalize_lizard_inversion_fusion.py \
        --lora <data>/lora/a2h_cond_r8/best --controlnet <data>/lora/a2h_cond_r8/best/controlnet \
        --images-dir <data>/lizard_heldout/images \
        --out-fused <data>/eval/lizard_normalised_images_inv_f3 \
        --out-unfused <data>/eval/lizard_normalised_images_inv [--limit 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Final SD1.5 pipeline (inversion + F3) over Lizard images.")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--controlnet", required=True)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-fused", required=True)
    ap.add_argument("--out-unfused", required=True)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--inversion-fraction", type=float, default=1.0)
    ap.add_argument("--inversion-steps", type=int, default=50)
    ap.add_argument("--translation-steps", type=int, default=50)
    ap.add_argument("--inversion-guidance", type=float, default=1.0)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--controlnet-scale", type=float, default=1.0)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--sigma", type=float, default=8.0, help="F3 blur sigma (P1-16 frozen value).")
    ap.add_argument("--beta", type=float, default=0.50, help="F3 residual weight (P1-16 frozen value).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="Max images (0 = all).")
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

    from canny import extract_canny_control_image
    from fuse_source_detail import fuse_f3

    if not torch.cuda.is_available():
        raise SystemExit("ABORT: no CUDA device -- inference must run on a GPU node.")
    device, dtype = "cuda", torch.float16

    out_fused, out_unfused = Path(args.out_fused), Path(args.out_unfused)
    out_fused.mkdir(parents=True, exist_ok=True)
    out_unfused.mkdir(parents=True, exist_ok=True)

    # ---- pipeline + schedulers (as infer_colour_source_ddim_inversion.py) -------------------------
    from diffusers import (ControlNetModel, DDIMInverseScheduler, DDIMScheduler,
                           StableDiffusionControlNetImg2ImgPipeline)
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.model, controlnet=controlnet, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = ddim_scheduler
    if ddim_scheduler.config.prediction_type != "epsilon":
        raise SystemExit(f"P1-10 was trained with epsilon prediction; scheduler says "
                         f"{ddim_scheduler.config.prediction_type!r}. Refusing to run.")
    inverse_scheduler = DDIMInverseScheduler.from_config(ddim_scheduler.config)
    pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    vae, unet, text_encoder, tokenizer = pipe.vae, pipe.unet, pipe.text_encoder, pipe.tokenizer
    scaling = vae.config.scaling_factor

    with torch.no_grad():
        def encode_prompt(text):
            tok = tokenizer(text, padding="max_length", truncation=True,
                            max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
            return text_encoder(tok.input_ids)[0].to(dtype)
        cond_embeds = encode_prompt(args.prompt)
        uncond_embeds = encode_prompt("")

    def build_control_tensor(rgb):
        canny = extract_canny_control_image(rgb)
        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        canny_t = torch.from_numpy(canny.astype(np.float32) / 255.0).permute(2, 0, 1)
        return torch.cat([rgb_t, canny_t], dim=0).unsqueeze(0).to(device, dtype=dtype)  # 1x6xHxW, [0,1]

    def predict_noise(latent, t, control, guidance_scale):
        if guidance_scale is None or guidance_scale <= 1.0:
            down_res, mid_res = controlnet(latent, t, encoder_hidden_states=cond_embeds, controlnet_cond=control,
                                           conditioning_scale=args.controlnet_scale, return_dict=False)
            return unet(latent, t, encoder_hidden_states=cond_embeds, down_block_additional_residuals=down_res,
                        mid_block_additional_residual=mid_res).sample
        latent_in = torch.cat([latent, latent], dim=0)
        embeds_in = torch.cat([uncond_embeds, cond_embeds], dim=0)
        control_in = torch.cat([control, control], dim=0)
        down_res, mid_res = controlnet(latent_in, t, encoder_hidden_states=embeds_in, controlnet_cond=control_in,
                                       conditioning_scale=args.controlnet_scale, return_dict=False)
        noise_pred = unet(latent_in, t, encoder_hidden_states=embeds_in, down_block_additional_residuals=down_res,
                          mid_block_additional_residual=mid_res).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)
        return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    def encode_latent(rgb):
        arr = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        return vae.encode(arr.unsqueeze(0).to(device, dtype=dtype)).latent_dist.mode() * scaling

    def decode_latent(latent):
        decoded = vae.decode(latent / scaling).sample
        if not torch.isfinite(decoded).all():
            raise RuntimeError("NaN/Inf in decoded output -- aborting.")
        return ((decoded[0].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    def invert(latent0, control):
        inverse_scheduler.set_timesteps(args.inversion_steps, device=device)
        k = max(1, round(args.inversion_steps * args.inversion_fraction))
        used = inverse_scheduler.timesteps[:k]
        latent = latent0
        for t in used:
            latent = inverse_scheduler.step(predict_noise(latent, t, control, args.inversion_guidance),
                                            t, latent, return_dict=True).prev_sample
        if not torch.isfinite(latent).all():
            raise RuntimeError("NaN/Inf in inverted latent -- aborting.")
        return latent, int(used[-1].item())

    def reconstruct(latent, t_start, control):
        ddim_scheduler.set_timesteps(args.translation_steps, device=device)
        fwd = ddim_scheduler.timesteps
        idx = len(fwd) - 1
        for i, tv in enumerate(fwd):
            if tv.item() <= t_start:
                idx = i
                break
        for t in fwd[idx:]:
            latent = ddim_scheduler.step(predict_noise(latent, t, control, args.guidance),
                                         t, latent, eta=0.0, return_dict=True).prev_sample
        if not torch.isfinite(latent).all():
            raise RuntimeError("NaN/Inf in reconstructed latent -- aborting.")
        return latent

    def seed_for(key):
        return int(hashlib.sha256(f"{key}|{args.seed}".encode()).hexdigest()[:8], 16)

    @torch.no_grad()
    def run_tile(tile, key):
        torch.manual_seed(seed_for(key))  # eta=0 DDIM injects no noise; kept for convention
        control = build_control_tensor(tile)  # correct source, for both inversion and reconstruction
        latent0 = encode_latent(tile)
        latent_inv, t_end = invert(latent0, control)
        h_pred = decode_latent(reconstruct(latent_inv, t_end, control))
        a_v = decode_latent(latent0)  # stock VAE-only reconstruction of the source tile
        return h_pred, a_v

    def pad_to_multiple(rgb, crop):  # identical to normalize_lizard.py
        h, w = rgb.shape[:2]
        ph, pw = (-h) % crop, (-w) % crop
        if ph == 0 and pw == 0:
            return rgb
        mode = "reflect" if ph < h and pw < w else "edge"
        return np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode=mode)

    image_paths = sorted(p for p in Path(args.images_dir).iterdir()
                         if p.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"})
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No images found in {args.images_dir}")
    print(f"P2-08b: {len(image_paths)} images, crop={args.crop}, f={args.inversion_fraction}, "
          f"steps={args.inversion_steps}+{args.translation_steps}, guidance inv/rec="
          f"{args.inversion_guidance}/{args.guidance}, F3 sigma={args.sigma} beta={args.beta}", flush=True)

    crop, n_done, n_skip = args.crop, 0, 0
    for p in image_paths:
        if (out_fused / f"{p.stem}.png").exists() and (out_unfused / f"{p.stem}.png").exists():
            n_skip += 1
            continue
        src = np.asarray(Image.open(p).convert("RGB"))
        h, w = src.shape[:2]
        padded = pad_to_multiple(src, crop)
        ph, pw = padded.shape[:2]
        pred_canvas, av_canvas = np.zeros_like(padded), np.zeros_like(padded)
        for y in range(0, ph, crop):
            for x in range(0, pw, crop):
                hp, av = run_tile(padded[y:y + crop, x:x + crop], f"{p.stem}_x{x}_y{y}")
                pred_canvas[y:y + crop, x:x + crop] = hp
                av_canvas[y:y + crop, x:x + crop] = av
        h_pred, a_v = pred_canvas[:h, :w], av_canvas[:h, :w]
        fused = fuse_f3(src, h_pred, a_v, sigma=args.sigma, beta=args.beta)
        for arr, d in ((fused, out_fused), (h_pred, out_unfused)):
            if arr.shape[:2] != (h, w):
                raise RuntimeError(f"{p.name}: output shape {arr.shape[:2]} != source {(h, w)}")
            Image.fromarray(arr).save(d / f"{p.stem}.png")
        n_done += 1
        print(f"  {p.stem}: {h}x{w} -> {(ph // crop) * (pw // crop)} tiles", flush=True)

    meta = {**vars(args), "n_images": len(image_paths), "n_written": n_done, "n_skipped_existing": n_skip}
    for d in (out_fused, out_unfused):
        json.dump(meta, open(d / "run_metadata.json", "w"), indent=2)
    print(f"\nDone: {n_done} written, {n_skip} already present -> {out_fused} and {out_unfused}")


if __name__ == "__main__":
    main()
