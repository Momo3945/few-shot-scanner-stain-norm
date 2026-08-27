#!/usr/bin/env python3
"""
infer_p1_10_differential_diffusion.py -- P1-17: Differential Diffusion
(Levin & Fried 2023, arXiv:2306.00950) change-map inference on top of the
P1-10 source-conditioned colour LoRA + ControlNet checkpoint
(lora/a2h_cond_r8/best). See tickets/PHASE1-TICKETS.md P1-17 and
tickets/P1-17_differential_diffusion_change_map.md for the full spec.

INFERENCE ONLY. Does not retrain anything, does not modify
infer_colour_translation.py or infer_colour_source_ddim_inversion.py.

What this replaces: infer_colour_translation.py's img2img pathway applies one
GLOBAL denoising strength to the entire crop, forcing one uniform structure/
colour tradeoff everywhere. Differential Diffusion instead gives every pixel
its own effective strength via a per-pixel "change map": each pixel is
"unlocked" into the free-running denoising trajectory once the trajectory's
progress fraction exceeds that pixel's own threshold, and is otherwise reset
every step to a freshly re-noised version of the source at the current
timestep. This project already computes a Canny edge map per crop for
ControlNet conditioning (canny.py) -- a direct, already-available input for
that map: dilate + blur the binary edges into a smooth "structural density"
field, then use it to protect edge/structurally-dense regions (low effective
strength) while letting flat regions change freely (high effective strength,
matching P1-10's own best global strengths).

IMPORTANT -- change-map sign convention (verified, not assumed): this
script's own `change_strength` field follows this project's existing
`--strength` convention throughout (0 = fully preserve source, 1 = fully
regenerate -- same meaning as infer_colour_translation.py's `--strength`).
The underlying Differential Diffusion algorithm's own `map` argument uses the
OPPOSITE convention -- HIGH map value = pixel stays anchored/protected
(LESS change), LOW map value = pixel is free to regenerate (MORE change).
This was confirmed directly against the official reference implementation
(https://github.com/exx8/differential-diffusion, SD2/diff_pipe.py) two ways:
(1) tracing `masks = map > thresholds` through the per-step loop shows a
map=1.0 pixel satisfies `map > threshold` for every threshold in [0, 1), so
it is reset to the noised source EVERY step and never joins the free-running
trajectory; a map=0.0 pixel is never reset past step 0, so it runs the full
free trajectory. (2) the project's own published teaser example
(assets/teaser.png, mosque photo) shows the map is WHITE exactly over the
mosque's minarets/dome -- which stay structurally intact in the output --
and BLACK over the sky/water, which is fully regenerated into a fantastical
scene. So `pipeline_map = 1.0 - change_strength` is computed once per crop
(build_pipeline_map()) immediately before it is handed to the per-step loop
-- get this backwards and the experiment silently inverts (protects flat
colour regions, freely regenerates edges), which would actively work against
the ticket's goal.

Scope, per the ticket: reuses lora/a2h_cond_r8/best exactly as-is (no
retraining, no new adapters). Only the primary variant (G1: differential map
composed with a standard VAE-encode + add-noise init, i.e. the same
initialization infer_colour_translation.py's img2img pathway uses) is
implemented here. Combining the change map with P1-11's DDIM-inversion latent
was scoped as a stretch goal, not a requirement, and is NOT implemented in
this first version.

Because diffusers has no packaged pipeline support for a per-pixel change
map on a ControlNet img2img pipeline, this script drives `pipe.vae` /
`pipe.unet` / `pipe.controlnet` / `pipe.text_encoder` directly in a manual
step loop -- the same pattern infer_colour_source_ddim_inversion.py (P1-11)
already established for this project, reusing its predict_noise()/
encode_latent()/decode_latent()/seed_for() helpers verbatim (duplicated, not
imported, per that script's own stated precedent: each new ticket gets its
own script with zero runtime dependency on a prior script's internals).

`--strength` is not accepted (matching P1-11's precedent of explicitly
rejecting it rather than silently reinterpreting it) -- the per-pixel change
map supersedes a single global strength entirely. Use --c-min/--c-max
instead.

Change-map parameters (--radius/--sigma/--c-min/--c-max) must be selected on
P1-10's internal training-domain validation split (--pairs-dir) ONLY, never
on the held-out slides -- see the ticket's "Parameter Selection" section.
Freeze them before running the held-out evaluation (--root/--heldout).

Manifest schema matches infer_colour_translation.py's strength-shaped output
plus a seed column (the config tag -- e.g. "r4_s2_cmin0.00_cmax0.70" -- is
written into the "strength" column; score_outputs.py groups by that column
as an opaque string key, so this needs no scorer changes), so the existing,
unmodified score_outputs.py scores this directly.

Usage
-----
    # Internal-validation parameter search (small grid, per the ticket):
    python infer_p1_10_differential_diffusion.py \
        --lora <ckpt> --controlnet <ckpt>/controlnet --pairs-dir <pairs>/train \
        --out eval/p1_17_diffdiff/param_search/r4_s2_cmin0.00_cmax0.70 \
        --radius 4 --sigma 2 --c-min 0.0 --c-max 0.70 --seeds 0

    # Full held-out run at the frozen, selected config:
    python infer_p1_10_differential_diffusion.py \
        --lora <ckpt> --controlnet <ckpt>/controlnet \
        --root <mitos_heldout> --heldout pairs/heldout_frames.csv \
        --out eval/p1_17_diffdiff/held_out \
        --radius 4 --sigma 2 --c-min 0.0 --c-max 0.70 --seeds 0 1 2

Dependencies: torch, diffusers, numpy, opencv-python-headless, Pillow.
Requires canny.py and registration.py (or train_colour_translation_lora.py
for --pairs-dir) on the path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="P1-17: Differential Diffusion change-map inference for the P1-10 checkpoint.")
    ap.add_argument("--lora", required=True,
                    help="Path to the trained P1-10 LoRA checkpoint dir (e.g. lora/a2h_cond_r8/best).")
    ap.add_argument("--controlnet", required=True,
                    help="Path to the trained P1-10 ControlNet dir (e.g. lora/a2h_cond_r8/best/controlnet).")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--root", default=None, help="Dataset root (heldout paths are relative to this). "
                    "Mutually exclusive with --pairs-dir.")
    ap.add_argument("--heldout", default=None, help="heldout_frames.csv. Mutually exclusive with --pairs-dir.")
    ap.add_argument("--pairs-dir", default=None,
                    help="Run against training pairs (e.g. pairs/train) instead of held-out frames -- "
                         "use this for the ticket's mandatory internal-validation parameter search. "
                         "Mutually exclusive with --root/--heldout.")
    ap.add_argument("--overfit-n", type=int, default=0,
                    help="With --pairs-dir: use only the first N pairs (0 = all).")
    ap.add_argument("--out", required=True, help="Output dir for crops + manifest + run_metadata.json.")
    ap.add_argument("--direction", choices=["A2H", "H2A"], default="A2H")
    ap.add_argument("--source-mode", choices=["correct", "zero", "shuffled"], default="correct",
                    help="ControlNet source-conditioning ablation, same semantics as "
                         "infer_colour_translation.py. 'correct' for the primary experiment.")
    ap.add_argument("--radius", type=int, default=4,
                    help="Dilation radius (px, source-crop resolution) applied to the binary Canny edge "
                         "mask before blurring, so entire structurally-dense regions (not just 1px edge "
                         "lines) are protected.")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="Gaussian blur sigma (px, source-crop resolution) applied after dilation, "
                         "producing a smooth structural-density field in [0, 1].")
    ap.add_argument("--c-min", type=float, default=0.0,
                    help="Effective change-strength at fully edge/structure-dense pixels (0.0 = frozen "
                         "entirely; same units as infer_colour_translation.py's --strength).")
    ap.add_argument("--c-max", type=float, default=0.70,
                    help="Effective change-strength at fully flat pixels (matches P1-10's own best "
                         "global strengths, e.g. 0.50 or 0.70).")
    ap.add_argument("--steps", type=int, default=50,
                    help="Total DDIM steps. The per-pixel map supersedes --strength entirely -- every "
                         "pixel visits the full N-step schedule, just anchored to the source for a "
                         "pixel-specific leading fraction of it.")
    ap.add_argument("--guidance", type=float, default=2.0,
                    help="Classifier-free guidance scale. Matches infer_colour_translation.py's default.")
    ap.add_argument("--controlnet-scale", type=float, default=1.0)
    ap.add_argument("--strength", type=float, default=None,
                    help="NOT used by differential diffusion -- passing this raises an error. Use "
                         "--c-min/--c-max instead.")
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help="Max held-out frames (0 = all).")
    ap.add_argument("--max-crops-per-frame", type=int, default=4, help="0 = all tissue crops.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--save-change-map", action="store_true",
                    help="Also save the per-crop change-strength field (visualised as a grayscale PNG) "
                         "next to each output, for qualitative inspection of the halo/seam check.")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.strength is not None:
        raise SystemExit(
            "ERROR: --strength is not used by differential diffusion.\n"
            "Use --c-min/--c-max instead.")
    if not (0.0 <= args.c_min <= args.c_max <= 1.0):
        raise SystemExit(f"Require 0 <= --c-min <= --c-max <= 1, got c_min={args.c_min} c_max={args.c_max}.")

    using_pairs_dir = bool(args.pairs_dir)
    using_heldout = bool(args.root or args.heldout)
    if using_pairs_dir and using_heldout:
        raise SystemExit("--pairs-dir is mutually exclusive with --root/--heldout.")
    if not using_pairs_dir and not (args.root and args.heldout):
        raise SystemExit("Either --pairs-dir, or both --root and --heldout, must be given.")

    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from canny import extract_canny_control_image
    if using_pairs_dir:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
        from train_colour_translation_lora import build_pairs
    else:
        from registration import read_rgb, register_h_to_a

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("No CUDA device -- inference must run on a GPU node.")
    dtype = torch.float16

    out_dir = Path(args.out)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)
    if args.save_change_map:
        (out_dir / "change_maps").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Crop loading -- identical convention to infer_colour_translation.py /
    # infer_colour_source_ddim_inversion.py. Duplicated rather than imported
    # (P1-11's own stated precedent: zero runtime dependency on a prior
    # script's internals ever changing underneath it).
    # ------------------------------------------------------------------
    def tissue_fraction(rgb):
        a = rgb.astype(np.int16); mx = a.max(2); mn = a.min(2)
        return float(((mx < 235) & ((mx - mn) > 12)).mean())

    def grid_offsets(length, crop):
        if length <= crop:
            return [0]
        offs = list(range(0, length - crop + 1, crop))
        if offs[-1] != length - crop:
            offs.append(length - crop)
        return sorted(set(offs))

    crops = []  # each: {slide, frame, x, y, tag_id, src, ref, aperio_path}
    if using_pairs_dir:
        pairs = build_pairs(args.pairs_dir)
        if args.overfit_n:
            pairs = pairs[: args.overfit_n]
        if args.limit:
            pairs = pairs[: args.limit]
        src_key, ref_key = ("aperio_path", "hamamatsu_path") if args.direction == "A2H" \
            else ("hamamatsu_path", "aperio_path")
        for p in pairs:
            src_c = np.asarray(Image.open(p[src_key]).convert("RGB"))
            ref_c = np.asarray(Image.open(p[ref_key]).convert("RGB"))
            crops.append({"slide": p["slide"], "frame": p["frame_id"], "x": 0, "y": 0,
                         "tag_id": p["pair_id"],
                         "src": src_c, "ref": ref_c, "aperio_path": p["aperio_path"]})
        print(f"  {args.pairs_dir}: {len(crops)} training pairs (overfit_n={args.overfit_n or 'all'})")
    else:
        with open(args.heldout, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if args.limit:
            rows = rows[: args.limit]
        for r in rows:
            r["aperio_path"] = r["aperio_path"].replace("\\", "/")
            r["hamamatsu_path"] = r["hamamatsu_path"].replace("\\", "/")

        for r in rows:
            a_rgb = read_rgb(Path(args.root) / r["aperio_path"])
            h_rgb = read_rgb(Path(args.root) / r["hamamatsu_path"])
            reg = register_h_to_a(a_rgb, h_rgb, ecc_min=args.ecc_min)
            h_reg = reg.h_registered_rgb
            H, W = a_rgb.shape[:2]
            src_frame, ref_frame = (a_rgb, h_reg) if args.direction == "A2H" else (h_reg, a_rgb)

            crops_done = 0
            for y in grid_offsets(H, args.crop):
                for x in grid_offsets(W, args.crop):
                    if args.max_crops_per_frame and crops_done >= args.max_crops_per_frame:
                        break
                    src_c = src_frame[y:y + args.crop, x:x + args.crop]
                    ref_c = ref_frame[y:y + args.crop, x:x + args.crop]
                    if (ref_c.max(2) < 6).mean() > 0.10:
                        continue
                    if tissue_fraction(src_c) < args.tissue_thresh:
                        continue
                    crops.append({"slide": r["aperio_slide"], "frame": r["frame_id"],
                                 "x": x, "y": y,
                                 "tag_id": f"{r['aperio_slide']}_{r['frame_id']}_x{x}_y{y}",
                                 "src": src_c, "ref": ref_c,
                                 "aperio_path": r["aperio_path"]})
                    crops_done += 1
            print(f"  {r['aperio_slide']}_{r['frame_id']}: {crops_done} crops")
    if not crops:
        raise SystemExit("No tissue crops selected -- check --tissue-thresh / heldout / pairs-dir inventory.")

    rng = np.random.default_rng(0)
    shuffled_idx = rng.permutation(len(crops))
    for i in range(len(crops)):
        if shuffled_idx[i] == i:
            j = (i + 1) % len(crops)
            shuffled_idx[i], shuffled_idx[j] = shuffled_idx[j], shuffled_idx[i]

    def control_source_for(i):
        if args.source_mode == "correct":
            return crops[i]["src"]
        if args.source_mode == "zero":
            return None
        return crops[shuffled_idx[i]]["src"]

    def build_control_tensor(rgb):
        canny = extract_canny_control_image(rgb)
        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        canny_t = torch.from_numpy(canny.astype(np.float32) / 255.0).permute(2, 0, 1)
        return torch.cat([rgb_t, canny_t], dim=0).unsqueeze(0)  # 1x6xHxW, [0,1]

    def control_tensor_or_zero(rgb):
        t = (torch.zeros(1, 6, args.crop, args.crop, dtype=dtype)
             if rgb is None else build_control_tensor(rgb).to(dtype=dtype))
        return t.to(device)

    def seed_for(pair_key: str, seed: int) -> int:
        h = hashlib.sha256(f"{pair_key}|{seed}".encode()).hexdigest()
        return int(h[:8], 16)

    # ------------------------------------------------------------------
    # Change-map construction -- source-only, no target leakage (ticket
    # guardrail). Raw binary Canny edges are 1px lines; dilate then blur so
    # entire structurally-dense regions (nucleus interiors, chromatin
    # texture near an edge), not just the edge pixel itself, are protected.
    # ------------------------------------------------------------------
    def build_change_strength_map(source_rgb):
        canny = extract_canny_control_image(source_rgb)[:, :, 0]  # binary {0,255}, HxW
        edge_mask = (canny > 0).astype(np.uint8)
        if args.radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.radius + 1, 2 * args.radius + 1))
            edge_mask = cv2.dilate(edge_mask, kernel)
        density = edge_mask.astype(np.float32)
        if args.sigma > 0:
            density = cv2.GaussianBlur(density, (0, 0), sigmaX=args.sigma)
        density = np.clip(density, 0.0, 1.0)
        # density=1 (edge/structure-dense) -> c_min (protected); density=0 (flat) -> c_max (free to change).
        change_strength = args.c_max - (args.c_max - args.c_min) * density
        return change_strength.astype(np.float32)  # HxW, source-crop pixel resolution, in [c_min, c_max]

    # ------------------------------------------------------------------
    # Pipeline + scheduler -- reuses infer_colour_source_ddim_inversion.py's
    # (P1-11) exact predict_noise()/encode_latent()/decode_latent() pattern.
    # ------------------------------------------------------------------
    from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetImg2ImgPipeline

    print(f"Loading P1-10 pipeline: LoRA={args.lora}  ControlNet={args.controlnet}  "
          f"radius={args.radius} sigma={args.sigma} c_min={args.c_min} c_max={args.c_max} ...")
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.model, controlnet=controlnet, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False)
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = ddim_scheduler
    pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    vae, unet, text_encoder, tokenizer = pipe.vae, pipe.unet, pipe.text_encoder, pipe.tokenizer
    scaling = vae.config.scaling_factor
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)  # 8 for stock SD1.5

    with torch.no_grad():
        def encode_prompt(text):
            tok = tokenizer(text, padding="max_length", truncation=True,
                            max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
            return text_encoder(tok.input_ids)[0].to(dtype)
        cond_embeds = encode_prompt(args.prompt)
        uncond_embeds = encode_prompt("")

    def predict_noise(latent, t, control, guidance_scale):
        if guidance_scale is None or guidance_scale <= 1.0:
            down_res, mid_res = controlnet(
                latent, t, encoder_hidden_states=cond_embeds, controlnet_cond=control,
                conditioning_scale=args.controlnet_scale, return_dict=False)
            return unet(latent, t, encoder_hidden_states=cond_embeds,
                       down_block_additional_residuals=down_res,
                       mid_block_additional_residual=mid_res).sample
        latent_in = torch.cat([latent, latent], dim=0)
        embeds_in = torch.cat([uncond_embeds, cond_embeds], dim=0)
        control_in = torch.cat([control, control], dim=0)
        down_res, mid_res = controlnet(
            latent_in, t, encoder_hidden_states=embeds_in, controlnet_cond=control_in,
            conditioning_scale=args.controlnet_scale, return_dict=False)
        noise_pred = unet(latent_in, t, encoder_hidden_states=embeds_in,
                          down_block_additional_residuals=down_res,
                          mid_block_additional_residual=mid_res).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)
        return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    def encode_latent(rgb):
        arr = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        arr = arr.unsqueeze(0).to(device, dtype=dtype)
        return vae.encode(arr).latent_dist.mode() * scaling

    def decode_latent(latent):
        decoded = vae.decode(latent / scaling).sample
        if not torch.isfinite(decoded).all():
            raise RuntimeError("NaN/Inf in decoded output -- aborting.")
        return ((decoded[0].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    def build_pipeline_map(change_strength_px, latent_hw):
        # Resize the pixel-resolution change-strength field down to latent
        # resolution with area averaging (appropriate for downsampling,
        # avoids aliasing) -- then invert to the reference algorithm's own
        # convention (see module docstring: high map value = protected).
        lh, lw = latent_hw
        resized = cv2.resize(change_strength_px, (lw, lh), interpolation=cv2.INTER_AREA)
        change_strength_latent = torch.from_numpy(resized).to(device=device, dtype=dtype)
        pipeline_map = 1.0 - change_strength_latent
        return pipeline_map.unsqueeze(0).unsqueeze(0)  # [1,1,Hl,Wl], broadcasts over the channel dim

    @torch.no_grad()
    def differential_denoise(source_rgb, change_strength_px, control, guidance_scale, n_steps, seed):
        generator = torch.Generator(device=device).manual_seed(seed)
        latent0 = encode_latent(source_rgb)
        noise = torch.randn(latent0.shape, generator=generator, device=device, dtype=latent0.dtype)
        ddim_scheduler.set_timesteps(n_steps, device=device)
        timesteps = ddim_scheduler.timesteps  # full N steps -- the map supersedes --strength entirely
        n = len(timesteps)
        pipeline_map = build_pipeline_map(change_strength_px, latent0.shape[-2:])

        latent = None
        for i, t in enumerate(timesteps):
            noised_original_i = ddim_scheduler.add_noise(latent0, noise, t.unsqueeze(0))
            if i == 0:
                latent = noised_original_i
            else:
                threshold_i = i / n
                mask_i = (pipeline_map > threshold_i).to(latent.dtype)
                latent = noised_original_i * mask_i + latent * (1 - mask_i)
            noise_pred = predict_noise(latent, t, control, guidance_scale)
            latent = ddim_scheduler.step(noise_pred, t, latent, eta=0.0, return_dict=True).prev_sample
        if not torch.isfinite(latent).all():
            raise RuntimeError("NaN/Inf in denoised latent -- aborting.")
        return decode_latent(latent)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    config_tag = f"r{args.radius}_s{args.sigma:g}_cmin{args.c_min:.2f}_cmax{args.c_max:.2f}"
    man_path = out_dir / "eval_manifest.csv"
    man = open(man_path, "w", newline="")
    mw = csv.writer(man)
    mw.writerow(["strength", "seed", "slide", "frame", "x", "y",
                "output_path", "reference_path", "aperio_path"])

    n_out = 0
    for i, c in enumerate(crops):
        tag = c["tag_id"]
        ref_path = out_dir / "reference" / f"{tag}.png"
        if not ref_path.exists():
            Image.fromarray(c["ref"]).save(ref_path)

        change_strength_px = build_change_strength_map(c["src"])
        if args.save_change_map:
            vis = (255 * (change_strength_px - args.c_min) / max(args.c_max - args.c_min, 1e-6)).astype("uint8")
            Image.fromarray(vis).save(out_dir / "change_maps" / f"{tag}.png")

        control = control_tensor_or_zero(control_source_for(i)).to(dtype=dtype)

        for s in args.seeds:
            seed_val = seed_for(tag, s)
            out = differential_denoise(c["src"], change_strength_px, control, args.guidance, args.steps, seed_val)
            out_path = out_dir / "outputs" / f"{tag}_{config_tag}_seed{s}.png"
            Image.fromarray(out).save(out_path)
            mw.writerow([config_tag, s, c["slide"], c["frame"], c["x"], c["y"],
                        os.path.relpath(out_path, out_dir),
                        os.path.relpath(ref_path, out_dir),
                        c["aperio_path"]])
            n_out += 1

    man.close()

    def git_commit():
        try:
            repo_root = Path(__file__).resolve().parent.parent.parent
            return subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"

    metadata = {
        "script": "infer_p1_10_differential_diffusion.py",
        "checkpoint_lora": args.lora,
        "checkpoint_controlnet": args.controlnet,
        "direction": args.direction,
        "source_mode": args.source_mode,
        "radius": args.radius,
        "sigma": args.sigma,
        "c_min": args.c_min,
        "c_max": args.c_max,
        "steps": args.steps,
        "guidance": args.guidance,
        "controlnet_conditioning_scale": args.controlnet_scale,
        "config_tag": config_tag,
        "seeds": args.seeds,
        "dtype": str(dtype),
        "model": args.model,
        "prompt": args.prompt,
        "vae_scale_factor": vae_scale_factor,
        "git_commit": git_commit(),
    }
    with open(out_dir / "run_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nWrote {n_out} output crops across {len(crops)} crops x {len(args.seeds)} seed(s), "
         f"config={config_tag}.")
    print(f"Manifest: {man_path}")
    print(f"Metadata: {out_dir / 'run_metadata.json'}")
    print("Next: score with score_outputs.py (unmodified) -- schema matches infer_colour_translation.py exactly.")


if __name__ == "__main__":
    main()
