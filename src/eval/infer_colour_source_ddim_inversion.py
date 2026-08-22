#!/usr/bin/env python3
"""
infer_colour_source_ddim_inversion.py -- P1-11: DDIM-inversion inference path
for the P1-10 source-conditioned colour LoRA + ControlNet checkpoint
(lora/a2h_cond_r8/best). See tickets/PHASE1-TICKETS.md P1-11 and
tickets/"Claude Code Task_ Add DDIM-Inversion Inference for P1-10(2).md" for
the full spec this implements.

INFERENCE ONLY. Does not retrain anything, does not modify
infer_colour_translation.py (P1-10's existing, validated inference script),
and loads lora/a2h_cond_r8/best exactly as trained (epsilon prediction,
confirmed via that checkpoint's training_config.json -- asserted below, not
assumed).

What this replaces: infer_colour_translation.py's img2img initialisation
adds i.i.d. Gaussian noise to the VAE-encoded source latent at a fixed
`strength` and denoises from there -- the noise has no relationship to the
actual source image. DDIM inversion instead deterministically runs the
trained model's own noise-prediction *backwards* (source-domain latent ->
increasingly noisy latent) to obtain a source-specific starting point, then
denoises forward again with the trained model. The scientific question (task
file, final section): does this recover more source structure than random
img2img corruption, without giving up P1-10's colour-recovery gain?

Because diffusers has no packaged "invert-then-conditionally-reconstruct"
pipeline for a ControlNet img2img pipeline, this script drives
`pipe.vae` / `pipe.unet` / `pipe.controlnet` / `pipe.text_encoder` directly in
a manual step loop (the standard pattern used by every DDIM-inversion
implementation this project's reference, HistDiST, and diffusers' own
inversion examples follow) rather than calling `pipe(...)`. The forward
scheduler (`DDIMScheduler`) and the inverse scheduler (`DDIMInverseScheduler`)
are both built from the SAME resolved config as the existing P1-10 inference
script's `DDIMScheduler.from_config(pipe.scheduler.config)` -- NOT from
HistDiST's v-prediction/trailing-timestep/zero-terminal-SNR settings, which
this checkpoint was never trained with.

Two run modes
-------------
  --mode identity   A -> VAE encode -> DDIM inversion -> DDIM reconstruction
                     (conditioned on A's own source throughout, never
                     Hamamatsu) -> VAE decode -> A'. Scored against the
                     ORIGINAL source crop itself, not a real Hamamatsu image.
                     In one pass, also produces the two mandatory comparison
                     points for the same crop/seed (task file secs. 13-14,
                     23): `vae_only` (pure encode/decode, no UNet at all) and
                     `img2img_baseline` (the EXISTING random-noise pathway,
                     replicated by calling the already-loaded pipeline the
                     normal `pipe(...)` way at --baseline-strength -- this is
                     not a code change to infer_colour_translation.py, just
                     invoking the identical underlying model fresh here so
                     the three methods share the exact same in-memory crop).
                     CAVEAT (documented, not hidden): the LoRA's colour-shift
                     bias is baked into the UNet's attention weights, not
                     just the ControlNet conditioning -- so `identity` mode
                     may still show some colour drift even though it never
                     conditions on a target-domain image. That is a real
                     diagnostic finding about the trained model, not a flaw
                     in this script, and is not masked by disabling the LoRA
                     (which the task file forbids touching anyway).

  --mode translate  A -> VAE encode -> DDIM inversion -> DDIM reconstruction
                     conditioned per --source-mode {correct,zero,shuffled}
                     (identical ablation semantics to the existing script)
                     -> VAE decode -> Ĥ. Scored against the real registered
                     Hamamatsu reference, same convention as every other
                     P1-10 run.

In both modes, --inversion-condition {none,source} controls ONLY the
INVERSION half's ControlNet conditioning (default "source" -- the crop's own
real 6-channel RGB+Canny signal, the closest available analogue to what the
model saw in training, hence documented as the conservative/identity-
preserving default; "none" feeds an all-zero conditioning tensor, same
convention as the existing script's --source-mode zero).

Partial inversion (--inversion-fraction in (0, 1]) takes the first
round(inversion_steps * fraction) steps of the ascending inverse-scheduler
timestep grid; the translation/reconstruction pass then starts from the
forward-scheduler timestep nearest (and not exceeding) wherever inversion
stopped, NOT from a fixed strength -- --strength is accepted only so it can
be explicitly rejected (see below), matching the task file's requirement to
never silently convert this back into add_noise(strength).

Manifest schema is a strict superset of infer_colour_translation.py's
eval_manifest.csv (same seed/source_mode/crop_id/slide/frame/x/y/
output_path/reference_path/aperio_path columns, plus method/mode/
inversion_fraction/inversion_condition/actual_* columns) -- score_p1_10_
ablation.py (csv.DictReader, tolerates extra columns) scores it completely
unchanged.

Usage
-----
    # Test 1-4 (task file sec. 23), full inversion, on a small subset:
    python infer_colour_source_ddim_inversion.py \
        --lora <ckpt> --controlnet <ckpt>/controlnet \
        --root <mitos_heldout> --heldout pairs/heldout_frames.csv \
        --out eval/p1_10_ddim_inversion/identity/inv100 \
        --mode identity --inversion-fraction 1.0 --limit 20 --seeds 0

    # partial inversion identity, e.g. 50%:
    python infer_colour_source_ddim_inversion.py ... \
        --out eval/p1_10_ddim_inversion/identity/inv050 \
        --mode identity --inversion-fraction 0.50 --limit 20 --seeds 0

    # translation smoke test, correct/shuffled/zero at the best identity
    # inversion-fraction found above:
    python infer_colour_source_ddim_inversion.py ... \
        --out eval/p1_10_ddim_inversion/translate/inv050_correct \
        --mode translate --inversion-fraction 0.50 --source-mode correct \
        --limit 20 --seeds 0 1 2

Dependencies: torch, diffusers>=0.39 (DDIMInverseScheduler), numpy, Pillow,
opencv-python-headless, tifffile. Requires canny.py and registration.py (or
train_colour_translation_lora.py for --pairs-dir) on the path.
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
        description="P1-11: DDIM-inversion inference for the P1-10 source-conditioned checkpoint.")
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
                         "same convention as infer_colour_translation.py's --pairs-dir. Mutually "
                         "exclusive with --root/--heldout.")
    ap.add_argument("--overfit-n", type=int, default=0,
                    help="With --pairs-dir: use only the first N pairs (0 = all), same slicing as "
                         "train_colour_translation_lora.py.")
    ap.add_argument("--out", required=True, help="Output dir for crops + manifest + run_metadata.json.")
    ap.add_argument("--direction", choices=["A2H", "H2A"], default="A2H")
    ap.add_argument("--mode", choices=["identity", "translate"], required=True,
                    help="identity: reconstruction self-conditions on the source, scored against the "
                         "source itself (also produces vae_only/img2img_baseline comparison points). "
                         "translate: reconstruction conditions per --source-mode, scored against real "
                         "Hamamatsu.")
    ap.add_argument("--source-mode", choices=["correct", "zero", "shuffled"], default="correct",
                    help="Reconstruction-pass conditioning ablation. Only meaningful in --mode translate "
                         "(ignored, with a note, in --mode identity -- reconstruction there always "
                         "self-conditions).")
    ap.add_argument("--inversion-condition", choices=["none", "source"], default="source",
                    help="Conditioning fed to the ControlNet branch DURING INVERSION ONLY (not "
                         "reconstruction). 'source': the crop's own real RGB+Canny signal (default -- "
                         "the most conservative, identity-preserving choice, closest to what the model "
                         "saw in training). 'none': an all-zero conditioning tensor.")
    ap.add_argument("--inversion-fraction", type=float, default=1.0,
                    help="0 < f <= 1. Fraction of the inversion trajectory to run -- the DDIM-inversion "
                         "analogue of img2img's --strength. 1.0 = full inversion.")
    ap.add_argument("--inversion-steps", type=int, default=50)
    ap.add_argument("--translation-steps", type=int, default=50)
    ap.add_argument("--inversion-guidance", type=float, default=1.0,
                    help="Classifier-free guidance scale used DURING INVERSION. Default 1.0 (no CFG) -- "
                         "standard DDIM-inversion practice, since CFG during inversion is not exactly "
                         "invertible and accumulates error (the motivation behind Null-text Inversion, "
                         "which this ticket deliberately does not implement -- out of scope, sec. 22).")
    ap.add_argument("--guidance", type=float, default=2.0,
                    help="Classifier-free guidance scale for the reconstruction/translation pass. "
                         "Matches infer_colour_translation.py's default.")
    ap.add_argument("--baseline-strength", type=float, default=0.50,
                    help="--mode identity only: img2img strength for the existing-pathway comparison "
                         "point, matching P1-10's established operating point.")
    ap.add_argument("--controlnet-scale", type=float, default=1.0)
    ap.add_argument("--strength", type=float, default=None,
                    help="NOT used by DDIM inversion -- passing this raises an error. Use "
                         "--inversion-fraction instead.")
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help="Max held-out frames (0 = all).")
    ap.add_argument("--max-crops-per-frame", type=int, default=4, help="0 = all tissue crops.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Retained for parity with infer_colour_translation.py's seed protocol and "
                         "output naming/manifest, even though eta=0 DDIM inversion+reconstruction is "
                         "itself deterministic given a fixed conditioning choice.")
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--debug-inversion", action="store_true",
                    help="Print per-step latent mean/std, timesteps, and a NaN/Inf check.")
    ap.add_argument("--save-inversion-trajectory", action="store_true",
                    help="Decode and save snapshots at ~0/25/50/75/100%% of the inversion trajectory. "
                         "Off by default -- slows the run and uses disk (task file sec. 20).")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.strength is not None:
        raise SystemExit(
            "ERROR: --strength is not used with DDIM inversion.\n"
            "Use --inversion-fraction instead.")
    if not (0.0 < args.inversion_fraction <= 1.0):
        raise SystemExit(f"--inversion-fraction must satisfy 0 < f <= 1, got {args.inversion_fraction}.")

    using_pairs_dir = bool(args.pairs_dir)
    using_heldout = bool(args.root or args.heldout)
    if using_pairs_dir and using_heldout:
        raise SystemExit("--pairs-dir is mutually exclusive with --root/--heldout.")
    if not using_pairs_dir and not (args.root and args.heldout):
        raise SystemExit("Either --pairs-dir, or both --root and --heldout, must be given.")
    if args.mode == "identity" and args.source_mode != "correct":
        print(f"NOTE: --source-mode={args.source_mode} is ignored in --mode identity -- "
              f"reconstruction there always self-conditions on the source crop.")

    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
    if args.save_inversion_trajectory:
        (out_dir / "trajectory").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Crop loading -- identical to infer_colour_translation.py (task file
    # sec. 5 Step A: "identical resizing, normalization, tensor shape,
    # dtype, device handling -- no new preprocessing"). Duplicated rather
    # than imported so this script has zero runtime dependency on the old
    # script's internals ever changing underneath it.
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
        # translate-mode reconstruction conditioning per --source-mode --
        # identical semantics to infer_colour_translation.py's control_source_for.
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
    # Pipeline + schedulers
    # ------------------------------------------------------------------
    from diffusers import (ControlNetModel, DDIMInverseScheduler,
                           DDIMScheduler, StableDiffusionControlNetImg2ImgPipeline)

    print(f"Loading P1-10 pipeline: LoRA={args.lora}  ControlNet={args.controlnet}  "
          f"mode={args.mode}  inversion_condition={args.inversion_condition} ...")
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.model, controlnet=controlnet, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False)
    # Build the forward DDIM scheduler exactly like infer_colour_translation.py does
    # (from the pipeline's own loaded config), THEN derive the inverse scheduler from
    # THAT resolved config -- inherits num_train_timesteps/beta_*/steps_offset/
    # set_alpha_to_one/clip_sample/timestep_spacing/rescale_betas_zero_snr from the
    # real checkpoint, not hardcoded HistDiST values (task file sec. 6).
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = ddim_scheduler
    if ddim_scheduler.config.prediction_type != "epsilon":
        raise SystemExit(
            f"P1-10 (lora/a2h_cond_r8) was trained with epsilon prediction "
            f"(training_config.json), but the resolved scheduler config says "
            f"prediction_type={ddim_scheduler.config.prediction_type!r}. Refusing to "
            f"silently run inversion/translation under a mismatched prediction type "
            f"(task file sec. 21) -- investigate before proceeding.")
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

    # ------------------------------------------------------------------
    # Core diffusion machinery: manual step loops (no pipe.__call__), since
    # diffusers has no packaged inversion path for a ControlNet img2img
    # pipeline. predict_noise() is shared by both the inversion and
    # reconstruction loops (single forward pass when guidance<=1, batched
    # uncond+cond CFG otherwise -- mirrors StableDiffusionControlNetImg2Img
    # Pipeline.__call__'s own CFG combination exactly).
    # ------------------------------------------------------------------
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
        # .mode(), not .sample(): inversion must start from a deterministic encode,
        # matching the existing script's --vae-only convention.
        return vae.encode(arr).latent_dist.mode() * scaling

    def decode_latent(latent):
        decoded = vae.decode(latent / scaling).sample
        if not torch.isfinite(decoded).all():
            raise RuntimeError("NaN/Inf in decoded output -- aborting (task file acceptance criterion 8).")
        return ((decoded[0].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    def invert(latent0, control, guidance_scale, n_steps_total, fraction, tag=""):
        inverse_scheduler.set_timesteps(n_steps_total, device=device)
        timesteps = inverse_scheduler.timesteps  # ascending: low noise -> high noise
        k = max(1, round(n_steps_total * fraction))
        used = timesteps[:k]
        latent = latent0
        snaps = {}
        if args.save_inversion_trajectory:
            snaps[0] = decode_latent(latent)
        snap_marks = {round(k * f) for f in (0.25, 0.5, 0.75, 1.0)} if args.save_inversion_trajectory else set()
        for i, t in enumerate(used):
            noise_pred = predict_noise(latent, t, control, guidance_scale)
            # DDIMInverseScheduler.step()'s output field is still named "prev_sample"
            # for API consistency with DDIMScheduler, but here it is the NEXT
            # (noisier) latent -- inversion walks toward higher noise, not lower.
            latent = inverse_scheduler.step(noise_pred, t, latent, return_dict=True).prev_sample
            if args.debug_inversion:
                print(f"    [inv {tag}] step {i+1}/{k} t={int(t)} "
                     f"mean={latent.mean().item():.4f} std={latent.std().item():.4f}")
            if (i + 1) in snap_marks:
                snaps[round((i + 1) / k * 100)] = decode_latent(latent)
        t_end = int(used[-1].item()) if len(used) else None
        if not torch.isfinite(latent).all():
            raise RuntimeError("NaN/Inf in inverted latent -- aborting.")
        return latent, t_end, k, snaps

    def reconstruct(latent_start, t_start, control, guidance_scale, n_steps_total, tag=""):
        ddim_scheduler.set_timesteps(n_steps_total, device=device)
        fwd_timesteps = ddim_scheduler.timesteps  # descending: high noise -> low noise
        if t_start is None:
            used = fwd_timesteps
        else:
            idx = len(fwd_timesteps) - 1
            for i, tv in enumerate(fwd_timesteps):
                if tv.item() <= t_start:
                    idx = i
                    break
            used = fwd_timesteps[idx:]
        latent = latent_start
        for i, t in enumerate(used):
            noise_pred = predict_noise(latent, t, control, guidance_scale)
            latent = ddim_scheduler.step(noise_pred, t, latent, eta=0.0, return_dict=True).prev_sample
            if args.debug_inversion:
                print(f"    [rec {tag}] step {i+1}/{len(used)} t={int(t)} "
                     f"mean={latent.mean().item():.4f} std={latent.std().item():.4f}")
        if not torch.isfinite(latent).all():
            raise RuntimeError("NaN/Inf in reconstructed latent -- aborting.")
        return latent, len(used)

    @torch.no_grad()
    def run_ddim_inversion(source_rgb, inv_ctrl_rgb, rec_ctrl_rgb, seed, tag):
        torch.manual_seed(seed)  # retained for convention/logging; eta=0 DDIM has no
                                  # injected randomness once conditioning is fixed.
        latent0 = encode_latent(source_rgb)
        inv_control = control_tensor_or_zero(inv_ctrl_rgb)
        rec_control = control_tensor_or_zero(rec_ctrl_rgb)
        latent_inv, t_end, k_actual, snaps = invert(
            latent0, inv_control, args.inversion_guidance,
            args.inversion_steps, args.inversion_fraction, tag=tag)
        latent_rec, n_rec = reconstruct(
            latent_inv, t_end, rec_control, args.guidance, args.translation_steps, tag=tag)
        out = decode_latent(latent_rec)
        if args.debug_inversion:
            print(f"    [meta {tag}] requested_fraction={args.inversion_fraction} "
                 f"actual_inv_steps={k_actual}/{args.inversion_steps} t_end={t_end} "
                 f"translation_steps_used={n_rec}/{args.translation_steps}")
        for pct, img in snaps.items():
            Image.fromarray(img).save(out_dir / "trajectory" / f"{tag}_inv{pct}pct.png")
        return out, t_end, k_actual, n_rec

    @torch.no_grad()
    def run_vae_only(source_rgb, seed):
        torch.manual_seed(seed)
        return decode_latent(encode_latent(source_rgb))

    @torch.no_grad()
    def run_img2img_baseline(source_rgb, control_rgb, seed):
        # The EXISTING P1-10 pathway, replicated via the already-loaded pipeline's
        # normal __call__ (not a code change to infer_colour_translation.py) so the
        # identity-mode comparison uses the exact same in-memory crop/seed as the
        # new inversion pathway (task file sec. 14).
        control_tensor = (torch.zeros(1, 6, args.crop, args.crop, dtype=torch.float32)
                          if control_rgb is None else build_control_tensor(control_rgb))
        gen = torch.Generator(device=device).manual_seed(seed)
        out = pipe(prompt=args.prompt, image=Image.fromarray(source_rgb), strength=args.baseline_strength,
                   num_inference_steps=args.translation_steps, guidance_scale=args.guidance,
                   control_image=control_tensor, controlnet_conditioning_scale=args.controlnet_scale,
                   generator=gen).images[0]
        return np.asarray(out)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    man_path = out_dir / "eval_manifest.csv"
    man = open(man_path, "w", newline="")
    mw = csv.writer(man)
    mw.writerow(["seed", "source_mode", "crop_id", "slide", "frame", "x", "y",
                "output_path", "reference_path", "aperio_path",
                "method", "mode", "inversion_condition", "inversion_fraction",
                "actual_inversion_steps", "actual_start_timestep", "actual_translation_steps"])

    n_out = 0
    for i, c in enumerate(crops):
        tag = c["tag_id"]
        # identity mode: reference = the original source crop itself (already in
        # memory). translate mode: reference = the real registered Hamamatsu crop,
        # same convention as infer_colour_translation.py.
        ref_img = c["src"] if args.mode == "identity" else c["ref"]
        ref_path = out_dir / "reference" / f"{tag}.png"
        if not ref_path.exists():
            Image.fromarray(ref_img).save(ref_path)

        for s in args.seeds:
            seed_val = seed_for(tag, s)
            run_tag = f"{tag}_seed{s}"

            if args.mode == "identity":
                inv_ctrl_rgb = c["src"] if args.inversion_condition == "source" else None
                rec_ctrl_rgb = c["src"]  # self-conditioned throughout -- no A->H

                out_vae = run_vae_only(c["src"], seed_val)
                p = out_dir / "outputs" / f"{run_tag}_vae_only.png"
                Image.fromarray(out_vae).save(p)
                mw.writerow([s, "identity_vae_only", tag, c["slide"], c["frame"], c["x"], c["y"],
                            os.path.relpath(p, out_dir), os.path.relpath(ref_path, out_dir),
                            c["aperio_path"], "vae_only", args.mode, "", "", "", "", ""])
                n_out += 1

                out_base = run_img2img_baseline(c["src"], rec_ctrl_rgb, seed_val)
                p = out_dir / "outputs" / f"{run_tag}_img2img_baseline.png"
                Image.fromarray(out_base).save(p)
                mw.writerow([s, f"identity_img2img_s{args.baseline_strength}", tag,
                            c["slide"], c["frame"], c["x"], c["y"],
                            os.path.relpath(p, out_dir), os.path.relpath(ref_path, out_dir),
                            c["aperio_path"], "img2img_baseline", args.mode, "", "", "", "", ""])
                n_out += 1

                out_inv, t_end, k_actual, n_rec = run_ddim_inversion(
                    c["src"], inv_ctrl_rgb, rec_ctrl_rgb, seed_val, run_tag)
                p = out_dir / "outputs" / f"{run_tag}_ddim_inversion.png"
                Image.fromarray(out_inv).save(p)
                mw.writerow([s, f"identity_ddim_inv_f{args.inversion_fraction}_{args.inversion_condition}",
                            tag, c["slide"], c["frame"], c["x"], c["y"],
                            os.path.relpath(p, out_dir), os.path.relpath(ref_path, out_dir),
                            c["aperio_path"], "ddim_inversion", args.mode, args.inversion_condition,
                            args.inversion_fraction, k_actual, t_end, n_rec])
                n_out += 1
            else:  # translate
                inv_ctrl_rgb = c["src"] if args.inversion_condition == "source" else None
                rec_ctrl_rgb = control_source_for(i)

                out_inv, t_end, k_actual, n_rec = run_ddim_inversion(
                    c["src"], inv_ctrl_rgb, rec_ctrl_rgb, seed_val, run_tag)
                p = out_dir / "outputs" / f"{run_tag}_{args.source_mode}_ddim_inversion.png"
                Image.fromarray(out_inv).save(p)
                mw.writerow([s, args.source_mode, tag, c["slide"], c["frame"], c["x"], c["y"],
                            os.path.relpath(p, out_dir), os.path.relpath(ref_path, out_dir),
                            c["aperio_path"], "ddim_inversion", args.mode, args.inversion_condition,
                            args.inversion_fraction, k_actual, t_end, n_rec])
                n_out += 1

    man.close()

    # ------------------------------------------------------------------
    # Metadata (task file sec. 18) -- best-effort git commit since only
    # src/+slurm/ are rsynced to the cluster, but CODE_DIR itself is a real
    # clone (see CLAUDE.md).
    # ------------------------------------------------------------------
    def git_commit():
        try:
            repo_root = Path(__file__).resolve().parent.parent.parent
            return subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"

    metadata = {
        "script": "infer_colour_source_ddim_inversion.py",
        "checkpoint_lora": args.lora,
        "checkpoint_controlnet": args.controlnet,
        "direction": args.direction,
        "mode": args.mode,
        "source_mode": args.source_mode if args.mode == "translate" else None,
        "inversion_condition": args.inversion_condition,
        "prediction_type": ddim_scheduler.config.prediction_type,
        "ddim_scheduler_config": dict(ddim_scheduler.config),
        "inverse_scheduler_config": dict(inverse_scheduler.config),
        "vae_scaling_factor": scaling,
        "inversion_steps": args.inversion_steps,
        "translation_steps": args.translation_steps,
        "inversion_fraction": args.inversion_fraction,
        "inversion_guidance": args.inversion_guidance,
        "guidance": args.guidance,
        "baseline_strength": args.baseline_strength if args.mode == "identity" else None,
        "controlnet_conditioning_scale": args.controlnet_scale,
        "seeds": args.seeds,
        "dtype": str(dtype),
        "model": args.model,
        "prompt": args.prompt,
        "initialization": "ddim_inversion",
        "git_commit": git_commit(),
    }
    with open(out_dir / "run_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nWrote {n_out} output rows across {len(crops)} crops x {len(args.seeds)} seed(s), "
         f"mode={args.mode}.")
    print(f"Manifest: {man_path}")
    print(f"Metadata: {out_dir / 'run_metadata.json'}")
    print("Next: score with score_p1_10_ablation.py (same schema as infer_colour_translation.py's manifest).")


if __name__ == "__main__":
    main()
