#!/usr/bin/env python3
"""
infer_colour_translation_sdxl.py -- P3-06: inference for the SDXL transfer of
P1-10's source-conditioned colour LoRA + fresh 6-channel ControlNet, trained
by train_colour_translation_lora_sdxl.py. See tickets/PHASE3-TICKETS.md P3-06
and tickets/PHASE1-TICKETS.md P1-10 for the full design rationale -- not
repeated here beyond what differs for SDXL.

Merges two already-validated scripts in this project:

  infer_colour_translation.py (P1-10, SD1.5) supplies the mandatory
  diagnostic controls and their exact semantics:
    --source-mode {correct,zero,shuffled} -- the source-conditioning
        ablation (correct must differ materially from zero/shuffled and win
        on structural/content metrics, or the ControlNet branch is being
        ignored).
    --vae-only -- the VAE-only floor control (skips UNet/ControlNet/LoRA
        entirely, encode-then-decode through the frozen VAE alone).
    --seeds (multiple, hash-derived per-crop determinism, NOT reset to one
        global seed per crop) -- report mean +/- spread, not a point estimate.
  These CLI semantics and the 6-channel control-tensor construction are
  copied verbatim -- they have nothing to do with the backbone.

  infer_colour_lora_sdxl.py (P3-03) supplies the SDXL-specific pipeline
  construction: AutoencoderKL loaded fp32 and passed into
  StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(...,
  vae=vae_fp32, torch_dtype=torch.float16) -- SDXL's official VAE NaNs under
  fp16, never "fix" this back. Unlike training, the pipeline's own prompt=
  argument handles the dual-encoder embedding internally -- no manual
  embedding code needed here.

Manifest schema is UNCHANGED from infer_colour_translation.py (seed,
source_mode, crop_id, slide, frame, x, y, output_path, reference_path,
aperio_path) -- score_p1_10_ablation.py scores this with zero changes, same
as the SD1.5 version.

Never touches infer_colour_translation.py, infer_colour_lora_sdxl.py, or any
existing eval/ directory -- new script, new eval tag (eval/a2h_cond_r8_sdxl/).

Usage
-----
    # mandatory source-conditioning ablation (run on the overfit-test
    # checkpoint before trusting a full training run):
    python infer_colour_translation_sdxl.py \
        --lora <out>/overfit_test_sdxl/final --controlnet <out>/overfit_test_sdxl/final/controlnet \
        --pairs-dir <pairs>/train --overfit-n 8 \
        --out eval/a2h_cond_r8_sdxl_ablation_correct --source-mode correct

    # VAE-only floor (no checkpoint needed):
    python infer_colour_translation_sdxl.py --vae-only \
        --root data/mitos --heldout pairs/heldout_frames.csv --out eval/vae_floor_sdxl

    # normal inference once the model passes its controls:
    python infer_colour_translation_sdxl.py \
        --lora lora/a2h_cond_r8_sdxl/final --controlnet lora/a2h_cond_r8_sdxl/final/controlnet \
        --root data/mitos --heldout pairs/heldout_frames.csv --out eval/a2h_cond_r8_sdxl \
        --steps 50 --seeds 0 1 2

Dependencies: torch, diffusers, numpy, Pillow, opencv-python-headless, tifffile.
Requires registration.py and canny.py on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="P3-06: SDXL inference for the source-conditioned colour LoRA + ControlNet.")
    ap.add_argument("--lora", default=None,
                    help="Path to the trained SDXL LoRA checkpoint dir "
                         "(e.g. lora/a2h_cond_r8_sdxl/final). Required unless --vae-only.")
    ap.add_argument("--controlnet", default=None,
                    help="Path to the trained SDXL ControlNet dir "
                         "(e.g. lora/a2h_cond_r8_sdxl/final/controlnet). Required unless --vae-only.")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--root", default=None, help="Dataset root (heldout paths are relative to this). "
                    "Mutually exclusive with --pairs-dir.")
    ap.add_argument("--heldout", default=None, help="heldout_frames.csv. Mutually exclusive with --pairs-dir.")
    ap.add_argument("--pairs-dir", default=None,
                    help="Run against the SAME training pairs a checkpoint was (over)fit on, "
                         "instead of held-out frames -- required to validate an --overfit-n "
                         "checkpoint on its own memorised pairs. Mutually exclusive with "
                         "--root/--heldout. Pairs are already 512x512 crops -- no tiling/"
                         "registration needed.")
    ap.add_argument("--overfit-n", type=int, default=0,
                    help="With --pairs-dir: use only the first N pairs (0 = all), matching "
                         "train_colour_translation_lora_sdxl.py's --overfit-n slicing exactly "
                         "(same sort order) so this evaluates precisely the pairs a given "
                         "overfit checkpoint was trained on.")
    ap.add_argument("--val-frames-json", default=None,
                    help="With --pairs-dir: restrict to only the pairs whose frame_id is in this "
                         "training run's saved pair_manifest.json 'val_frames' list -- the "
                         "internal validation split held out during training, NOT the held-out "
                         "A06/A08/A09/A13/A16 test set. Required for P3-07 D4's "
                         "controlnet-conditioning-scale sweep, which must tune on internal "
                         "validation only per this project's methodology guardrail (never tune "
                         "on the full held-out set).")
    ap.add_argument("--out", required=True, help="Output dir for crops + manifest.")
    ap.add_argument("--direction", choices=["A2H", "H2A"], default="A2H")
    ap.add_argument("--source-mode", choices=["correct", "zero", "shuffled"], default="correct",
                    help="Mandatory source-conditioning ablation control. 'correct' is normal "
                         "operation; 'zero'/'shuffled' are diagnostic-only.")
    ap.add_argument("--vae-only", action="store_true",
                    help="VAE-only floor control: skip UNet/ControlNet/LoRA entirely, just VAE "
                         "encode+decode. Ignores --lora/--controlnet/--source-mode.")
    ap.add_argument("--no-lora", action="store_true",
                    help="Diagnostic-only (P3-07 D2): keep the trained ControlNet + frozen SDXL "
                         "base + source conditioning, but skip pipe.load_lora_weights() so the "
                         "colour LoRA is never applied. Isolates whether the base+ControlNet path "
                         "or the colour LoRA itself is responsible for a negative recovery delta. "
                         "--lora is still required (used only to locate --controlnet's sibling "
                         "dir in existing call sites) but its weights are never loaded. Ignored "
                         "with --vae-only (which already skips the LoRA).")
    ap.add_argument("--condition-mode", choices=["rgb_canny", "rgb_only", "canny_only"],
                    default="rgb_canny",
                    help="Diagnostic-only (P3-07 D3): which half of the 6-channel source "
                         "condition [source RGB | source Canny] is actually passed to the "
                         "ControlNet, when --source-mode=correct. rgb_canny = unmodified (the "
                         "trained condition, default). rgb_only = zero the Canny channels. "
                         "canny_only = zero the RGB channels. Tests whether source-RGB "
                         "conditioning specifically (vs edge/morphology structure) is what "
                         "preserves Aperio scanner colour appearance. Does not alter the trained "
                         "weights. Ignored with --vae-only or --source-mode zero/shuffled "
                         "(zero is already an all-channels-zero condition; shuffled swaps the "
                         "whole 6-channel tensor to a different crop's, orthogonal to this split).")
    ap.add_argument("--controlnet-scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=50,
                    help="DDIM steps. Matches P1-10's SD1.5 default -- 50-step DDIM first, "
                         "no LCM here (P1-11-on-SDXL is a separate, later ticket).")
    ap.add_argument("--strength", type=float, default=0.50,
                    help="img2img strength. Single value (matches P1-10's SD1.5 script) -- "
                         "establish conditional-model quality at one operating point first.")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help="Max held-out frames (0 = all).")
    ap.add_argument("--max-crops-per-frame", type=int, default=4, help="0 = all tissue crops.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Multiple seeds -- report mean +/- spread across these, not a "
                         "single-seed point estimate (matches P1-10's seed protocol).")
    ap.add_argument("--online", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.vae_only and (not args.lora or not args.controlnet):
        raise SystemExit("--lora and --controlnet are required unless --vae-only is set.")
    using_pairs_dir = bool(args.pairs_dir)
    using_heldout = bool(args.root or args.heldout)
    if using_pairs_dir and using_heldout:
        raise SystemExit("--pairs-dir is mutually exclusive with --root/--heldout.")
    if not using_pairs_dir and not (args.root and args.heldout):
        raise SystemExit("Either --pairs-dir, or both --root and --heldout, must be given.")
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
        from train_colour_translation_lora_sdxl import build_pairs
    else:
        from registration import read_rgb, register_h_to_a

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("No CUDA device -- inference must run on a GPU node.")

    out_dir = Path(args.out)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)

    # ---- tissue + crop helpers (identical convention across this project) ----
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

    crops = []  # each: {slide, frame, x, y, src, ref, aperio_path}
    if using_pairs_dir:
        pairs = build_pairs(args.pairs_dir)
        if args.val_frames_json:
            with open(args.val_frames_json) as fh:
                val_frames = set(json.load(fh)["val_frames"])
            pairs = [p for p in pairs if p["frame_id"] in val_frames]
            print(f"  Restricted to internal validation split ({args.val_frames_json}): "
                  f"{len(pairs)} pairs from {len(val_frames)} frame(s) {sorted(val_frames)}.")
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
        if args.vae_only:
            return None
        if args.source_mode == "correct":
            return crops[i]["src"]
        if args.source_mode == "zero":
            return None
        return crops[shuffled_idx[i]]["src"]  # shuffled

    def build_control_tensor(rgb, condition_mode="rgb_canny"):
        # Must match train_colour_translation_lora_sdxl.py's PairDataset
        # construction exactly, or the trained ControlNet sees an
        # out-of-distribution input. condition_mode zeroes one half
        # AFTER building the normal 6-channel tensor (P3-07 D3 diagnostic
        # only) -- the trained weights/channel layout never change, only
        # which half carries real information at inference time.
        canny = extract_canny_control_image(rgb)
        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        canny_t = torch.from_numpy(canny.astype(np.float32) / 255.0).permute(2, 0, 1)
        if condition_mode == "rgb_only":
            canny_t = torch.zeros_like(canny_t)
        elif condition_mode == "canny_only":
            rgb_t = torch.zeros_like(rgb_t)
        return torch.cat([rgb_t, canny_t], dim=0).unsqueeze(0)  # 1x6xHxW, [0,1]

    def seed_for(pair_key: str, seed: int) -> int:
        h = hashlib.sha256(f"{pair_key}|{seed}".encode()).hexdigest()
        return int(h[:8], 16)

    # ------------------------------------------------------------------
    # Pipeline: VAE-only floor, or full ControlNet+LoRA img2img (SDXL)
    # ------------------------------------------------------------------
    if args.vae_only:
        from diffusers import AutoencoderKL
        # fp32 VAE -- SDXL's official VAE NaNs under fp16 (P3-03's finding,
        # applies identically here -- never "fix" this back to fp16).
        vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32).to(device)
        vae.eval()
        scaling = vae.config.scaling_factor

        @torch.no_grad()
        def run_crop(src_rgb, seed):
            torch.manual_seed(seed)
            arr = torch.from_numpy(src_rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
            arr = arr.unsqueeze(0).to(device, dtype=torch.float32)
            latents = vae.encode(arr).latent_dist.mode() * scaling
            decoded = vae.decode(latents / scaling).sample
            out = ((decoded[0].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            return out
    else:
        from diffusers import (AutoencoderKL, ControlNetModel, DDIMScheduler,
                               StableDiffusionXLControlNetImg2ImgPipeline)
        print(f"Loading P3-06 SDXL pipeline: LoRA={'DISABLED (--no-lora)' if args.no_lora else args.lora}  "
              f"ControlNet={args.controlnet}  source_mode={args.source_mode}  "
              f"condition_mode={args.condition_mode} ...")
        vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
        controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            args.model, controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        if not args.no_lora:
            pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        def run_crop(target_src_rgb, control_src_rgb, seed):
            h, w = target_src_rgb.shape[:2]
            control_tensor = (torch.zeros(1, 6, h, w, dtype=torch.float32)
                              if control_src_rgb is None
                              else build_control_tensor(control_src_rgb, args.condition_mode))
            gen = torch.Generator(device=device).manual_seed(seed)
            out = pipe(prompt=args.prompt, image=Image.fromarray(target_src_rgb), strength=args.strength,
                       num_inference_steps=args.steps, guidance_scale=args.guidance,
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
                 "output_path", "reference_path", "aperio_path"])

    n_out = 0
    seeds = [0] if args.vae_only else args.seeds
    for i, c in enumerate(crops):
        tag = c["tag_id"]
        ref_path = out_dir / "reference" / f"{tag}.png"
        if not ref_path.exists():
            Image.fromarray(c["ref"]).save(ref_path)

        for s in seeds:
            seed_val = seed_for(tag, s)
            if args.vae_only:
                out = run_crop(c["src"], seed_val)
            else:
                out = run_crop(c["src"], control_source_for(i), seed_val)
            suffix = "" if args.vae_only else f"_{args.source_mode}"
            out_path = out_dir / "outputs" / f"{tag}{suffix}_seed{s}.png"
            Image.fromarray(out).save(out_path)
            mw.writerow([s, ("vae_only" if args.vae_only else args.source_mode), tag,
                        c["slide"], c["frame"], c["x"], c["y"],
                        os.path.relpath(out_path, out_dir), os.path.relpath(ref_path, out_dir),
                        c["aperio_path"]])
            n_out += 1

    man.close()
    print(f"\nWrote {n_out} output crops across {len(crops)} locations x {len(seeds)} seed(s), "
          f"source_mode={'vae_only' if args.vae_only else args.source_mode}.")
    print(f"Manifest: {man_path}")
    print("Next: score with score_p1_10_ablation.py against the manifest.")


if __name__ == "__main__":
    main()
