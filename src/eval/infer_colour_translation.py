#!/usr/bin/env python3
"""
infer_colour_translation.py -- P1-10 corrective experiment: inference for the
source-conditioned colour LoRA + ControlNet trained by
train_colour_translation_lora.py. Loads the NEWLY TRAINED ControlNet (not the
pretrained lllyasviel/sd-controlnet-canny used by infer_colour_lora.py) plus
this ticket's own LoRA checkpoint, via StableDiffusionControlNetImg2ImgPipeline
-- same pipeline class already used everywhere else in this project, no new
inference plumbing.

Per the ticket's staged-rollout requirement: deterministic 50-step DDIM only,
no LCM-LoRA, no histopathology warm-start LoRA. Establish the conditional
model's own quality first; only after it passes its quality gate should
acceleration/adapter stacking be layered on (a later, separate script/run, not
this one).

Two mandatory P1-10 diagnostic controls are built into this script directly,
not bolted on after:

  --source-mode {correct,zero,shuffled}
      The source-conditioning ablation. For the SAME target crop, compare:
        correct  -- the real per-crop Canny map (normal operation).
        zero     -- an all-zero control image (the ControlNet branch is
                    structurally present but given no information).
        shuffled -- a different crop's Canny map fed in for this target.
      correct must differ materially from both and win on structural/content
      metrics, or the new branch is being ignored and the ticket has not
      fixed the identified problem (ticket's own acceptance criterion).

  --vae-only
      The VAE-only floor control. Skips the UNet/ControlNet/LoRA entirely --
      encodes then immediately decodes each crop through the frozen VAE
      alone. Separates unavoidable VAE reconstruction loss from denoising
      drift. Scored with the same score_outputs.py pipeline as every other
      run in this project.

Seed protocol: --seeds accepts multiple values (default 3) and each crop's
generator seed is derived deterministically from a hash of
(pair/crop identity, requested seed) rather than resetting to one fixed
global seed for every crop -- report mean +/- spread across seeds, per the
ticket's requirement, not a single-seed point estimate. eval_manifest.csv
gets a "seed" column so score_outputs.py's existing per-crop grouping still
works unchanged; aggregate mean/spread across seeds is left to the analysis
step reading that CSV (same division of labour as score_outputs.py already
splitting by strength).

Usage
-----
    # source-conditioning ablation on the overfit-test checkpoint (mandatory
    # control, run this BEFORE trusting a full training run):
    python infer_colour_translation.py \
        --lora <out>/overfit_test/final --controlnet <out>/overfit_test/final/controlnet \
        --root data/mitos --heldout pairs/heldout_frames.csv \
        --out eval/a2h_cond_r8_ablation_correct --source-mode correct --limit 4
    python infer_colour_translation.py ... --out eval/a2h_cond_r8_ablation_zero --source-mode zero --limit 4
    python infer_colour_translation.py ... --out eval/a2h_cond_r8_ablation_shuffled --source-mode shuffled --limit 4

    # VAE-only floor, on the canonical 496-crop set:
    python infer_colour_translation.py --vae-only \
        --root data/mitos --heldout pairs/heldout_frames.csv --out eval/vae_floor

    # normal inference once the model passes its controls:
    python infer_colour_translation.py \
        --lora lora/a2h_cond_r8/final --controlnet lora/a2h_cond_r8/final/controlnet \
        --root data/mitos --heldout pairs/heldout_frames.csv --out eval/a2h_cond_r8 \
        --steps 50 --seeds 0 1 2

Dependencies: torch, diffusers, numpy, Pillow, opencv-python-headless, tifffile.
Requires registration.py and canny.py on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="P1-10: inference for the source-conditioned colour LoRA + ControlNet.")
    ap.add_argument("--lora", default=None,
                    help="Path to the trained P1-10 LoRA checkpoint dir "
                         "(e.g. lora/a2h_cond_r8/final). Required unless --vae-only.")
    ap.add_argument("--controlnet", default=None,
                    help="Path to the trained P1-10 ControlNet dir "
                         "(e.g. lora/a2h_cond_r8/final/controlnet). Required unless --vae-only.")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--root", default=None, help="Dataset root (heldout paths are relative to this). "
                    "Mutually exclusive with --pairs-dir.")
    ap.add_argument("--heldout", default=None, help="heldout_frames.csv. Mutually exclusive with --pairs-dir.")
    ap.add_argument("--pairs-dir", default=None,
                    help="Run against the SAME training pairs a checkpoint was (over)fit on "
                         "(e.g. pairs/train), instead of held-out frames -- required to actually "
                         "validate an --overfit-n checkpoint on its own memorised pairs, per the "
                         "ticket's overfit-test control. Mutually exclusive with --root/--heldout. "
                         "Pairs are already 512x512 crops -- no tiling/registration needed.")
    ap.add_argument("--overfit-n", type=int, default=0,
                    help="With --pairs-dir: use only the first N pairs (0 = all), matching "
                         "train_colour_translation_lora.py's --overfit-n slicing exactly (same "
                         "sort order) so this evaluates precisely the pairs a given overfit "
                         "checkpoint was trained on.")
    ap.add_argument("--out", required=True, help="Output dir for crops + manifest.")
    ap.add_argument("--direction", choices=["A2H", "H2A"], default="A2H")
    ap.add_argument("--source-mode", choices=["correct", "zero", "shuffled"], default="correct",
                    help="P1-10's mandatory source-conditioning ablation control. 'correct' is "
                         "normal operation; 'zero'/'shuffled' are diagnostic-only.")
    ap.add_argument("--vae-only", action="store_true",
                    help="P1-10's mandatory VAE-only floor control: skip UNet/ControlNet/LoRA "
                         "entirely, just VAE encode+decode. Ignores --lora/--controlnet/--source-mode.")
    ap.add_argument("--controlnet-scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=50,
                    help="DDIM steps. Ticket requires 50-step DDIM first -- no LCM here.")
    ap.add_argument("--strength", type=float, default=0.50,
                    help="img2img strength. Single value (not a sweep like infer_colour_lora.py) "
                         "-- P1-10 establishes conditional-model quality at one operating point "
                         "before any strength/acceleration sweep.")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help="Max held-out frames (0 = all).")
    ap.add_argument("--max-crops-per-frame", type=int, default=4, help="0 = all tissue crops.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Multiple seeds, per the ticket's seed protocol -- report mean +/- "
                         "spread across these, not a single-seed point estimate.")
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
        from train_colour_translation_lora import build_pairs
    else:
        from registration import read_rgb, register_h_to_a

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("No CUDA device -- inference must run on a GPU node.")

    out_dir = Path(args.out)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)

    # ---- tissue + crop helpers (identical convention to infer_colour_lora.py) ----
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
        # ---- overfit-checkpoint control: evaluate the EXACT training pairs a
        # checkpoint was (over)fit on, not held-out frames -- --overfit-n uses
        # the identical slicing as train_colour_translation_lora.py's
        # build_pairs()[:N] (same sort order, same function), so this is
        # provably the same pair set the checkpoint memorised. ----
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
            # tag_id = the full pair_id (e.g. "A03_00A_c003"), NOT slide+frame+x+y --
            # multiple pairs share the same frame with x=y=0 placeholders here (these
            # are pre-cropped 512x512 pair images, not sub-crops of a larger frame), so
            # slide+frame+x+y alone collides and silently overwrites output files
            # (confirmed: the first 8 overfit pairs reduce to only 3 unique frames).
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

        # collect all tissue crops first (source_mode="shuffled" needs the full
        # pool of crops available before it can pick a mismatched one per target)
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
    # guard against any crop being "shuffled" onto itself
    for i in range(len(crops)):
        if shuffled_idx[i] == i:
            j = (i + 1) % len(crops)
            shuffled_idx[i], shuffled_idx[j] = shuffled_idx[j], shuffled_idx[i]

    def control_source_for(i):
        # Returns the RGB source crop to build the 6-channel control tensor
        # from, or None for "zero" (explicit all-zero tensor, not Canny-of-a-
        # black-image, which could behave unpredictably under Otsu
        # thresholding on a zero-variance input).
        if args.vae_only:
            return None
        if args.source_mode == "correct":
            return crops[i]["src"]
        if args.source_mode == "zero":
            return None
        return crops[shuffled_idx[i]]["src"]  # shuffled

    def build_control_tensor(rgb):
        # 6-channel conditioning: source RGB (appearance) + source Canny
        # (structure) -- same construction as train_colour_translation_lora.py's
        # PairDataset, must match exactly or the trained ControlNet sees an
        # out-of-distribution input.
        canny = extract_canny_control_image(rgb)
        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        canny_t = torch.from_numpy(canny.astype(np.float32) / 255.0).permute(2, 0, 1)
        return torch.cat([rgb_t, canny_t], dim=0).unsqueeze(0)  # 1x6xHxW, [0,1]

    def seed_for(pair_key: str, seed: int) -> int:
        h = hashlib.sha256(f"{pair_key}|{seed}".encode()).hexdigest()
        return int(h[:8], 16)

    # ------------------------------------------------------------------
    # Pipeline: VAE-only floor, or full ControlNet+LoRA img2img
    # ------------------------------------------------------------------
    if args.vae_only:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float16).to(device)
        vae.eval()
        scaling = vae.config.scaling_factor

        @torch.no_grad()
        def run_crop(src_rgb, seed):
            # Deterministic reconstruction (.mode(), NOT .sample()) -- this is
            # the "floor" control: it must isolate unavoidable VAE compression
            # loss from denoising drift, so it cannot itself inject sampling
            # randomness. seed is still threaded through (torch.manual_seed)
            # for consistency with every other run_crop's seed protocol, even
            # though .mode() has no randomness left to seed.
            torch.manual_seed(seed)
            arr = torch.from_numpy(src_rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
            arr = arr.unsqueeze(0).to(device, dtype=torch.float16)
            latents = vae.encode(arr).latent_dist.mode() * scaling
            decoded = vae.decode(latents / scaling).sample
            out = ((decoded[0].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            return out
    else:
        from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetImg2ImgPipeline
        print(f"Loading P1-10 pipeline: LoRA={args.lora}  ControlNet={args.controlnet}  "
              f"source_mode={args.source_mode} ...")
        controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            args.model, controlnet=controlnet, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        def run_crop(target_src_rgb, control_src_rgb, seed):
            # Passed as a raw torch.Tensor (not a PIL Image -- PIL caps at 4
            # channels) directly to control_image=; verified against diffusers'
            # VaeImageProcessor.preprocess: a 4D tensor whose channel count
            # doesn't match vae_latent_channels skips PIL-only RGB conversion
            # and (with the ControlNet pipeline's do_normalize=False
            # convention) passes through unchanged in [0,1], which is exactly
            # what build_control_tensor already produces.
            control_tensor = (torch.zeros(1, 6, args.crop, args.crop, dtype=torch.float32)
                              if control_src_rgb is None else build_control_tensor(control_src_rgb))
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
            # Deliberately NOT keyed on source_mode: the ticket's ablation control
            # requires holding noise/timestep constant across correct/zero/shuffled
            # for the same crop, so only the conditioning input varies -- keying the
            # seed on source_mode would give each mode a different noise trajectory
            # and confound the comparison.
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
    print("Next: score with score_outputs.py against the manifest.")


if __name__ == "__main__":
    main()
