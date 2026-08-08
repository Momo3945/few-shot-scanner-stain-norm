#!/usr/bin/env python3
"""
infer_colour_lora.py -- A2 colour-LoRA img2img inference + denoising-strength sweep.

For each held-out MITOS frame pair it: registers the Hamamatsu frame into the Aperio
grid (real ground truth), tiles tissue crops, and for each crop runs SD 1.5 img2img
with the trained colour LoRA at each requested denoising strength. It saves the
normalised output crop, the registered-Hamamatsu reference crop (once per location),
and an eval_manifest.csv pairing them -- which score_outputs.py then scores.

This is the A2 configuration: frozen SD 1.5 base + colour LoRA, standard DDIM.
(No ControlNet -- that is A3; no LCM -- that is A4.)

Strength is the RQ3 variable: low preserves structure but shifts colour weakly;
high shifts colour but risks structural drift. The sweep finds the safe window.

Usage
-----
    python infer_colour_lora.py \
        --lora /datasets/mhoosen/stain-norm/lora/a2h_r8/final \
        --root data/mitos --heldout pairs/heldout_frames.csv \
        --out  eval/a2h_r8 --strengths 0.3 0.4 0.5 --steps 50 \
        --limit 8 --max-crops-per-frame 4

Dependencies: torch, diffusers, numpy, Pillow, opencv-python-headless, tifffile.
Requires registration.py on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Colour-LoRA img2img inference + strength sweep.")
    ap.add_argument("--lora", required=True, help="Path to trained LoRA dir (contains pytorch_lora_weights.safetensors).")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--root", required=True, help="Dataset root (heldout paths are relative to this).")
    ap.add_argument("--heldout", required=True, help="heldout_frames.csv.")
    ap.add_argument("--out", required=True, help="Output dir for crops + manifest.")
    ap.add_argument("--direction", choices=["A2H", "H2A"], default="A2H",
                    help="A2H: normalise Aperio->Hamamatsu (default). H2A: the reverse.")
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.4, 0.5])
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps (proposal reference = 50).")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help="Max held-out frames (0 = all).")
    ap.add_argument("--max-crops-per-frame", type=int, default=4, help="0 = all tissue crops.")
    ap.add_argument("--seed", type=int, default=0)
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
    from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler

    from registration import read_rgb, register_h_to_a

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("No CUDA device -- inference must run on a GPU node.")

    out_dir = Path(args.out)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)

    # ---- pipeline: SD 1.5 img2img + trained colour LoRA, DDIM ----
    print(f"Loading SD 1.5 img2img pipeline + LoRA ({args.lora}) ...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(args.lora)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # ---- tissue + crop helpers (consistent with metrics/extractor) ----
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

    with open(args.heldout, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    # For A2H we normalise Aperio inputs toward Hamamatsu (reference = registered H).
    # For H2A we would swap roles; kept explicit for clarity.
    man_path = out_dir / "eval_manifest.csv"
    man = open(man_path, "w", newline="")
    mw = csv.writer(man)
    mw.writerow(["strength", "slide", "frame", "x", "y",
                 "output_path", "reference_path", "aperio_path"])

    n_out = 0
    for r in rows:
        a_rgb = read_rgb(Path(args.root) / r["aperio_path"])
        h_rgb = read_rgb(Path(args.root) / r["hamamatsu_path"])
        reg = register_h_to_a(a_rgb, h_rgb, ecc_min=args.ecc_min)
        h_reg = reg.h_registered_rgb
        H, W = a_rgb.shape[:2]

        # choose which frame is the "input to normalise" vs the "reference"
        src_frame, ref_frame = (a_rgb, h_reg) if args.direction == "A2H" else (h_reg, a_rgb)

        crops_done = 0
        for y in grid_offsets(H, args.crop):
            for x in grid_offsets(W, args.crop):
                if args.max_crops_per_frame and crops_done >= args.max_crops_per_frame:
                    break
                src_c = src_frame[y:y + args.crop, x:x + args.crop]
                ref_c = ref_frame[y:y + args.crop, x:x + args.crop]
                # skip black registration border + non-tissue
                if (ref_c.max(2) < 6).mean() > 0.10:
                    continue
                if tissue_fraction(src_c) < args.tissue_thresh:
                    continue

                tag = f"{r['aperio_slide']}_{r['frame_id']}_x{x}_y{y}"
                ref_path = out_dir / "reference" / f"{tag}.png"
                Image.fromarray(ref_c).save(ref_path)
                src_pil = Image.fromarray(src_c)

                for s in args.strengths:
                    gen = torch.Generator(device=device).manual_seed(args.seed)
                    out = pipe(prompt=args.prompt, image=src_pil, strength=float(s),
                               num_inference_steps=args.steps, guidance_scale=args.guidance,
                               generator=gen).images[0]
                    out_path = out_dir / "outputs" / f"{tag}_s{s:.2f}.png"
                    out.save(out_path)
                    mw.writerow([f"{s:.2f}", r["aperio_slide"], r["frame_id"], x, y,
                                 os.path.relpath(out_path, out_dir),
                                 os.path.relpath(ref_path, out_dir),
                                 r["aperio_path"]])
                    n_out += 1
                crops_done += 1
        print(f"  {r['aperio_slide']}_{r['frame_id']}: {crops_done} crops x {len(args.strengths)} strengths")

    man.close()
    print(f"\nWrote {n_out} output crops across strengths {args.strengths}.")
    print(f"Manifest: {man_path}")
    print("Next: score with score_outputs.py against the manifest.")


if __name__ == "__main__":
    main()
