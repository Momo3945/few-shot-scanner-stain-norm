#!/usr/bin/env python3
"""
roundtrip_a06.py -- P2-07 cycle consistency: round-trip reconstruction.

  A06 --LoRA(A->H)--> Hhat06 --LoRA(H->A)--> Ahat06 <--> Original A06

Per sec:experiments "Round Trip Reconstruction" (docs/proposal_draft(6).tex):
this test uses a deterministic 50-step DDIM sampler with the trained colour
LoRA weights only -- NOT LCM (its stochastic drift on the return pass would
contaminate the structural-deviation measurement being tested here) and, per
the proposal's equation, no ControlNet (--controlnet is available for an
opt-in follow-up against the full deployed A3/A4 stack, but is off by
default -- the literal test isolates what the trained colour LoRA alone does
to structure across a round trip). Metrics: grayscale SSIM, PSNR, MAE
(score_aligned_pair() in metrics.py also reports LAB Wasserstein/windowed-LAB/
CIEDE2000 for free -- included as bonus context, not the proposal's headline
metrics for this experiment).

Not a reuse of infer_colour_lora.py: that script always registers a real
Aperio/Hamamatsu pair via ECC and scores against the real registered
reference (register_h_to_a() runs unconditionally for every row, regardless
of --direction). This experiment never touches a real Hamamatsu image at
all -- Hhat06 is synthetic, and Ahat06 is compared back to the SAME original
A06 crop it started from, which is trivially pixel-aligned by construction
(no scanner offset was ever introduced). Forcing that shape through
infer_colour_lora.py's registration-first schema would require a fake
heldout row and would run an unwanted/unneeded ECC alignment step on a
synthetic image. Reuses metrics.py's tissue_fraction/grid_offsets (same
crop-selection helpers infer_colour_lora.py and metrics.py's run_baseline
already use) and score_aligned_pair (SSIM/PSNR/MAE + bonus colour metrics),
and registration.py's read_rgb -- no new metric or crop-selection code.

Two sequential passes, each with its own freshly-loaded single-LoRA pipeline
(not diffusers multi-adapter set_adapters -- unlike A4/A5, only one LoRA is
ever active at a time here, so the plain single-adapter load_lora_weights()
path infer_colour_lora.py already uses for A2/A3 applies directly): pass 1
runs every A06 crop through the A->H LoRA and saves Hhat06 to disk; the first
pipe is then freed (del + empty_cache) before pass 2 loads the H->A LoRA and
reads Hhat06 back in from disk. Scoring runs last, purely on the saved PNGs
(no GPU needed for that part).

Usage
-----
    python roundtrip_a06.py \
        --a2h-lora /datasets/mhoosen/stain-norm/lora/a2h_r8/final \
        --h2a-lora /datasets/mhoosen/stain-norm/lora/h2a_r8/final \
        --root /datasets/mhoosen/stain-norm/mitos_heldout \
        --heldout pairs/heldout_frames.csv \
        --out eval/roundtrip_a06 --strengths 0.2 --steps 50 --limit 2

Dependencies: torch, diffusers, numpy, Pillow, opencv-python-headless,
scipy, scikit-image, tifffile. Requires registration.py and metrics.py on
the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="P2-07 cycle consistency: A06 -> LoRA(A2H) -> LoRA(H2A) -> A06, DDIM only.")
    ap.add_argument("--a2h-lora", required=True, help="Path to trained A->H colour LoRA dir.")
    ap.add_argument("--h2a-lora", required=True, help="Path to trained H->A colour LoRA dir.")
    ap.add_argument("--controlnet", default=None,
                    help="Optional ControlNet repo id (e.g. lllyasviel/sd-controlnet-canny). "
                         "Off by default -- the proposal's round-trip equation names only the "
                         "trained LoRA weights, no structural conditioning. Set this only for a "
                         "deliberate follow-up testing the full deployed A3-style stack.")
    ap.add_argument("--controlnet-scale", type=float, default=1.0)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--root", required=True, help="Dataset root (heldout paths are relative to this).")
    ap.add_argument("--heldout", required=True, help="heldout_frames.csv.")
    ap.add_argument("--out", required=True, help="Output dir for crops + manifest + scores.")
    ap.add_argument("--slide", default="A06", help="aperio_slide to round-trip (proposal names A06).")
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.20],
                    help="Applied identically to both the A2H and H2A pass. Default 0.20: P1-09's "
                         "best general-purpose point, and the strength this experiment's own axis "
                         "(structural preservation) most rewards. Pass extra values for a "
                         "supplementary sweep, matching the P1-09 sweep precedent.")
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps (proposal: deterministic 50).")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help="Max A06 held-out frames (0 = all).")
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
    from diffusers import DDIMScheduler
    if args.controlnet:
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
    else:
        from diffusers import StableDiffusionImg2ImgPipeline

    from registration import read_rgb
    from metrics import score_aligned_pair, tissue_fraction, grid_offsets
    if args.controlnet:
        from canny import extract_canny_control_image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("No CUDA device -- inference must run on a GPU node.")

    out_dir = Path(args.out)
    orig_dir = out_dir / "original"
    hhat_dir = out_dir / "hhat"
    ahat_dir = out_dir / "ahat"
    for d in (orig_dir, hhat_dir, ahat_dir):
        d.mkdir(parents=True, exist_ok=True)

    def load_pipe(lora_dir: str):
        if args.controlnet:
            controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                args.model, controlnet=controlnet, torch_dtype=torch.float16,
                safety_checker=None, requires_safety_checker=False)
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                args.model, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        # weight_name must be explicit: diffusers normally auto-detects it via a Hub
        # API call, which is unavailable under HF_HUB_OFFLINE=1 (set above deliberately).
        pipe.load_lora_weights(lora_dir, weight_name="pytorch_lora_weights.safetensors")
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def run_one(pipe, src_rgb, strength, seed):
        src_pil = Image.fromarray(src_rgb)
        control_kwargs = {}
        if args.controlnet:
            control_kwargs = {
                "control_image": Image.fromarray(extract_canny_control_image(src_rgb)),
                "controlnet_conditioning_scale": args.controlnet_scale,
            }
        gen = torch.Generator(device=device).manual_seed(seed)
        out = pipe(prompt=args.prompt, image=src_pil, strength=float(strength),
                   num_inference_steps=args.steps, guidance_scale=args.guidance,
                   generator=gen, **control_kwargs).images[0]
        return np.asarray(out)

    with open(args.heldout, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["aperio_slide"] == args.slide]
    if not rows:
        raise SystemExit(f"No held-out rows found for slide={args.slide!r} in {args.heldout}")
    if args.limit:
        rows = rows[: args.limit]
    # heldout_frames.csv was generated on Windows and stores backslash-separated
    # relative paths; normalise to forward slashes so Path() joins correctly on Linux.
    for r in rows:
        r["aperio_path"] = r["aperio_path"].replace("\\", "/")

    # ---- select crops once (shared across both passes and every strength) ----
    crops = []  # (frame_id, x, y, a_orig_rgb)
    for r in rows:
        a_rgb = read_rgb(Path(args.root) / r["aperio_path"])
        H, W = a_rgb.shape[:2]
        crops_done = 0
        for y in grid_offsets(H, args.crop):
            for x in grid_offsets(W, args.crop):
                if args.max_crops_per_frame and crops_done >= args.max_crops_per_frame:
                    break
                a_c = a_rgb[y:y + args.crop, x:x + args.crop]
                if tissue_fraction(a_c) < args.tissue_thresh:
                    continue
                crops.append((r["frame_id"], x, y, a_c))
                crops_done += 1
        print(f"  {r['aperio_slide']}_{r['frame_id']}: {crops_done} crops")
    if not crops:
        raise SystemExit("No tissue crops selected -- check --tissue-thresh / held-out inventory.")
    print(f"\n{len(crops)} A06 crops x {len(args.strengths)} strength(s), "
          f"{args.steps}-step DDIM, {'ControlNet ' + args.controlnet if args.controlnet else 'LoRA only'}.")

    for frame_id, x, y, a_c in crops:
        Image.fromarray(a_c).save(orig_dir / f"{frame_id}_x{x}_y{y}.png")

    # ---- pass 1: A -> Hhat (A2H LoRA) ----
    print("\nPass 1/2: A06 -> Hhat06 (A->H LoRA)...")
    pipe_a2h = load_pipe(args.a2h_lora)
    entries = []  # (frame_id, x, y, strength)
    for frame_id, x, y, a_c in crops:
        for s in args.strengths:
            h_hat = run_one(pipe_a2h, a_c, s, args.seed)
            Image.fromarray(h_hat).save(hhat_dir / f"{frame_id}_x{x}_y{y}_s{s:.2f}.png")
            entries.append((frame_id, x, y, s))
    del pipe_a2h
    torch.cuda.empty_cache()

    # ---- pass 2: Hhat -> Ahat (H2A LoRA) ----
    print("\nPass 2/2: Hhat06 -> Ahat06 (H->A LoRA)...")
    pipe_h2a = load_pipe(args.h2a_lora)
    for frame_id, x, y, s in entries:
        h_hat = np.asarray(Image.open(hhat_dir / f"{frame_id}_x{x}_y{y}_s{s:.2f}.png").convert("RGB"))
        a_hat = run_one(pipe_h2a, h_hat, s, args.seed)
        Image.fromarray(a_hat).save(ahat_dir / f"{frame_id}_x{x}_y{y}_s{s:.2f}.png")
    del pipe_h2a
    torch.cuda.empty_cache()

    # ---- scoring: Ahat06 vs the ORIGINAL A06 crop (no registration -- trivially aligned) ----
    print("\nScoring Ahat06 vs original A06...")
    per_crop_path = out_dir / "per_crop.csv"
    metric_keys = ("ssim", "psnr", "mae", "lab_total", "wlab_mean", "de2000_mean")
    by_strength = {s: {k: [] for k in metric_keys} for s in args.strengths}
    with open(per_crop_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strength", "frame_id", "x", "y", *metric_keys])
        for frame_id, x, y, a_c in crops:
            for s in args.strengths:
                a_hat = np.asarray(Image.open(ahat_dir / f"{frame_id}_x{x}_y{y}_s{s:.2f}.png").convert("RGB"))
                m = score_aligned_pair(a_hat, a_c)
                w.writerow([f"{s:.2f}", frame_id, x, y, *(round(m[k], 5) for k in metric_keys)])
                for k in metric_keys:
                    by_strength[s][k].append(m[k])

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strength", "n_crops", *metric_keys])
        for s in args.strengths:
            n = len(by_strength[s]["ssim"])
            means = {k: round(statistics.mean(by_strength[s][k]), 5) for k in metric_keys}
            w.writerow([f"{s:.2f}", n, *(means[k] for k in metric_keys)])
            print(f"  strength={s:.2f}  n={n}  ssim={means['ssim']:.4f}  "
                  f"psnr={means['psnr']:.2f}  mae={means['mae']:.2f}  lab_total={means['lab_total']:.2f}")

    print(f"\nPer-crop : {per_crop_path}")
    print(f"Summary  : {summary_path}")


if __name__ == "__main__":
    main()
