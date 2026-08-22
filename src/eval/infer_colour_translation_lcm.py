#!/usr/bin/env python3
"""
infer_colour_translation_lcm.py -- P1-10 + LCM-LoRA acceleration: denoising-
strength sweep for the source-conditioned colour LoRA + trained ControlNet
(lora/a2h_cond_r8/best) with the pretrained latent-consistency/lcm-lora-sdv1-5
attached as a second adapter, for few-step inference.

Per infer_colour_translation.py's own docstring, LCM acceleration for P1-10
was deliberately deferred to "a later, separate script/run, not this one" --
this is that script. infer_colour_translation.py itself is untouched.

Pipeline construction mirrors infer_colour_translation.py exactly (P1-10's
own custom-trained 6-channel-conditioned ControlNet + StableDiffusionControlNet
Img2ImgPipeline, NOT infer_colour_lora.py's pretrained 3-channel Canny-only
ControlNet -- these are different ControlNet checkpoints entirely). LCM
attachment (named adapters + set_adapters + LCMScheduler) mirrors
infer_colour_lora.py's already-validated --lcm pattern (A4/A5) exactly, since
that composition mechanism is orthogonal to which ControlNet checkpoint is
loaded underneath it.

Step-count note (P1-05's finding, applies identically here since it's a
property of the scheduler/step-count interaction, not the specific adapter
weights): at N inference steps, img2img strength s selects
floor(N*s) real denoising steps -- at N=4 this collided (0.30 and 0.40 both
resolved to 1 real step), so N=8 is the default here too, not 4.

Manifest schema matches infer_colour_lora.py's strength-shaped output
EXACTLY (strength, slide, frame, x, y, output_path, reference_path,
aperio_path) -- NOT the source_mode-shaped schema infer_colour_translation.py/
infer_colour_source_ddim_inversion.py use. This means the existing,
unmodified score_outputs.py (via score_outputs.slurm) scores this directly;
no new scorer needed.

Usage
-----
    python infer_colour_translation_lcm.py \
        --lora lora/a2h_cond_r8/best --controlnet lora/a2h_cond_r8/best/controlnet \
        --root data/mitos --heldout pairs/heldout_frames.csv --out eval/p1_10_lcm_sweep \
        --strengths 0.20 0.30 0.40 0.50 0.70 --steps 8 --guidance 1.5

Dependencies: torch, diffusers, numpy, Pillow, opencv-python-headless, tifffile.
Requires registration.py and canny.py on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="P1-10 + LCM-LoRA: denoising-strength sweep for the source-conditioned checkpoint.")
    ap.add_argument("--lora", required=True,
                    help="Path to the trained P1-10 LoRA checkpoint dir (e.g. lora/a2h_cond_r8/best).")
    ap.add_argument("--controlnet", required=True,
                    help="Path to the trained P1-10 ControlNet dir (e.g. lora/a2h_cond_r8/best/controlnet).")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--root", required=True, help="Dataset root (heldout paths are relative to this).")
    ap.add_argument("--heldout", required=True, help="heldout_frames.csv.")
    ap.add_argument("--out", required=True, help="Output dir for crops + manifest.")
    ap.add_argument("--direction", choices=["A2H", "H2A"], default="A2H")
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.20, 0.30, 0.40, 0.50, 0.70],
                    help="Denoising strengths to sweep, matching P1-09's established boundary points "
                         "for 8-step LCM (only strengths crossing an integer floor(steps*strength) "
                         "boundary are genuinely distinct).")
    ap.add_argument("--steps", type=int, default=8,
                    help="LCM inference steps. 8, not 4 -- avoids the strength-quantisation collision "
                         "P1-05 found at 4 steps (0.30/0.40 both resolved to 1 real step).")
    ap.add_argument("--guidance", type=float, default=1.5,
                    help="LCM's documented recommended guidance range is 1.0-2.0; 1.5 matches A4/A5.")
    ap.add_argument("--lcm-scale", type=float, default=1.0, help="Adapter weight for the LCM-LoRA.")
    ap.add_argument("--controlnet-scale", type=float, default=1.0)
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
    from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetImg2ImgPipeline

    from canny import extract_canny_control_image
    from registration import read_rgb, register_h_to_a

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("No CUDA device -- inference must run on a GPU node.")

    out_dir = Path(args.out)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pipeline: P1-10's custom-trained 6-channel ControlNet + colour LoRA,
    # PLUS the pretrained LCM-LoRA as a second named adapter (infer_colour_
    # lora.py's already-validated multi_lora/set_adapters pattern).
    # ------------------------------------------------------------------
    print(f"Loading P1-10 + LCM pipeline: LoRA={args.lora}  ControlNet={args.controlnet} "
          f"steps={args.steps} guidance={args.guidance} ...")
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.model, controlnet=controlnet, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors", adapter_name="colour")
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5",
                           weight_name="pytorch_lora_weights.safetensors", adapter_name="lcm")
    pipe.set_adapters(["colour", "lcm"], adapter_weights=[1.0, args.lcm_scale])
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # ---- tissue + crop helpers (identical convention to infer_colour_translation.py) ----
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

    def build_control_tensor(rgb):
        # 6-channel conditioning (source RGB + source Canny), matching P1-10's
        # trained ControlNet exactly -- NOT the plain 3-channel Canny image
        # infer_colour_lora.py's pretrained ControlNet expects.
        canny = extract_canny_control_image(rgb)
        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        canny_t = torch.from_numpy(canny.astype(np.float32) / 255.0).permute(2, 0, 1)
        return torch.cat([rgb_t, canny_t], dim=0).unsqueeze(0)  # 1x6xHxW, [0,1]

    with open(args.heldout, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    for r in rows:
        r["aperio_path"] = r["aperio_path"].replace("\\", "/")
        r["hamamatsu_path"] = r["hamamatsu_path"].replace("\\", "/")

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

                tag = f"{r['aperio_slide']}_{r['frame_id']}_x{x}_y{y}"
                ref_path = out_dir / "reference" / f"{tag}.png"
                Image.fromarray(ref_c).save(ref_path)
                control_tensor = build_control_tensor(src_c)

                for s in args.strengths:
                    gen = torch.Generator(device=device).manual_seed(args.seed)
                    out = pipe(prompt=args.prompt, image=Image.fromarray(src_c), strength=float(s),
                              num_inference_steps=args.steps, guidance_scale=args.guidance,
                              control_image=control_tensor, controlnet_conditioning_scale=args.controlnet_scale,
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
    print("Next: score with score_outputs.py (unmodified) -- schema matches infer_colour_lora.py exactly.")


if __name__ == "__main__":
    main()
