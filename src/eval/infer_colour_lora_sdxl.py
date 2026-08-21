#!/usr/bin/env python3
"""
infer_colour_lora_sdxl.py -- SDXL img2img inference for the A4 config (P3-03:
transfer of the best SD1.5 rung -- ControlNet-Canny + colour LoRA + LCM-LoRA --
to SDXL). Ported from infer_colour_lora.py; NOT the full A0-A5 ablation-ladder
generality of that script -- P3-03's scope is transferring only A4, so --hist-lora
(A5) is intentionally omitted here.

For each held-out MITOS frame pair it: registers the Hamamatsu frame into the
Aperio grid (real ground truth), tiles tissue crops, and for each crop runs SDXL
img2img (optionally + trained colour LoRA, optionally + ControlNet-Canny
conditioning, optionally + LCM-LoRA/LCMScheduler for few-step inference,
optionally + the frozen histopathology warm-start LoRA for A5/P3-05) at each
requested denoising strength. It saves the normalised output crop, the
registered-Hamamatsu reference crop (once per location), and an
eval_manifest.csv pairing them -- score_outputs.py scores this identically to
every SD1.5 run (confirmed fully generic over resolution, zero changes needed).

--hist-lora (P3-05, added after P3-03's initial transfer) stacks the frozen
SDXL histopathology-prior LoRA (trained by train_hist_lora_sdxl.py) alongside
--lora via the same set_adapters composition already used for --lcm -- exact
port of infer_colour_lora.py's P1-07 extension, generalised from 2-way to
3-way adapter composition.

Differences from infer_colour_lora.py (SD1.5), all required by SDXL:
  - Pipeline classes: StableDiffusionXLControlNetImg2ImgPipeline /
    StableDiffusionXLImg2ImgPipeline (SDXL's dual text encoders + pooled embeds +
    micro-conditioning are handled internally by these pipelines when you just
    pass prompt= -- no manual embedding code needed here, unlike the training
    script which builds them by hand).
  - VAE forced to fp32, loaded separately and passed into from_pretrained(...,
    vae=vae_fp32, torch_dtype=torch.float16) -- SDXL's official VAE is documented
    to NaN under fp16, so the rest of the pipeline (UNet, text encoders,
    ControlNet) stays fp16 but the VAE does not. Do NOT "fix" this back to fp16.
  - --model / --controlnet defaults point at the cached SDXL base +
    diffusers/controlnet-canny-sdxl-1.0 (verified cached, real weights, P3-02).
  - --lcm loads latent-consistency/lcm-lora-sdxl instead of the SD1.5 LCM-LoRA.

Everything else (crop/tiling helpers, registration.py reuse, canny.py reuse, the
set_adapters multi-adapter composition pattern and its explicit weight_name=
requirement under HF_HUB_OFFLINE=1, eval_manifest.csv schema) is an unchanged port.

Usage
-----
    python infer_colour_lora_sdxl.py \
        --lora /datasets/mhoosen/stain-norm/lora/a2h_r8_sdxl/final \
        --root /datasets/mhoosen/stain-norm/mitos_heldout --heldout pairs/heldout_frames.csv \
        --out  eval/a4_sdxl --controlnet diffusers/controlnet-canny-sdxl-1.0 --lcm \
        --strengths 0.20 --steps 8 --limit 8 --max-crops-per-frame 4

Dependencies: torch, diffusers, numpy, Pillow, opencv-python-headless, tifffile.
Requires registration.py and canny.py on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="SDXL colour-LoRA img2img inference (P3-03, A4 config).")
    ap.add_argument("--lora", default=None,
                    help="Path to trained SDXL colour LoRA dir (contains "
                         "pytorch_lora_weights.safetensors). Omit to run the frozen SDXL base.")
    ap.add_argument("--controlnet", default="diffusers/controlnet-canny-sdxl-1.0",
                    help="SDXL ControlNet repo id, resolved from HF_HOME cache. Empty string disables it.")
    ap.add_argument("--controlnet-scale", type=float, default=1.0,
                    help="controlnet_conditioning_scale -- how strongly the Canny map influences "
                         "generation. Only used when --controlnet is set.")
    ap.add_argument("--lcm", action="store_true",
                    help="Attach the pretrained SDXL LCM-LoRA (latent-consistency/lcm-lora-sdxl) "
                         "alongside --lora (if set) via diffusers' multi-adapter set_adapters, and "
                         "switch to LCMScheduler for few-step inference (A4 config).")
    ap.add_argument("--lcm-scale", type=float, default=1.0,
                    help="Adapter weight for the LCM-LoRA when --lcm is set (via set_adapters).")
    ap.add_argument("--hist-lora", default=None,
                    help="Path to the frozen SDXL histopathology warm-start LoRA dir (A5/P3-05, "
                         "lora/hist_r32_sdxl/final). Trained independently by "
                         "train_hist_lora_sdxl.py -- never train jointly with the colour LoRA. "
                         "Composed alongside --lora (if set) at inference via set_adapters, same "
                         "mechanism as --lcm.")
    ap.add_argument("--hist-scale", type=float, default=1.0,
                    help="Adapter weight for the histopathology LoRA when --hist-lora is set.")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--root", required=True, help="Dataset root (heldout paths are relative to this).")
    ap.add_argument("--heldout", required=True, help="heldout_frames.csv.")
    ap.add_argument("--out", required=True, help="Output dir for crops + manifest.")
    ap.add_argument("--direction", choices=["A2H", "H2A"], default="A2H",
                    help="A2H: normalise Aperio->Hamamatsu (default, P3-03's scope). H2A: the reverse.")
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.20],
                    help="Default 0.20: P1-09's best general-purpose SD1.5 operating point, chosen "
                         "so the eventual P3-04 SDXL-vs-SD1.5 comparison is apples-to-apples.")
    ap.add_argument("--steps", type=int, default=8, help="LCM steps (A4 config default).")
    ap.add_argument("--guidance", type=float, default=1.5)
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
    from diffusers import AutoencoderKL, DDIMScheduler
    if args.lcm:
        from diffusers import LCMScheduler
    if args.controlnet:
        from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
    else:
        from diffusers import StableDiffusionXLImg2ImgPipeline

    from registration import read_rgb, register_h_to_a
    if args.controlnet:
        from canny import extract_canny_control_image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("No CUDA device -- inference must run on a GPU node.")

    out_dir = Path(args.out)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)

    # ---- pipeline: SDXL img2img, optionally + trained colour LoRA, optionally +
    # ControlNet, optionally + LCM-LoRA, optionally + hist-LoRA (DDIM otherwise).
    # VAE forced to fp32 -- SDXL's official VAE NaNs under fp16; do not change
    # this back to fp16. multi_lora condition matches infer_colour_lora.py's
    # (SD1.5) exactly: >1 LoRA-type adapter -> named adapters + set_adapters. ----
    multi_lora = args.lcm or bool(args.hist_lora)
    label = " + ".join(filter(None, [
        "LoRA" if args.lora else None, f"ControlNet({args.controlnet})" if args.controlnet else None,
        "Hist-LoRA" if args.hist_lora else None, "LCM-LoRA" if args.lcm else None,
    ])) or "no adapters (SDXL base)"
    print(f"Loading SDXL img2img pipeline: {label} ...")
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    if args.controlnet:
        controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            args.model, controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            args.model, vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = (LCMScheduler if args.lcm else DDIMScheduler).from_config(pipe.scheduler.config)
    active, weights = [], []
    if args.lora:
        # weight_name must be explicit: diffusers normally auto-detects it via a Hub
        # API call, which is unavailable under HF_HUB_OFFLINE=1 (set above deliberately).
        lora_kwargs = {"weight_name": "pytorch_lora_weights.safetensors"}
        if multi_lora:
            lora_kwargs["adapter_name"] = "colour"
        pipe.load_lora_weights(args.lora, **lora_kwargs)
        active.append("colour"); weights.append(1.0)
    if args.hist_lora:
        pipe.load_lora_weights(args.hist_lora,
                                weight_name="pytorch_lora_weights.safetensors", adapter_name="hist")
        active.append("hist"); weights.append(args.hist_scale)
    if args.lcm:
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl",
                                weight_name="pytorch_lora_weights.safetensors",
                                adapter_name="lcm" if multi_lora else None)
        if multi_lora:
            active.append("lcm"); weights.append(args.lcm_scale)
    if multi_lora:
        pipe.set_adapters(active, adapter_weights=weights)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # ---- tissue + crop helpers (consistent with metrics/extractor, unchanged port) ----
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
    # heldout_frames.csv was generated on Windows and stores backslash-separated
    # relative paths; normalise to forward slashes so Path() joins correctly on Linux.
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
                # skip black registration border + non-tissue
                if (ref_c.max(2) < 6).mean() > 0.10:
                    continue
                if tissue_fraction(src_c) < args.tissue_thresh:
                    continue

                tag = f"{r['aperio_slide']}_{r['frame_id']}_x{x}_y{y}"
                ref_path = out_dir / "reference" / f"{tag}.png"
                Image.fromarray(ref_c).save(ref_path)
                src_pil = Image.fromarray(src_c)
                control_kwargs = {}
                if args.controlnet:
                    control_kwargs = {
                        "control_image": Image.fromarray(extract_canny_control_image(src_c)),
                        "controlnet_conditioning_scale": args.controlnet_scale,
                    }

                for s in args.strengths:
                    gen = torch.Generator(device=device).manual_seed(args.seed)
                    out = pipe(prompt=args.prompt, image=src_pil, strength=float(s),
                               num_inference_steps=args.steps, guidance_scale=args.guidance,
                               generator=gen, **control_kwargs).images[0]
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
