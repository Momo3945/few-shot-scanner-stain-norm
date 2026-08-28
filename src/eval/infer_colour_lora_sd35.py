#!/usr/bin/env python3
"""
infer_colour_lora_sd35.py -- SD 3.5 A2H colour-LoRA img2img inference (PR-02,
SD3.5 probe). Ported from infer_colour_lora_sdxl.py; NOT the full A0-A5
ablation-ladder generality -- PR-02's scope is LoRA-only colour fidelity, so
ControlNet (PR-03) and LCM/Turbo (PR-04) are intentionally omitted -- see
tickets/PROBE-SD35-TICKETS.md's own warning not to conflate the two.

For each held-out MITOS frame pair it: registers the Hamamatsu frame into the
Aperio grid (real ground truth), tiles tissue crops, and for each crop runs
SD 3.5 img2img (optionally + the trained colour LoRA from PR-01) at each
requested denoising strength. It saves the normalised output crop, the
registered-Hamamatsu reference crop (once per location), and an
eval_manifest.csv pairing them, in the exact schema infer_colour_lora.py /
infer_colour_lora_sdxl.py already write -- score_outputs.py scores this
identically to every other run in this repo, zero changes needed.

Differences from infer_colour_lora_sdxl.py, all required by SD 3.5's MMDiT
architecture (see train_colour_lora_sd35.py's docstring for the full
architecture background):
  - Pipeline: StableDiffusion3Img2ImgPipeline. SD3.5's own
    FlowMatchEulerDiscreteScheduler is the pipeline's native scheduler --
    unlike the SD1.5/SDXL scripts, no scheduler swap is needed at all.
  - VRAM: loading the transformer (bf16, ~16.1GB) + all three text encoders
    (CLIP-L/CLIP-G/T5-XXL fp16, ~11GB) simultaneously would exceed
    bigbatch's 24GB even before activations -- the same shape of problem
    PR-01 hit during training. Fixed the way HuggingFace's own SD3 pipeline
    documents as the standard low-VRAM inference pattern: drop T5-XXL
    entirely via text_encoder_3=None, tokenizer_3=None (T5 is the one text
    encoder SD3's pipeline treats as optional; CLIP-L/CLIP-G are not). This
    brings simultaneous residency to ~17.8GB, in the same range as PR-01's
    proven 17.77GB training peak -- and inference carries no optimizer/
    gradient state on top. Intentional, standard tradeoff (slightly weaker
    prompt conditioning); acceptable since the prompt is a short fixed
    string and this is a feasibility probe, not a rigor-critical result.
  - LoRA loading: pipe.load_lora_weights(...) auto-routes into
    pipe.transformer -- diffusers' StableDiffusion3LoraLoaderMixin detects
    the transformer.*-prefixed keys that
    StableDiffusion3Pipeline.save_lora_weights(transformer_lora_layers=...)
    wrote during PR-01 training. No manual key remapping.
  - local_files_only=True on every from_pretrained call -- same offline-
    mode/sharded-checkpoint fix train_colour_lora_sd35.py needed (the bug is
    in diffusers' shard-loading path generally, not training-specific).
  - --steps default 28 (SD3.5's own documented recommended step count for
    its flow-matching Euler scheduler) -- a different sampler family from
    SD1.5/SDXL's DDIM/LCM defaults, so their 50/8-step conventions don't
    transfer.
  - --crop default 1024, not 512: SD3.5's img2img pipeline defaults to
    1024x1024 output regardless of input image size (unlike SD1.5/SDXL's
    img2img pipelines, which infer output size from the input) -- job
    47457 confirmed a 512-crop run comes back as 1024x1024 output crops
    against a 512x512 reference, crashing score_outputs.py's windowed LAB
    comparison. Rather than downscale SD3.5's natural output back to 512,
    extract 1024x1024 crops directly and score at that resolution -- the
    same choice already established for P3-07's native-1024 SDXL variant
    (`--crop 1024`, same `heldout_frames.csv`/raw tiles, proven to fit).
    height=/width=args.crop is still passed explicitly to every pipe() call
    for correctness at any --crop value, not just the default.
  - VAE NOT forced to fp32 here (unlike train_colour_lora_sd35.py and the
    SDXL inference script) -- SD3.5's own img2img pipeline
    (pipeline_stable_diffusion_3_img2img.py's prepare_latents) casts the
    preprocessed input image to the pipeline's execution dtype before
    calling vae.encode(), not to vae.dtype, so a separately-loaded fp32 VAE
    crashes with "Input type (BFloat16) and bias type (float) should be the
    same" (confirmed via job 47393's traceback). Keeping the whole pipeline
    -- VAE included -- in a single bf16 torch_dtype avoids the mismatch.
    force_upcast=true in SD3's VAE config is specifically an fp16-NaN guard;
    bf16 has fp32's exponent range, so precision risk here is low.

Everything else (tissue/crop helpers, registration.py reuse,
heldout_frames.csv backslash normalisation, eval_manifest.csv schema) is an
unchanged port.

Usage
-----
    python infer_colour_lora_sd35.py \
        --lora /datasets/mhoosen/stain-norm/lora/a2h_r8_sd35/final \
        --root /datasets/mhoosen/stain-norm/mitos_heldout \
        --heldout pairs/heldout_frames.csv \
        --out eval/pr02_sd35 --strengths 0.30 --steps 28 --crop 1024 \
        --limit 5 --max-crops-per-frame 4

Dependencies: torch, diffusers, transformers, peft, safetensors, numpy,
Pillow, opencv-python-headless, tifffile.
Requires registration.py on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="SD3.5 colour-LoRA img2img inference (PR-02 probe).")
    ap.add_argument("--lora", default=None,
                    help="Path to trained SD3.5 colour LoRA dir (contains "
                         "pytorch_lora_weights.safetensors, from PR-01). Omit to run the "
                         "frozen SD3.5 base with no adapter.")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-large",
                    help="HF repo id (resolved from HF_HOME cache; runs offline).")
    ap.add_argument("--root", required=True, help="Dataset root (heldout paths are relative to this).")
    ap.add_argument("--heldout", required=True, help="heldout_frames.csv.")
    ap.add_argument("--out", required=True, help="Output dir for crops + manifest.")
    ap.add_argument("--direction", choices=["A2H", "H2A"], default="A2H",
                    help="A2H: normalise Aperio->Hamamatsu (default, PR-02's scope). H2A: the reverse.")
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.30])
    ap.add_argument("--steps", type=int, default=28,
                    help="Flow-matching Euler steps (SD3.5's own recommended default).")
    ap.add_argument("--guidance", type=float, default=2.0,
                    help="Carried over from the SD1.5 script's exact value for consistency -- "
                         "untuned for SD3.5, out of scope for this probe measurement.")
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--crop", type=int, default=1024,
                    help="SD3.5 img2img defaults to 1024x1024 output regardless of input size -- "
                         "extract crops at that resolution directly (same choice as P3-07's "
                         "native-1024 SDXL variant) rather than downscale back to 512.")
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=5, help="Max held-out frames (0 = all).")
    ap.add_argument("--max-crops-per-frame", type=int, default=4, help="0 = all tissue crops.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--online", action="store_true",
                    help="Allow HF network access (default: offline, use local cache).")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    local_only = not args.online

    import numpy as np
    import torch
    from PIL import Image
    from diffusers import StableDiffusion3Img2ImgPipeline

    from registration import read_rgb, register_h_to_a

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("No CUDA device -- inference must run on a GPU node.")

    out_dir = Path(args.out)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)

    # ---- pipeline: SD3.5 img2img, optionally + the trained colour LoRA.
    # Whole pipeline (VAE included) loaded uniformly in bf16 -- see
    # docstring: the img2img pipeline casts the input image to the
    # pipeline's execution dtype before vae.encode(), so a separately-fp32
    # VAE crashes with a dtype mismatch (job 47393). text_encoder_3 (T5-XXL)
    # dropped -- see docstring for the VRAM rationale; this is SD3's own
    # documented low-VRAM inference pattern, not a hack. ----
    label = "LoRA" if args.lora else "no adapter (SD3.5 base)"
    print(f"Loading SD 3.5 img2img pipeline: {label} ...")
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        text_encoder_3=None, tokenizer_3=None, local_files_only=local_only)
    if args.lora:
        # weight_name must be explicit: diffusers normally auto-detects it via a Hub
        # API call, which is unavailable under HF_HUB_OFFLINE=1 (set above deliberately).
        pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
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

                for s in args.strengths:
                    gen = torch.Generator(device=device).manual_seed(args.seed)
                    # height/width must be explicit: unlike the SD1.5/SDXL img2img
                    # pipelines (which infer output size from the input image),
                    # SD3.5's img2img pipeline defaults to 1024x1024 regardless of
                    # input size -- confirmed via job 47457's scoring crash (a
                    # 1024x1024 output vs a 512x512 reference produced an empty
                    # LAB-tile slice in metrics.py's windowed comparison).
                    out = pipe(prompt=args.prompt, image=src_pil, strength=float(s),
                               height=args.crop, width=args.crop,
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
