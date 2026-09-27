#!/usr/bin/env python3
"""
floor_check_pixcell.py -- PXC-02 Part B: zero-shot (no fine-tuning)
embedding-conditioned generation floor, scored against real registered
held-out targets. See tickets/PROBE-PIXCELL-TICKETS.md.

Not part of the graded proposal scope -- see that ticket file's own
caveat. This is the real make-or-break number for the whole PixCell
probe: does conditioning purely on the source crop's UNI2-h embedding (no
source-RGB path, no partial denoising of the real image -- PixCell has no
native img2img at all) preserve enough structure for this project's
registered-pair SSIM/PSNR/MAE/LAB-Wasserstein metrics, with ZERO
fine-tuning? PXC-01a (src/eval/probe_pixcell_pipeline.py) already proved
the wiring works end-to-end (UNI2-h -> PixCell-256, 20 steps, guidance
1.5, negative_uni_embeds via pipe.get_unconditional_embedding()) -- this
script reuses that exact confirmed call pattern, not a re-guess.

Held-out crop gathering mirrors benchmark_vae_reconstruction.py's
--stage heldout path exactly (registration.py's register_h_to_a, same
tissue-fraction/grid-offset filters) -- kept as this script's own small
copy, matching this project's established per-script convention (every
sibling eval script -- infer_colour_lora.py, train_atypia_classifier.py,
benchmark_vae_reconstruction.py -- already owns its own copy of this same
~20-line routine rather than importing it, so this is consistent, not a
new pattern).

Scoring reuses metrics.py's score_aligned_pair() unchanged -- per this
project's standing instruction (see PROBE-SD35-TICKETS.md PR-02), never a
parallel metrics implementation.

Usage
-----
    python floor_check_pixcell.py \
        --root <data>/mitos_heldout --heldout <data>/pairs/heldout_frames.csv \
        --limit 2 --out <out_dir>

Dependencies: torch, diffusers, transformers, timm, Pillow, numpy,
opencv-python-headless, scikit-image. Requires metrics.py/registration.py
on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="PXC-02 Part B: zero-shot PixCell generation floor vs. real registered targets.")
    ap.add_argument("--pixcell-repo", default="StonyBrook-CVLab/PixCell-256")
    ap.add_argument("--pixcell-pipeline-repo", default="StonyBrook-CVLab/PixCell-pipeline")
    ap.add_argument("--sd35-vae-repo", default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--uni2h-repo", default="hf-hub:MahmoodLab/UNI2-h")
    ap.add_argument("--root", required=True, help="mitos_heldout dataset root.")
    ap.add_argument("--heldout", required=True, help="heldout_frames.csv.")
    ap.add_argument("--crop", type=int, default=256, help="PixCell-256's native resolution.")
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=2, help="Max held-out frames (cheap first check).")
    ap.add_argument("--max-crops-per-frame", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance-scale", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--online", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import timm
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL, DiffusionPipeline

    from metrics import score_aligned_pair
    from registration import read_rgb, register_h_to_a

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("No CUDA device -- this script is meant for a GPU node.")
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Held-out crop gathering -- same convention as benchmark_vae_
    # reconstruction.py's --stage heldout (registration + tissue filter +
    # grid tiling), crop=256 to match PixCell-256's native resolution.
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

    with open(args.heldout, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    for r in rows:
        r["aperio_path"] = r["aperio_path"].replace("\\", "/")
        r["hamamatsu_path"] = r["hamamatsu_path"].replace("\\", "/")

    crops = []
    for r in rows:
        a_rgb = read_rgb(Path(args.root) / r["aperio_path"])
        h_rgb = read_rgb(Path(args.root) / r["hamamatsu_path"])
        reg = register_h_to_a(a_rgb, h_rgb, ecc_min=args.ecc_min)
        h_reg = reg.h_registered_rgb
        H, W = a_rgb.shape[:2]

        crops_done = 0
        for y in grid_offsets(H, args.crop):
            for x in grid_offsets(W, args.crop):
                if args.max_crops_per_frame and crops_done >= args.max_crops_per_frame:
                    break
                a_c = a_rgb[y:y + args.crop, x:x + args.crop]
                h_c = h_reg[y:y + args.crop, x:x + args.crop]
                if (h_c.max(2) < 6).mean() > 0.10:
                    continue
                if tissue_fraction(a_c) < args.tissue_thresh:
                    continue
                crops.append({"slide": r["aperio_slide"], "frame": r["frame_id"], "x": x, "y": y,
                             "tag_id": f"{r['aperio_slide']}_{r['frame_id']}_x{x}_y{y}",
                             "aperio": a_c, "hamamatsu": h_c})
                crops_done += 1
        print(f"  {r['aperio_slide']}_{r['frame_id']}: {crops_done} crops")
    print(f"Held-out: {len(crops)} crops at {args.crop}x{args.crop}.")
    if not crops:
        raise SystemExit("No crops selected -- check --root/--heldout/--tissue-thresh.")

    # ------------------------------------------------------------------
    # Load models -- exact confirmed pattern from probe_pixcell_pipeline.py
    # (PXC-01a, job 61121, verified working end-to-end).
    # ------------------------------------------------------------------
    print(f"Loading UNI2-h ({args.uni2h_repo}) via timm ...")
    uni2h_kwargs = dict(
        img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
        embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
        reg_tokens=8, dynamic_img_size=True,
    )
    uni_model = timm.create_model(args.uni2h_repo, pretrained=True, **uni2h_kwargs)
    uni_model.eval().to(device)
    uni_config = timm.data.resolve_data_config(uni_model.pretrained_cfg)
    uni_transform = timm.data.create_transform(**uni_config)

    print(f"Loading SD3.5 VAE ({args.sd35_vae_repo}, bf16) ...")
    sd35_vae = AutoencoderKL.from_pretrained(
        args.sd35_vae_repo, subfolder="vae", torch_dtype=torch.bfloat16).to(device)

    print(f"Loading PixCell pipeline ({args.pixcell_repo}) ...")
    pipe = DiffusionPipeline.from_pretrained(
        args.pixcell_repo, vae=sd35_vae,
        custom_pipeline=args.pixcell_pipeline_repo,
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)

    def embed(rgb_uint8):
        img = Image.fromarray(rgb_uint8)
        tensor = uni_transform(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            raw = uni_model(tensor)
        return raw.unsqueeze(1).to(torch.bfloat16)

    negative_uni_embeds = pipe.get_unconditional_embedding(batch_size=1).to(
        device=device, dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    # Generate (zero fine-tuning) + score against the real registered
    # Hamamatsu target AND the raw Aperio source (sanity reference, same
    # convention as this project's other raw-baseline checks).
    # ------------------------------------------------------------------
    metric_keys = ["lab_total", "wlab_mean", "de2000_mean", "ssim", "psnr", "mae"]
    per_crop_path = out_dir / "per_crop.csv"
    fh = open(per_crop_path, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["crop_id", "slide", "frame", "x", "y", "vs",
               *metric_keys, "gen_elapsed_sec"])

    rows_vs_hamamatsu, rows_vs_aperio = [], []
    t_start = time.time()
    for i, c in enumerate(crops):
        uni_embeds = embed(c["aperio"])
        t0 = time.time()
        with torch.inference_mode():
            image = pipe(
                uni_embeds=uni_embeds, negative_uni_embeds=negative_uni_embeds,
                num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                height=args.crop, width=args.crop,
                generator=torch.Generator(device=device).manual_seed(args.seed),
            ).images[0]
        elapsed = time.time() - t0
        gen = np.asarray(image.convert("RGB"))
        if gen.shape[:2] != c["hamamatsu"].shape[:2]:
            gen = np.asarray(image.resize((c["hamamatsu"].shape[1], c["hamamatsu"].shape[0])))

        m_ham = score_aligned_pair(gen, c["hamamatsu"])
        m_ape = score_aligned_pair(gen, c["aperio"])
        rows_vs_hamamatsu.append(m_ham)
        rows_vs_aperio.append(m_ape)
        w.writerow([c["tag_id"], c["slide"], c["frame"], c["x"], c["y"], "hamamatsu",
                   *(round(m_ham[k], 5) for k in metric_keys), round(elapsed, 2)])
        w.writerow([c["tag_id"], c["slide"], c["frame"], c["x"], c["y"], "aperio",
                   *(round(m_ape[k], 5) for k in metric_keys), round(elapsed, 2)])
        fh.flush()
        image.save(out_dir / f"{c['tag_id']}_generated.png")
        print(f"  [{i+1}/{len(crops)}] {c['tag_id']}: ssim_vs_hamamatsu={m_ham['ssim']:.4f} "
              f"ssim_vs_aperio={m_ape['ssim']:.4f} lab_vs_hamamatsu={m_ham['lab_total']:.2f} "
              f"({elapsed:.2f}s)")
    fh.close()

    def mean_of(rows, key):
        return sum(r[key] for r in rows) / len(rows)

    summary = {
        "n_crops": len(crops),
        "vs_hamamatsu": {k: mean_of(rows_vs_hamamatsu, k) for k in metric_keys},
        "vs_aperio": {k: mean_of(rows_vs_aperio, k) for k in metric_keys},
        "total_elapsed_sec": round(time.time() - t_start, 1),
    }
    with open(out_dir / "summary.json", "w") as fh2:
        json.dump(summary, fh2, indent=2)
    print(f"\n[ALL n={len(crops)}] vs_hamamatsu: ssim={summary['vs_hamamatsu']['ssim']:.4f} "
          f"lab_total={summary['vs_hamamatsu']['lab_total']:.2f} "
          f"psnr={summary['vs_hamamatsu']['psnr']:.2f} mae={summary['vs_hamamatsu']['mae']:.2f}")
    print(f"[ALL n={len(crops)}] vs_aperio (sanity):    ssim={summary['vs_aperio']['ssim']:.4f} "
          f"lab_total={summary['vs_aperio']['lab_total']:.2f}")
    print(f"Written: {out_dir / 'summary.json'}  and  {per_crop_path}")


if __name__ == "__main__":
    main()
