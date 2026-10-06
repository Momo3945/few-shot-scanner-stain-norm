#!/usr/bin/env python3
"""
normalize_lizard_learned.py -- StainNet / ParamNet / StainGAN normalisation over a flat
directory of Lizard images (P2-11b structure-safety arm; the learned-baseline counterpart
of normalize_lizard.py, which does the same for the diffusion pipeline).

Produces the "B" side of the G/A/B triangle for lizard_dice.py: run infer_hovernet.slurm
on this script's output, then score_lizard.slurm with --against lizard_original for
Relative Dice = Dice(B,G)/Dice(A,G).

Tiling deliberately mirrors normalize_lizard.py so the comparison with the diffusion
pipeline is like-for-like: reflect-pad each image to a multiple of --crop, run the network
on disjoint crop x crop tiles (the 512 px operating point the held-out eval uses), write
each tile into the canvas, crop back to the source (H, W) and save <out>/<stem>.png.
lizard_dice.py requires the output shape to equal the source shape exactly.

Caveat (not engineered around): ParamNet predicts its colour-map weights per INPUT TILE
from a 128 px resample, so adjacent tiles can get slightly different maps (possible faint
tile seams); StainNet is a fixed per-pixel map and StainGAN a fully convolutional
generator, so they have no such per-tile parameter. --crop 0 processes each whole image in
one pass instead (not the default: it changes ParamNet's operating point vs held-out eval).

Usage
-----
    python normalize_lizard_learned.py --method paramnet \
        --ckpt baselines/a2h_paramnet/paramnet_a2h.pt \
        --images-dir lizard_heldout/images --out eval/lizard_normalised_images_paramnet [--limit 5]

Requires infer_learned_baseline.py (sibling) and src/baselines/models.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from infer_learned_baseline import build_model, make_transform


def parse_args():
    ap = argparse.ArgumentParser(description="Learned-baseline normalisation over a Lizard image dir.")
    ap.add_argument("--method", choices=["stainnet", "paramnet", "staingan"], required=True)
    ap.add_argument("--ckpt", required=True, help="<method>_a2h.pt from train_learned_baseline.py.")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=512, help="Tile size; 0 = whole image in one pass.")
    ap.add_argument("--limit", type=int, default=0, help="Max images (0 = all).")
    return ap.parse_args()


def pad_to_multiple(rgb: np.ndarray, crop: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    ph, pw = (-h) % crop, (-w) % crop
    if ph == 0 and pw == 0:
        return rgb
    mode = "reflect" if ph < h and pw < w else "edge"
    return np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode=mode)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ABORT: no GPU on this node (torch.cuda.is_available() is False).")
    device = torch.device("cuda")

    ck = torch.load(args.ckpt, map_location="cpu")
    if ck["method"] != args.method or ck["direction"] != "A2H":
        raise SystemExit(f"Checkpoint is {ck['method']}/{ck['direction']}; need {args.method}/A2H "
                         "(Lizard is normalised A2H, as in P2-08).")
    transform = make_transform(build_model(args.method, ck).to(device), device)

    images_dir, out_dir = Path(args.images_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"})
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No images found in {images_dir}")
    print(f"Method={args.method}  Ckpt={args.ckpt}  steps={ck['steps']}", flush=True)
    print(f"Normalising {len(paths)} images from {images_dir} (crop={args.crop})", flush=True)

    for p in paths:
        src = np.asarray(Image.open(p).convert("RGB"))
        h, w = src.shape[:2]
        if args.crop == 0:
            result = transform(src)
            n_tiles = 1
        else:
            padded = pad_to_multiple(src, args.crop)
            ph, pw = padded.shape[:2]
            canvas = np.zeros_like(padded)
            for y in range(0, ph, args.crop):
                for x in range(0, pw, args.crop):
                    canvas[y:y + args.crop, x:x + args.crop] = transform(padded[y:y + args.crop, x:x + args.crop])
            result = canvas[:h, :w]
            n_tiles = (ph // args.crop) * (pw // args.crop)
        if result.shape[:2] != (h, w):
            raise RuntimeError(f"{p.name}: output shape {result.shape[:2]} != source {(h, w)}")
        Image.fromarray(result).save(out_dir / f"{p.stem}.png")
        print(f"  {p.stem}: {h}x{w} -> {n_tiles} tiles", flush=True)

    print(f"\nWrote {len(paths)} images to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
