#!/usr/bin/env python3
"""
normalize_lizard_classical.py -- Macenko / Reinhard / Histogram-Matching normalisation over a
flat directory of Lizard images (P2-11b structure-safety arm; the classical counterpart of
normalize_lizard.py / normalize_lizard_learned.py). CPU-only.

Produces the "B" side of the G/A/B triangle for lizard_dice.py: run infer_hovernet.slurm on
this script's output, then score_lizard.slurm with --against lizard_original.

Tiling mirrors the other Lizard normalisers so the three families are like-for-like:
reflect-pad each image to a multiple of --crop, normalise disjoint crop x crop tiles
(the 512 px operating point the held-out eval uses), write each into the canvas, crop back to
the source (H, W) and save <out>/<stem>.png (lizard_dice.py needs shape == source).

Each tile is normalised independently (these methods fit source statistics per input, exactly
as infer_baseline.py does per held-out crop). The fit target is the same fixed few-shot
Hamamatsu reference crop the colour LoRA and every other baseline use (A2H).

Degenerate tiles: held-out crops were tissue-filtered; Lizard tiles are not, so a tile can be
nearly all background (Macenko then has too few stained pixels for the SVD stain estimate and
can divide by zero). Such tiles, and any tile whose transform raises or yields non-finite
intermediate values, are passed through UNCHANGED and counted -- never silently dropped or
replaced with garbage. The count per image is written to <out>/fallbacks.json and printed.

Usage
-----
    python normalize_lizard_classical.py --method macenko \
        --target-image pairs/train/A03_00A_c000_hamamatsu.png \
        --images-dir lizard_heldout/images --out eval/lizard_normalised_images_macenko [--limit 5]

Dependencies: numpy, opencv-python-headless, scikit-image, Pillow. Siblings: baseline_methods.py,
registration.py (read_rgb).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from baseline_methods import (MacenkoNormalizer, ReinhardNormalizer, histogram_match_normalize,
                              rgb_to_od, standardize_brightness)
from registration import read_rgb

MIN_STAINED_PIXELS = 1000   # Macenko's beta=0.15 OD filter must keep at least this many pixels


def parse_args():
    ap = argparse.ArgumentParser(description="Classical normalisation over a Lizard image dir.")
    ap.add_argument("--method", choices=["macenko", "reinhard", "histogram_matching"], required=True)
    ap.add_argument("--target-image", required=True, help="Fixed Hamamatsu reference crop (A2H).")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="Max images (0 = all).")
    return ap.parse_args()


def pad_to_multiple(rgb: np.ndarray, crop: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    ph, pw = (-h) % crop, (-w) % crop
    if ph == 0 and pw == 0:
        return rgb
    mode = "reflect" if ph < h and pw < w else "edge"
    return np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode=mode)


def too_few_stained_pixels(tile: np.ndarray, beta: float = 0.15) -> bool:
    od = rgb_to_od(standardize_brightness(tile)).reshape(-1, 3)
    return int((od > beta).any(axis=1).sum()) < MIN_STAINED_PIXELS


def main():
    args = parse_args()
    target = read_rgb(args.target_image)
    if args.method == "macenko":
        norm = MacenkoNormalizer(); norm.fit(target); core = norm.transform
    elif args.method == "reinhard":
        norm = ReinhardNormalizer(); norm.fit(target); core = norm.transform
    else:
        core = lambda src: histogram_match_normalize(src, target)

    def transform(tile: np.ndarray):
        """Return (normalised tile, fell_back: bool)."""
        if args.method == "macenko" and too_few_stained_pixels(tile):
            return tile, True
        try:
            # divide-by-zero / invalid -> exception, not silent NaN->0. NOT all="raise": benign
            # underflow in np.exp (od_to_rgb) is routine and must not trigger a fallback.
            with np.errstate(divide="raise", invalid="raise"):
                out = core(tile)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return tile, True
        return out, False

    images_dir, out_dir = Path(args.images_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"})
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No images found in {images_dir}")
    print(f"Method={args.method}  Target={args.target_image}", flush=True)
    print(f"Normalising {len(paths)} images from {images_dir} (crop={args.crop})", flush=True)

    fallbacks, total_tiles, total_fb = {}, 0, 0
    for p in paths:
        src = np.asarray(Image.open(p).convert("RGB"))
        h, w = src.shape[:2]
        padded = pad_to_multiple(src, args.crop)
        ph, pw = padded.shape[:2]
        canvas = np.zeros_like(padded)
        n_tiles = n_fb = 0
        for y in range(0, ph, args.crop):
            for x in range(0, pw, args.crop):
                out, fb = transform(padded[y:y + args.crop, x:x + args.crop])
                canvas[y:y + args.crop, x:x + args.crop] = out
                n_tiles += 1
                n_fb += int(fb)
        result = canvas[:h, :w]
        if result.shape[:2] != (h, w):
            raise RuntimeError(f"{p.name}: output shape {result.shape[:2]} != source {(h, w)}")
        Image.fromarray(result).save(out_dir / f"{p.stem}.png")
        fallbacks[p.stem] = {"tiles": n_tiles, "fallback_tiles": n_fb}
        total_tiles += n_tiles; total_fb += n_fb
        print(f"  {p.stem}: {h}x{w} -> {n_tiles} tiles, {n_fb} passed through unchanged", flush=True)

    with open(out_dir / "fallbacks.json", "w") as fh:
        json.dump({"method": args.method, "total_tiles": total_tiles, "total_fallback_tiles": total_fb,
                   "per_image": fallbacks}, fh, indent=1)
    print(f"\nWrote {len(paths)} images to {out_dir}; {total_fb}/{total_tiles} tiles passed through "
          f"unchanged (degenerate/failed).", flush=True)


if __name__ == "__main__":
    main()
