#!/usr/bin/env python3
"""
infer_baseline.py -- run a classical stain-normalisation baseline (P2-11) over the
held-out set, producing an eval_manifest.csv in the same schema infer_colour_lora.py
writes. score_outputs.py is fully generic over that schema, so it scores these
outputs with ZERO changes -- this script is the only new inference-side code needed.

CPU-only -- no torch, no GPU node required.

Usage
-----
    python infer_baseline.py --method macenko \
        --root data/mitos --heldout pairs/heldout_frames.csv \
        --target-image pairs/train/A03_00A_c000_hamamatsu.png \
        --out eval/macenko

Dependencies: opencv-python-headless, numpy, scikit-image.
Requires registration.py, metrics.py, baseline_methods.py on the path.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image

from registration import read_rgb, register_h_to_a
from metrics import grid_offsets, tissue_fraction
from baseline_methods import MacenkoNormalizer, ReinhardNormalizer, histogram_match_normalize


def parse_args():
    ap = argparse.ArgumentParser(description="Run a classical stain-normalisation baseline (P2-11).")
    ap.add_argument("--method", choices=["macenko", "reinhard", "histogram_matching"], required=True)
    ap.add_argument("--root", required=True, help="Dataset root (heldout paths are relative to this).")
    ap.add_argument("--heldout", required=True, help="heldout_frames.csv.")
    ap.add_argument("--target-image", required=True,
                    help="Single fixed Hamamatsu reference crop the method fits on "
                         "(few-shot, same scope as the colour LoRA's own training pair).")
    ap.add_argument("--out", required=True, help="Output dir for crops + manifest.")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help="Max held-out frames (0 = all).")
    ap.add_argument("--max-crops-per-frame", type=int, default=4, help="0 = all tissue crops.")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)

    target_rgb = read_rgb(args.target_image)
    print(f"Method={args.method}  Target={args.target_image}")

    if args.method == "macenko":
        normalizer = MacenkoNormalizer()
        normalizer.fit(target_rgb)
        transform = normalizer.transform
    elif args.method == "reinhard":
        normalizer = ReinhardNormalizer()
        normalizer.fit(target_rgb)
        transform = normalizer.transform
    else:  # histogram_matching -- no fit step, target passed at transform time
        transform = lambda src: histogram_match_normalize(src, target_rgb)

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

        # A2H convention (matches infer_colour_lora.py's default): normalise Aperio
        # toward Hamamatsu, reference = registered real Hamamatsu.
        src_frame, ref_frame = a_rgb, h_reg

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

                out_c = transform(src_c)
                out_path = out_dir / "outputs" / f"{tag}.png"
                Image.fromarray(out_c).save(out_path)
                mw.writerow(["na", r["aperio_slide"], r["frame_id"], x, y,
                             os.path.relpath(out_path, out_dir),
                             os.path.relpath(ref_path, out_dir),
                             r["aperio_path"]])
                n_out += 1
                crops_done += 1
        print(f"  {r['aperio_slide']}_{r['frame_id']}: {crops_done} crops")

    man.close()
    print(f"\nWrote {n_out} output crops (method={args.method}).")
    print(f"Manifest: {man_path}")
    print("Next: score with score_outputs.py against the manifest.")


if __name__ == "__main__":
    main()
