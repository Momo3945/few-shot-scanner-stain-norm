#!/usr/bin/env python3
"""
upscale_camelyon_patches.py -- P3-09: upscale CAMELYON17's existing 512x512
patches to 1024x1024 so they match P3-07c/P3-08's native training
resolution. See tickets/PHASE3-TICKETS.md P3-09.

Chosen over a fresh native-1024 local extraction (option 2 in the scoping
discussion) because the raw 232GB CAMELYON17 WSIs were mostly deleted
locally after P2-10's original extraction (only zips kept as backup) --
upscaling the already-uploaded 512 patch set avoids a multi-hour local
re-extraction detour. A freshly-matched D_pre (score_camelyon_wasserstein.py
on THIS script's output, not the original 512 one) keeps the before/after
comparison resolution-fair.

Preserves centre_<i>_patient_<id>/ subdirectory structure exactly, same
convention normalize_camelyon.py and score_camelyon_wasserstein.py both
already rely on.

Usage
-----
    python upscale_camelyon_patches.py \
        --root camelyon17_patches --out camelyon17_patches_1024 --size 1024

Dependencies: Pillow.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="P3-09: upscale CAMELYON17 patches to a target size.")
    ap.add_argument("--root", required=True, help="Dir containing centre_<0-4>_patient_<id>/*.png.")
    ap.add_argument("--out", required=True, help="Output dir, same subdir structure.")
    ap.add_argument("--size", type=int, default=1024)
    return ap.parse_args()


def main():
    args = parse_args()
    from PIL import Image

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(root.glob("centre_*_patient_*/*.png"))
    if not paths:
        raise SystemExit(f"No centre_*_patient_*/*.png patches found under {root}")

    n_ok = 0
    for p in paths:
        img = Image.open(p).convert("RGB")
        if img.size != (args.size, args.size):
            img = img.resize((args.size, args.size), Image.LANCZOS)
        dest_dir = out_dir / p.parent.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        img.save(dest_dir / p.name)
        n_ok += 1
        if n_ok % 200 == 0 or n_ok == len(paths):
            print(f"  {n_ok}/{len(paths)} done")

    print(f"\nUpscaled {n_ok}/{len(paths)} patches ({args.size}x{args.size}) into {out_dir}")


if __name__ == "__main__":
    main()
