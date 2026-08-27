#!/usr/bin/env python3
"""
sample_hist_mitos.py -- MITOS contribution to the A5 histopathology warm-start pool.

Samples N (default 500) UNPAIRED 512x512 x20 crops across all 11 MITOS training
slide pairs (22 files: both Aperio and Hamamatsu), balanced across files and
frames, tissue-filtered, deterministic.

Why unpaired (unlike extract_pairs.py): the A5 warm-start LoRA learns a generic
H&E texture/colour prior, NOT the directional A->H scanner mapping. So there is
no pairing, no coordinate correspondence, and no registration -- individual crops
from either scanner are all just "H&E prior" samples. The A03/H03 colour-LoRA
crops ARE allowed to appear here (A5 is an unpaired prior); the five held-out
test pairs (A06/A08/A09/A13/A16) are EXCLUDED.

Reads only the training_* folders, so held-out slides are excluded by directory
boundary; an explicit ID guard is applied as well.

Usage
-----
    python sample_hist_mitos.py --root data/mitos --out data/hist_lora_pool \
        --n 500 --crop 512 --tissue-thresh 0.30 --seed 0

Outputs
-------
    <out>/patches/mitos_<slide>_<frame>_c<k>.png     (e.g. mitos_A03_00A_c007.png)
    <out>/mitos_manifest.csv

Dependencies: Pillow, numpy.
"""

import argparse
import csv
import glob
import os
import random
import re
import sys

import numpy as np

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except Exception:  # noqa: BLE001
    print("ERROR: Pillow required (pip install Pillow numpy).", file=sys.stderr)
    raise

TRAIN_APERIO = "mitos_atypia_2014_training_aperio"
TRAIN_HAM = "mitos_atypia_2014_training_hamamatsu"
HELDOUT_NUMS = {"06", "08", "09", "13", "16"}
SLIDE_DIR_RE = re.compile(r"^([AH])(\d{2})$")
FRAME_RE = re.compile(r"^[AH]\d{2}_(?P<fid>.+)\.tiff$", re.IGNORECASE)


def frames_x20(root, top, slide):
    """x20 frames for a slide, tolerant of the doubled-directory extraction quirk
    (training_aperio/A03/A03/frames/x20 vs training_aperio/A03/frames/x20)."""
    for candidate in (
        os.path.join(root, top, slide, "frames", "x20"),
        os.path.join(root, top, slide, slide, "frames", "x20"),
    ):
        if os.path.isdir(candidate):
            paths = {}
            for p in sorted(glob.glob(os.path.join(candidate, "*.tiff"))):
                m = FRAME_RE.match(os.path.basename(p))
                if m:
                    paths[m.group("fid").upper()] = p
            if paths:
                return paths
    return {}


def discover_training_files(root):
    """Return [(slide_id, scanner, {frame_id: path}), ...] for all training slides,
    both scanners, excluding held-out numbers."""
    files = []
    for top, scanner in ((TRAIN_APERIO, "Aperio"), (TRAIN_HAM, "Hamamatsu")):
        top_dir = os.path.join(root, top)
        if not os.path.isdir(top_dir):
            print(f"WARNING: missing {top_dir}", file=sys.stderr)
            continue
        for entry in sorted(os.listdir(top_dir)):
            m = SLIDE_DIR_RE.match(entry)
            if not m or m.group(2) in HELDOUT_NUMS:
                continue
            frames = frames_x20(root, top, entry)
            if frames:
                files.append((entry, scanner, frames))
    return files


def tissue_fraction(arr):
    a = arr.astype(np.int16)
    mx = a.max(2); mn = a.min(2)
    return float(((mx < 235) & ((mx - mn) > 12)).mean())


def grid_offsets(length, crop):
    if length <= crop:
        return [0]
    offs = list(range(0, length - crop + 1, crop))
    if offs[-1] != length - crop:
        offs.append(length - crop)
    return sorted(set(offs))


def candidate_boxes(w, h, crop):
    return [(x, y) for y in grid_offsets(h, crop) for x in grid_offsets(w, crop)]


def distribute(n, k):
    """Split n as evenly as possible into k parts summing to exactly n."""
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def main():
    ap = argparse.ArgumentParser(description="Sample unpaired MITOS x20 crops for the A5 pool.")
    ap.add_argument("--root", required=True, help="Dataset root (contains the training_* folders).")
    ap.add_argument("--out", default="data/hist_lora_pool")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--per-frame-cap", type=int, default=4,
                    help="Max crops taken from any single frame (spreads the sample).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = discover_training_files(args.root)
    if not files:
        print("ERROR: no training slides found. Check --root.", file=sys.stderr)
        sys.exit(1)

    slide_ids = [f"{s}({sc[0]})" for s, sc, _ in files]
    print(f"Training files found: {len(files)}  ->  {', '.join(f[0] for f in files)}")
    if len(files) != 22:
        print(f"NOTE: expected 22 files (11 pairs x 2 scanners); found {len(files)}. "
              f"Proceeding with what is present.")

    # shuffle each file's frame order deterministically
    file_frames = []
    for slide, scanner, frames in files:
        fl = list(frames.items())          # [(frame_id, path), ...]
        rng.shuffle(fl)
        file_frames.append((slide, scanner, fl))

    quotas = distribute(args.n, len(files))

    patches_dir = os.path.join(args.out, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    manifest_path = os.path.join(args.out, "mitos_manifest.csv")

    rows = []
    per_slide_count = {}
    k = 0

    def take_from_file(idx, need):
        """Crop+save up to `need` tissue crops from file idx; return count taken."""
        nonlocal k
        slide, scanner, fl = file_frames[idx]
        taken = 0
        for frame_id, path in fl:
            if taken >= need:
                break
            img = Image.open(path).convert("RGB")
            W, Hh = img.size
            boxes = candidate_boxes(W, Hh, args.crop)
            rng.shuffle(boxes)
            per_frame = 0
            arr = np.asarray(img)
            for (x, y) in boxes:
                if taken >= need or per_frame >= args.per_frame_cap:
                    break
                crop_arr = arr[y:y + args.crop, x:x + args.crop]
                if tissue_fraction(crop_arr) < args.tissue_thresh:
                    continue
                name = f"mitos_{slide}_{frame_id}_c{k:04d}.png"
                Image.fromarray(crop_arr).save(os.path.join(patches_dir, name))
                rows.append({
                    "filename": name, "source": "mitos",
                    "original_path": os.path.relpath(path, args.root),
                    "slide_id": slide, "scanner": scanner, "frame_id": frame_id,
                    "x": x, "y": y, "width": args.crop, "height": args.crop,
                    "tissue_frac": round(tissue_fraction(crop_arr), 3),
                })
                per_slide_count[slide] = per_slide_count.get(slide, 0) + 1
                taken += 1
                per_frame += 1
                k += 1
        return taken

    # Pass 1: each file fills its quota.
    shortfall = 0
    for i, need in enumerate(quotas):
        got = take_from_file(i, need)
        shortfall += (need - got)

    # Pass 2: top up any remaining deficit by cycling files with capacity.
    guard = 0
    while len(rows) < args.n and guard < len(files) * 3:
        for i in range(len(files)):
            if len(rows) >= args.n:
                break
            take_from_file(i, 1)
        guard += 1

    # Write manifest.
    with open(manifest_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved {len(rows)} MITOS crops -> {patches_dir}")
    print(f"Manifest: {manifest_path}")
    if len(rows) < args.n:
        print(f"WARNING: only {len(rows)}/{args.n} crops available at tissue-thresh="
              f"{args.tissue_thresh}. Lower it or raise --per-frame-cap.")
    print("\nCrops per slide:")
    for s in sorted(per_slide_count):
        print(f"    {s}: {per_slide_count[s]}")

    # Explicit held-out safety check.
    leaked = [r["slide_id"] for r in rows if r["slide_id"][1:] in HELDOUT_NUMS]
    print(f"\nHeld-out leak check: {len(leaked)} crops from held-out slides "
          f"(must be 0). {'OK' if not leaked else 'FAIL: ' + ','.join(set(leaked))}")


if __name__ == "__main__":
    main()
