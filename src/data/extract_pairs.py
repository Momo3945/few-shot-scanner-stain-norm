#!/usr/bin/env python3
"""
extract_pairs.py -- MITOS-ATYPIA-14 coordinate-corresponding crop-pair extractor.

Builds the few-shot training set for the scanner-colour LoRA (H1): up to
--max-pairs coordinate-corresponding 512x512 crop pairs from A03/H03 x20 frames,
plus a held-out inventory of the matched testing-set frame pairs for the eval
harness to register and score later.

Dataset layout assumed (as confirmed from your tree.json)::

    <root>/
      mitos_atypia_2014_training_aperio/A03/frames/x20/A03_00A.tiff ...
      mitos_atypia_2014_training_hamamatsu/H03/frames/x20/H03_00A.tiff ...
      mitos_atypia_2014_testing_aperio/A06/frames/x20/... (held-out)
      mitos_atypia_2014_testing_hamamatsu/H06/frames/x20/...

Pairing
-------
Aperio frame  A03_<ID>  <-->  Hamamatsu frame  H03_<ID>  (same tissue region).
The two scanners differ in native resolution (Aperio ~1539x1376,
Hamamatsu ~1663x1485), so a crop box in Aperio pixel space is mapped into
Hamamatsu space by the per-axis resolution ratio and resized to the crop size.
These are *coordinate-corresponding*, not pixel-exact -- matching the proposal.
Pixel-exact alignment (affine registration) is a separate, eval-only step and
is deliberately NOT done here.

Usage
-----
    python extract_pairs.py --root /path/to/MITOS --out ./pairs
    python extract_pairs.py --root /path/to/MITOS --out ./pairs \
        --crop 512 --max-pairs 50 --tissue-thresh 0.30 --seed 0

Outputs
-------
    <out>/train/A03_<ID>_c<k>_aperio.png
    <out>/train/A03_<ID>_c<k>_hamamatsu.png
    <out>/train_manifest.csv         (provenance + crop boxes in both spaces)
    <out>/heldout_frames.csv         (matched testing frame pairs for eval)

Dependencies: Pillow, numpy.
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import sys

import numpy as np

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except Exception:  # noqa: BLE001
    print("ERROR: Pillow is required (pip install Pillow numpy).", file=sys.stderr)
    raise

TRAIN_APERIO = "mitos_atypia_2014_training_aperio"
TRAIN_HAM = "mitos_atypia_2014_training_hamamatsu"
TEST_APERIO = "mitos_atypia_2014_testing_aperio"
TEST_HAM = "mitos_atypia_2014_testing_hamamatsu"

HELDOUT_APERIO = ["A06", "A08", "A09", "A13", "A16"]

FRAME_RE = re.compile(r"^[AH]\d{2}_(?P<fid>.+)\.tiff$", re.IGNORECASE)


# ----------------------------------------------------------------------
# Frame discovery / matching
# ----------------------------------------------------------------------
def frames_x20(root, top, slide):
    """Return {frame_id: path} for a slide's x20 frames.

    The training archives extract with a doubled slide directory
    (<top>/A03/A03/frames/x20) while the testing archives do not
    (<top>/A06/frames/x20), so accept either layout.
    """
    candidates = [
        os.path.join(root, top, slide, "frames", "x20"),
        os.path.join(root, top, slide, slide, "frames", "x20"),
    ]
    d = next((c for c in candidates if os.path.isdir(c)), candidates[0])
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.tiff"))):
        m = FRAME_RE.match(os.path.basename(p))
        if m:
            out[m.group("fid").upper()] = p
    return out


def match_frames(aperio_map, ham_map):
    """Return sorted list of (frame_id, aperio_path, ham_path) present in both."""
    ids = sorted(set(aperio_map) & set(ham_map))
    return [(fid, aperio_map[fid], ham_map[fid]) for fid in ids]


# ----------------------------------------------------------------------
# Tissue detection + crop grid
# ----------------------------------------------------------------------
def tissue_fraction(arr):
    """Fraction of non-background pixels. Background in H&E is bright + low sat.

    arr: HxWx3 uint8. A pixel counts as tissue if it is not near-white and has
    some colour saturation.
    """
    rgb = arr.astype(np.int16)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = mx - mn                       # crude saturation
    bright = mx                         # near-white if very bright
    tissue = (bright < 235) & (sat > 12)
    return float(tissue.mean())


def grid_offsets(length, crop, stride):
    """Offsets tiling [0, length) with a final crop flush to the edge."""
    if length <= crop:
        return [0]
    offs = list(range(0, length - crop + 1, stride))
    if offs[-1] != length - crop:
        offs.append(length - crop)
    return sorted(set(offs))


def candidate_boxes(w, h, crop, stride):
    boxes = []
    for y in grid_offsets(h, crop, stride):
        for x in grid_offsets(w, crop, stride):
            boxes.append((x, y, crop, crop))
    return boxes


def map_box_to_hamamatsu(box, a_size, h_size):
    """Map an Aperio pixel box to the corresponding Hamamatsu pixel box."""
    ax, ay, aw, ah = box
    (Wa, Ha), (Wh, Hh) = a_size, h_size
    sx, sy = Wh / Wa, Hh / Ha
    hx = int(round(ax * sx))
    hy = int(round(ay * sy))
    hw = int(round(aw * sx))
    hh = int(round(ah * sy))
    hx = max(0, min(hx, Wh - 1))
    hy = max(0, min(hy, Hh - 1))
    hw = max(1, min(hw, Wh - hx))
    hh = max(1, min(hh, Hh - hy))
    return (hx, hy, hw, hh)


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------
def extract(root, out, crop, max_pairs, tissue_thresh, overlap, seed):
    rng = random.Random(seed)
    ap_map = frames_x20(root, TRAIN_APERIO, "A03")
    hm_map = frames_x20(root, TRAIN_HAM, "H03")
    pairs = match_frames(ap_map, hm_map)
    if not pairs:
        print("ERROR: no matched A03/H03 x20 frames found. Check --root.", file=sys.stderr)
        sys.exit(1)
    print(f"Matched A03/H03 x20 frames: {len(pairs)}")

    stride = max(1, int(round(crop * (1.0 - overlap))))

    # Build candidate crops per frame, keep tissue-rich ones.
    per_frame = []   # list of (frame_id, a_img, h_img, a_size, h_size, [boxes])
    total_candidates = 0
    for fid, ap_path, hm_path in pairs:
        a_img = Image.open(ap_path).convert("RGB")
        h_img = Image.open(hm_path).convert("RGB")
        Wa, Ha = a_img.size
        Wh, Hh = h_img.size
        a_arr = np.asarray(a_img)
        kept = []
        for box in candidate_boxes(Wa, Ha, crop, stride):
            x, y, cw, ch = box
            frac = tissue_fraction(a_arr[y:y + ch, x:x + cw])
            if frac >= tissue_thresh:
                kept.append((box, frac))
        total_candidates += len(kept)
        if kept:
            rng.shuffle(kept)
            per_frame.append((fid, a_img, h_img, (Wa, Ha), (Wh, Hh), kept))
    print(f"Tissue-passing candidate crops across frames: {total_candidates}")
    if total_candidates == 0:
        print("ERROR: no crops passed the tissue threshold; lower --tissue-thresh.",
              file=sys.stderr)
        sys.exit(1)

    # Round-robin across frames so the <=50 budget spreads over tissue regions.
    selected = []   # (frame_id, a_img, h_img, a_size, h_size, box, frac)
    cursors = [0] * len(per_frame)
    while len(selected) < max_pairs:
        progressed = False
        for i, (fid, a_img, h_img, a_size, h_size, kept) in enumerate(per_frame):
            if len(selected) >= max_pairs:
                break
            if cursors[i] < len(kept):
                box, frac = kept[cursors[i]]
                cursors[i] += 1
                selected.append((fid, a_img, h_img, a_size, h_size, box, frac))
                progressed = True
        if not progressed:
            break
    print(f"Selected pairs (capped at {max_pairs}): {len(selected)}")

    # Write crops + manifest.
    train_dir = os.path.join(out, "train")
    os.makedirs(train_dir, exist_ok=True)
    man_path = os.path.join(out, "train_manifest.csv")
    per_frame_count = {}
    with open(man_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "frame_id", "tissue_frac",
                    "aperio_file", "hamamatsu_file",
                    "ax", "ay", "aw", "ah", "hx", "hy", "hw", "hh",
                    "aperio_size", "hamamatsu_size"])
        for k, (fid, a_img, h_img, a_size, h_size, box, frac) in enumerate(selected):
            per_frame_count[fid] = per_frame_count.get(fid, 0) + 1
            ax, ay, aw, ah = box
            hbox = map_box_to_hamamatsu(box, a_size, h_size)
            hx, hy, hw, hh = hbox
            a_crop = a_img.crop((ax, ay, ax + aw, ay + ah))
            h_crop = h_img.crop((hx, hy, hx + hw, hy + hh)).resize(
                (crop, crop), Image.LANCZOS)
            base = f"A03_{fid}_c{k:03d}"
            a_name = base + "_aperio.png"
            h_name = base + "_hamamatsu.png"
            a_crop.save(os.path.join(train_dir, a_name))
            h_crop.save(os.path.join(train_dir, h_name))
            w.writerow([k, fid, f"{frac:.3f}", a_name, h_name,
                        ax, ay, aw, ah, hx, hy, hw, hh,
                        f"{a_size[0]}x{a_size[1]}", f"{h_size[0]}x{h_size[1]}"])

    print(f"\nWrote {len(selected)} crop pairs to {train_dir}")
    print(f"Manifest: {man_path}")
    print("Crops per source frame:")
    for fid in sorted(per_frame_count):
        print(f"    A03_{fid}: {per_frame_count[fid]}")

    if len(selected) < max_pairs:
        print(f"\nNOTE: only {len(selected)} pairs available at tissue-thresh="
              f"{tissue_thresh}. Raise --overlap or lower --tissue-thresh for more.")

    return len(selected)


def inventory_heldout(root, out):
    """Record matched held-out (testing) x20 frame pairs for the eval harness."""
    rows = []
    for a_slide in HELDOUT_APERIO:
        h_slide = "H" + a_slide[1:]
        ap = frames_x20(root, TEST_APERIO, a_slide)
        hm = frames_x20(root, TEST_HAM, h_slide)
        for fid, ap_path, hm_path in match_frames(ap, hm):
            rows.append([a_slide, h_slide, fid,
                         os.path.relpath(ap_path, root),
                         os.path.relpath(hm_path, root)])
    path = os.path.join(out, "heldout_frames.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["aperio_slide", "hamamatsu_slide", "frame_id",
                    "aperio_path", "hamamatsu_path"])
        w.writerows(rows)
    print(f"\nHeld-out matched frame pairs inventoried: {len(rows)} -> {path}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Extract A03/H03 coordinate-corresponding crop pairs.")
    ap.add_argument("--root", required=True, help="Dataset root containing the 4 top folders.")
    ap.add_argument("--out", default="./pairs", help="Output directory.")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--max-pairs", type=int, default=50, help="Few-shot cap (H1).")
    ap.add_argument("--tissue-thresh", type=float, default=0.30,
                    help="Min fraction of tissue pixels to keep a crop.")
    ap.add_argument("--overlap", type=float, default=0.0,
                    help="Crop overlap fraction (0=non-overlapping tiling).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-heldout", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"ERROR: not a directory: {args.root}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.out, exist_ok=True)

    n = extract(args.root, args.out, args.crop, args.max_pairs,
                args.tissue_thresh, args.overlap, args.seed)
    if not args.skip_heldout:
        inventory_heldout(args.root, args.out)

    with open(os.path.join(args.out, "extract_config.json"), "w") as fh:
        json.dump(vars(args) | {"pairs_written": n}, fh, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()
