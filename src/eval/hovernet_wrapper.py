#!/usr/bin/env python3
"""
hovernet_wrapper.py -- run pretrained HoVer-Net (via tiatoolbox) over a directory
of images, producing a per-image predicted-nucleus-instance dict plus a rasterised
binary nucleus mask for Dice scoring (P2-08, Structural Safety).

This is the "A"/"B" side of the P2-08 triangle (G=Lizard ground truth, A=HoVer-Net
on the original patch, B=HoVer-Net on the normalised patch) -- run this script
identically on the original Lizard images and on their normalised counterparts
(once the colour-LoRA pipeline produces those), then score both against G with
lizard_dice.py.

Uses `mode="tile"` (not "wsi") -- these are ordinary image regions, not
pyramidal whole-slide files, so tiatoolbox's internal tiling+stitching runs the
model at its native patch size and reassembles a full-image instance dict with
no manual tiling on our side (an earlier design draft assumed manual 256x256
tiling via metrics.py's grid_offsets(); that would have introduced boundary
artefacts -- clipped/split nuclei at tile edges -- that tiatoolbox's own
internal stitching is built to avoid).

MUST run in the separate `stainnorm-hovernet` conda env (tiatoolbox 1.6.0), NOT
the main `stainnorm` env -- tiatoolbox's dependency stack downgrades
scipy/scikit-image/opencv-python-headless below what the rest of this project's
metrics pipeline needs (see tickets/PHASE2-TICKETS.md, P2-08).

Usage
-----
    python hovernet_wrapper.py --images-dir lizard_heldout/images \
        --out eval/lizard_original --pretrained-model hovernet_fast-pannuke

Dependencies: tiatoolbox==1.6.0, torch, numpy, opencv-python (stainnorm-hovernet
env). Deliberately re-serialises tiatoolbox's own joblib-based output into
stdlib pickle (see instances_to_mask / main below) so the downstream scoring
script (lizard_dice.py) can run in the plain `stainnorm` env without needing
joblib or tiatoolbox installed there.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np

from progress import progress


def instances_to_mask(inst_dict: dict, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a tiatoolbox instance-prediction dict into a binary nucleus mask.

    `inst_dict` is tiatoolbox's native per-image output:
    {nuc_id: {'contour': [[x,y],...], 'centroid': [x,y], 'type': int, ...}}.
    `shape` is (H, W) of the source image the prediction was made on. Returns a
    bool HxW array, True wherever any nucleus contour covers that pixel -- this
    is the foreground side of the binary Dice metric P2-08 uses. Deliberately
    ignores per-nucleus `type` (tiatoolbox's PanNuke-derived classes and
    Lizard's own class scheme don't share a taxonomy) -- binary/pooled Dice
    sidesteps that mismatch entirely, which is part of why it was chosen over
    an instance- or class-matched Dice (see tickets/PHASE2-TICKETS.md P2-08).
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for inst in inst_dict.values():
        contour = np.asarray(inst["contour"], dtype=np.int32).reshape(-1, 1, 2)
        if contour.shape[0] >= 3:
            cv2.fillPoly(mask, [contour], 1)
    return mask.astype(bool)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run pretrained HoVer-Net (tiatoolbox) over a directory of images.")
    ap.add_argument("--images-dir", required=True, help="Directory of input images.")
    ap.add_argument("--out", required=True,
                    help="Output dir: instances/ (pickled nucleus dicts), "
                         "masks/ (rasterised binary .npy), manifest.csv.")
    ap.add_argument("--pretrained-model", default="hovernet_fast-pannuke",
                    help="tiatoolbox pretrained model name. 'fast' mode's "
                         "256x256 input matches Lizard's native patch convention "
                         "better than 'original' mode's 270x270.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-loader-workers", type=int, default=2)
    ap.add_argument("--num-postproc-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all).")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but torch.cuda.is_available() "
                             "is False -- fail fast rather than silently crawl on CPU "
                             "(see CLAUDE.md: this has burned a job before).")

    # Imported here, not at module load: tiatoolbox pulls in a heavy stack
    # (torch, jupyter deps, etc) that this project deliberately keeps isolated
    # in the stainnorm-hovernet env -- importing at module level would make
    # `python -m py_compile` (or any import of this file) fail outside that env.
    from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
    import joblib  # tiatoolbox dependency, only needed to read its native .dat format

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out)
    inst_dir = out_dir / "instances"
    mask_dir = out_dir / "masks"
    inst_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    )
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    print(f"Running {args.pretrained_model} over {len(image_paths)} images "
          f"from {images_dir} (device={args.device})")

    segmentor = NucleusInstanceSegmentor(
        pretrained_model=args.pretrained_model,
        batch_size=args.batch_size,
        num_loader_workers=args.num_loader_workers,
        num_postproc_workers=args.num_postproc_workers,
    )
    # tiatoolbox writes its own nested output layout under save_dir; we relocate
    # the per-image results into our own stable instances/+masks/ layout below
    # so this script's --out contract doesn't depend on tiatoolbox internals.
    raw_out = out_dir / "_tiatoolbox_raw"
    results = segmentor.predict(
        imgs=[str(p) for p in image_paths],
        save_dir=str(raw_out),
        mode="tile",
        device=args.device,
        crash_on_exception=False,
    )

    man_path = out_dir / "manifest.csv"
    n_ok = 0
    with open(man_path, "w", newline="") as fh:
        mw = csv.writer(fh)
        mw.writerow(["image", "instances_path", "mask_path", "n_instances", "height", "width"])
        for img_path, raw_dat_path in progress(results, desc="rasterising"):
            stem = Path(img_path).stem
            src_dat = Path(f"{raw_dat_path}.dat")
            if not src_dat.exists():
                print(f"  WARNING: no output for {stem} (expected {src_dat}) "
                      f"-- crashed or skipped, see tiatoolbox log above.")
                continue

            inst_dict = joblib.load(src_dat)

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  WARNING: could not re-read {img_path} for shape -- skipping.")
                continue
            h, w = img.shape[:2]
            mask = instances_to_mask(inst_dict, (h, w))

            inst_dest = inst_dir / f"{stem}.pkl"
            with open(inst_dest, "wb") as ifh:
                pickle.dump(inst_dict, ifh)
            mask_dest = mask_dir / f"{stem}.npy"
            np.save(mask_dest, mask)

            mw.writerow([stem,
                        str(inst_dest.relative_to(out_dir)),
                        str(mask_dest.relative_to(out_dir)),
                        len(inst_dict), h, w])
            n_ok += 1

    print(f"\nProcessed {n_ok}/{len(image_paths)} images.")
    print(f"Instances : {inst_dir}")
    print(f"Masks     : {mask_dir}")
    print(f"Manifest  : {man_path}")
    print("Next: score with lizard_dice.py (runs in the plain stainnorm env).")


if __name__ == "__main__":
    main()
