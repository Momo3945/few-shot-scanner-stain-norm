#!/usr/bin/env python3
"""
registration.py -- affine ECC registration of a Hamamatsu frame into the Aperio
pixel grid, for held-out pixel-level evaluation ONLY.

Why this exists
---------------
Held-out colour metrics compare a normalised Aperio patch (model output, in the
Aperio grid) against the *real* Hamamatsu ground truth of the same tissue. But
the two scanners sample the same field of view on different pixel grids
(Aperio ~1539x1376, Hamamatsu ~1663x1485), so a naive same-coordinate crop of the
two frames is spatially offset. SSIM/PSNR/MAE on offset patches measure
misalignment, not colour error. This module removes that offset: it resizes the
Hamamatsu frame to the Aperio grid, then affine-ECC registers it into Aperio
coordinates, so a crop at (x, y) means the same tissue in both.

This is deliberately NOT used for training-crop extraction -- training pairs are
coordinate-corresponding, not pixel-exact, by design (see extract_pairs.py).

Registration engine adapted from mitos_paired_patches.py.

CLI (registers the held-out inventory and writes an audit report)::

    python registration.py \
        --root data/mitos \
        --heldout pairs/heldout_frames.csv \
        --out pairs/registered_heldout \
        --save-audit

Dependencies: opencv-python-headless, numpy, tifffile.
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile

from progress import progress

warnings.filterwarnings("ignore")

# ECC score below this is treated as a failed/untrustworthy registration.
DEFAULT_ECC_MIN = 0.30


@dataclass(frozen=True)
class RegistrationResult:
    """Registered Hamamatsu frame (in Aperio grid) plus audit metadata."""

    h_registered_rgb: np.ndarray
    warp_matrix: np.ndarray
    ecc_score: float
    h_original_shape: tuple[int, int]
    a_shape: tuple[int, int]
    h_was_resized: bool
    ok: bool

    @property
    def warp_flat(self) -> list[float]:
        return [float(v) for v in self.warp_matrix.reshape(-1)]


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def read_rgb(path: str | Path) -> np.ndarray:
    """Read TIFF/PNG/JPEG into uint8 RGB HxWx3."""
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        arr = tifffile.imread(str(path))
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Could not read image: {path}")
    arr = np.asarray(arr)

    # CHW -> HWC if needed
    if arr.ndim == 3 and arr.shape[0] in {3, 4} and arr.shape[-1] not in {3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unsupported image shape for {path}: {arr.shape}")

    if path.suffix.lower() not in {".tif", ".tiff"}:  # cv2 reads BGR
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def save_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"Failed to write image: {path}")


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------
def _gray_for_registration(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray.astype(np.float32) / 255.0


def _resize_h_to_a_grid(a_rgb: np.ndarray, h_rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    a_h, a_w = a_rgb.shape[:2]
    h_h, h_w = h_rgb.shape[:2]
    if (a_h, a_w) == (h_h, h_w):
        return h_rgb, False
    shrinking = a_w <= h_w and a_h <= h_h
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(h_rgb, (a_w, a_h), interpolation=interp), True


def register_h_to_a(
    a_rgb: np.ndarray,
    h_rgb: np.ndarray,
    ecc_iterations: int = 5000,
    ecc_eps: float = 1e-7,
    ecc_min: float = DEFAULT_ECC_MIN,
) -> RegistrationResult:
    """Register a Hamamatsu frame into the Aperio grid (pre-resize + affine ECC)."""
    h_original_shape = h_rgb.shape[:2]
    h_resized_rgb, h_was_resized = _resize_h_to_a_grid(a_rgb, h_rgb)

    template = _gray_for_registration(a_rgb)
    moving = _gray_for_registration(h_resized_rgb)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iterations, ecc_eps)

    ok = True
    try:
        ecc_score, warp_matrix = cv2.findTransformECC(
            template, moving, warp_matrix, cv2.MOTION_AFFINE, criteria, None, 1
        )
        ecc_score = float(ecc_score)
    except cv2.error:
        # ECC failed to converge -> fall back to identity (resize-only alignment).
        ecc_score = 0.0
        ok = False

    if ecc_score < ecc_min:
        ok = False

    h_registered = cv2.warpAffine(
        h_resized_rgb,
        warp_matrix,
        dsize=(a_rgb.shape[1], a_rgb.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return RegistrationResult(
        h_registered_rgb=h_registered,
        warp_matrix=warp_matrix,
        ecc_score=ecc_score,
        h_original_shape=h_original_shape,
        a_shape=a_rgb.shape[:2],
        h_was_resized=h_was_resized,
        ok=ok,
    )


# ----------------------------------------------------------------------
# CLI: register the held-out inventory and write an audit report
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Register held-out Hamamatsu frames into Aperio grid.")
    ap.add_argument("--root", required=True, help="Dataset root (paths in heldout csv are relative to this).")
    ap.add_argument("--heldout", required=True, help="heldout_frames.csv from extract_pairs.py.")
    ap.add_argument("--out", required=True, help="Output dir for registered frames + report.")
    ap.add_argument("--ecc-min", type=float, default=DEFAULT_ECC_MIN)
    ap.add_argument("--save-audit", action="store_true",
                    help="Save registered Hamamatsu frames + A/H overlay for visual inspection.")
    ap.add_argument("--limit", type=int, default=0, help="Register only the first N pairs (0 = all).")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.heldout, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    print(f"Registering {len(rows)} held-out frame pairs (affine ECC) -> {out}")
    if args.save_audit:
        print("Audit images enabled: writing registered frame + overlay per pair.")

    report_path = out / "registration_report.csv"
    n_ok = n_fail = 0
    flagged: list[str] = []
    with open(report_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["aperio_slide", "frame_id", "ecc_score", "h_was_resized", "ok",
                    "a_h", "a_w", "h_orig_h", "h_orig_w"])
        bar = progress(rows, desc="Registering", unit="frame")
        for r in bar:
            a_path = root / r["aperio_path"]
            h_path = root / r["hamamatsu_path"]
            a_rgb = read_rgb(a_path)
            h_rgb = read_rgb(h_path)
            reg = register_h_to_a(a_rgb, h_rgb, ecc_min=args.ecc_min)
            n_ok += int(reg.ok)
            n_fail += int(not reg.ok)
            if not reg.ok:
                flagged.append(f"{r['aperio_slide']}_{r['frame_id']}")
            bar.set_postfix_str(
                f"{r['aperio_slide']}_{r['frame_id']} ecc={reg.ecc_score:.3f} "
                f"ok={n_ok} flagged={n_fail}")
            w.writerow([r["aperio_slide"], r["frame_id"], f"{reg.ecc_score:.6f}",
                        reg.h_was_resized, reg.ok,
                        reg.a_shape[0], reg.a_shape[1],
                        reg.h_original_shape[0], reg.h_original_shape[1]])
            if args.save_audit:
                tag = f"{r['aperio_slide']}_{r['frame_id']}"
                save_rgb(out / f"{tag}_H_registered.png", reg.h_registered_rgb)
                overlay = (0.5 * a_rgb + 0.5 * reg.h_registered_rgb).astype(np.uint8)
                save_rgb(out / f"{tag}_overlay.png", overlay)

    print(f"\nRegistered {len(rows)} held-out frame pairs.")
    print(f"  ok (ecc >= {args.ecc_min}): {n_ok}")
    print(f"  flagged (low ecc / failed): {n_fail}")
    if flagged:
        shown = ", ".join(flagged[:10]) + (" ..." if len(flagged) > 10 else "")
        print(f"    {shown}")
    print(f"Report: {report_path}")
    if args.save_audit:
        print(f"Audit images (registered + 50/50 overlay): {out}")


if __name__ == "__main__":
    main()
