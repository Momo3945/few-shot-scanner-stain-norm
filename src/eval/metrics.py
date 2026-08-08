#!/usr/bin/env python3
"""
metrics.py -- colour-fidelity and structural metrics for stain normalisation.

Provides the measurement functions your hypotheses are stated in:
  * LAB-histogram Wasserstein distance   (H2 colour fidelity)
  * grayscale SSIM, PSNR, MAE            (cycle-consistency / pixel error, post-registration)

These are model-independent, so they can be unit-tested now and wired to the A2+
colour-LoRA outputs later with no change.

It also ships a `baseline` CLI: for the held-out MITOS pairs it registers each
Hamamatsu frame into the Aperio grid (via registration.py), tiles tissue crops,
and computes the RAW / do-nothing baseline -- raw Aperio vs real (registered)
Hamamatsu. That is the "Raw Hamamatsu" reference every ablation's recovery delta
is measured against, and it validates the whole register->crop->score stack
before any model exists.

The baseline reports per-slide, flags outlier slides automatically (robust
median/MAD modified z-score), and reports the aggregate both with and without
flagged outliers -- because a single extreme slide (e.g. A06) otherwise distorts
the pooled number and can mask itself under naive mean/std.

Usage
-----
    # baseline over the held-out inventory
    python metrics.py baseline \
        --root data/mitos \
        --heldout pairs/heldout_frames.csv \
        --out pairs/baseline_metrics --crop 512 --tissue-thresh 0.30

    # ad-hoc: score two image files (already aligned)
    python metrics.py pair --a out_hat_H.png --b real_H.png

Dependencies: opencv-python-headless, numpy, scipy, scikit-image, tifffile.
Requires registration.py and progress.py on the path (same folder is fine).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity

from progress import progress
from registration import read_rgb, register_h_to_a


# ----------------------------------------------------------------------
# Colour fidelity
# ----------------------------------------------------------------------
def lab_wasserstein(rgb1: np.ndarray, rgb2: np.ndarray,
                    max_samples: int = 200_000, seed: int = 0) -> dict:
    """Per-channel 1D Wasserstein distance between LAB pixel distributions.

    Returns {'L':.., 'a':.., 'b':.., 'total':..}. Lower is closer. The metric is
    distributional (no spatial alignment needed), so it is valid on unregistered
    crops as well as registered ones.
    """
    lab1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    lab2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    if lab1.shape[0] > max_samples:
        lab1 = lab1[rng.choice(lab1.shape[0], max_samples, replace=False)]
    if lab2.shape[0] > max_samples:
        lab2 = lab2[rng.choice(lab2.shape[0], max_samples, replace=False)]
    out = {}
    for i, ch in enumerate("Lab"):
        out[ch] = float(wasserstein_distance(lab1[:, i], lab2[:, i]))
    out["total"] = out["L"] + out["a"] + out["b"]
    return out


# ----------------------------------------------------------------------
# Structural / pixel error (expects spatially aligned inputs)
# ----------------------------------------------------------------------
def grayscale_ssim(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    g1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2GRAY)
    return float(structural_similarity(g1, g2, data_range=255))


def psnr(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    mse = np.mean((rgb1.astype(np.float64) - rgb2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def mae(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    """Mean absolute error on the 0-255 RGB scale."""
    return float(np.mean(np.abs(rgb1.astype(np.float64) - rgb2.astype(np.float64))))


def score_aligned_pair(rgb_pred: np.ndarray, rgb_ref: np.ndarray) -> dict:
    """All metrics for one aligned (pred, reference) pair."""
    lw = lab_wasserstein(rgb_pred, rgb_ref)
    return {
        "lab_L": lw["L"], "lab_a": lw["a"], "lab_b": lw["b"], "lab_total": lw["total"],
        "ssim": grayscale_ssim(rgb_pred, rgb_ref),
        "psnr": psnr(rgb_pred, rgb_ref),
        "mae": mae(rgb_pred, rgb_ref),
    }


# ----------------------------------------------------------------------
# Tissue + crop helpers (kept consistent with the extractor)
# ----------------------------------------------------------------------
def tissue_fraction(rgb: np.ndarray) -> float:
    a = rgb.astype(np.int16)
    mx = a.max(2); mn = a.min(2)
    tissue = (mx < 235) & ((mx - mn) > 12)
    return float(tissue.mean())


def grid_offsets(length: int, crop: int) -> list[int]:
    if length <= crop:
        return [0]
    offs = list(range(0, length - crop + 1, crop))
    if offs[-1] != length - crop:
        offs.append(length - crop)
    return sorted(set(offs))


# ----------------------------------------------------------------------
# Outlier detection (robust)
# ----------------------------------------------------------------------
def flag_outliers(slide_value: dict, z_thresh: float = 3.5) -> dict:
    """Robust per-slide outlier flags on a scalar (here: mean LAB total).

    Uses the Iglewicz-Hoaglin modified z-score, M = 0.6745*(x - median)/MAD,
    flagging |M| > z_thresh. This is deliberately NOT mean/std: with ~5 slides a
    single extreme member inflates the standard deviation enough to mask itself
    (A06 sits near 2.0 under mean/std but ~35 under median/MAD). Falls back to
    mean/std only when MAD == 0 (all-but-one identical), and never flags with
    fewer than 3 slides.

    Returns {slide: {"z": float, "outlier": bool}}.
    """
    slides = list(slide_value)
    vals = [slide_value[s] for s in slides]
    if len(vals) < 3:
        return {s: {"z": 0.0, "outlier": False} for s in slides}

    med = statistics.median(vals)
    mad = statistics.median([abs(v - med) for v in vals])
    out = {}
    if mad == 0:
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals) or 1.0
        for s, v in zip(slides, vals):
            z = (v - mu) / sd
            out[s] = {"z": z, "outlier": abs(z) > 2.0}
    else:
        for s, v in zip(slides, vals):
            z = 0.6745 * (v - med) / mad
            out[s] = {"z": z, "outlier": abs(z) > z_thresh}
    return out


def _mean_finite(values):
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return round(statistics.mean(finite), 5) if finite else None


# ----------------------------------------------------------------------
# Baseline over the held-out inventory
# ----------------------------------------------------------------------
def run_baseline(root: Path, heldout_csv: Path, out_dir: Path,
                 crop: int, tissue_thresh: float, ecc_min: float,
                 outlier_z: float, limit: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(heldout_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if limit:
        rows = rows[:limit]

    per_crop_path = out_dir / "baseline_per_crop.csv"
    fields = ["aperio_slide", "frame_id", "x", "y", "tissue_frac", "reg_ok", "ecc_score",
              "lab_L", "lab_a", "lab_b", "lab_total", "ssim", "psnr", "mae"]
    metric_keys = ("lab_total", "ssim", "psnr", "mae")
    per_slide = {}
    n_crops = n_frames_flagged = 0

    print(f"Scoring RAW baseline over {len(rows)} held-out frame pairs "
          f"(crop={crop}, tissue>={tissue_thresh:.2f})")
    print("Each frame is registered (affine ECC) before its crops are scored.\n")

    with open(per_crop_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        bar = progress(rows, desc="Scoring", unit="frame")
        for r in bar:
            a_rgb = read_rgb(root / r["aperio_path"])
            h_rgb = read_rgb(root / r["hamamatsu_path"])
            reg = register_h_to_a(a_rgb, h_rgb, ecc_min=ecc_min)
            n_frames_flagged += int(not reg.ok)
            h_reg = reg.h_registered_rgb
            H, W = a_rgb.shape[:2]
            for y in grid_offsets(H, crop):
                for x in grid_offsets(W, crop):
                    a_c = a_rgb[y:y + crop, x:x + crop]
                    h_c = h_reg[y:y + crop, x:x + crop]
                    # skip crops touching the black registration border
                    if (h_c.max(2) < 6).mean() > 0.10:
                        continue
                    tf = tissue_fraction(a_c)
                    if tf < tissue_thresh:
                        continue
                    m = score_aligned_pair(a_c, h_c)   # RAW Aperio vs real Hamamatsu
                    slide = r["aperio_slide"]
                    per_slide.setdefault(slide, {k: [] for k in metric_keys})
                    for k in metric_keys:
                        per_slide[slide][k].append(m[k])
                    n_crops += 1
                    w.writerow({"aperio_slide": slide, "frame_id": r["frame_id"],
                                "x": x, "y": y, "tissue_frac": round(tf, 3),
                                "reg_ok": reg.ok, "ecc_score": round(reg.ecc_score, 5),
                                **{k: round(v, 5) for k, v in m.items()}})
            done_ssim = [v for s in per_slide.values() for v in s["ssim"]]
            status = f"{r['aperio_slide']}_{r['frame_id']} crops={n_crops}"
            if done_ssim:
                status += f" ssim={statistics.mean(done_ssim):.3f}"
            if n_frames_flagged:
                status += f" flagged={n_frames_flagged}"
            bar.set_postfix_str(status)

    # -- per-slide summaries + robust outlier flags ----------------------
    slide_summary = {s: {k: _mean_finite(per_slide[s][k]) for k in metric_keys}
                     for s in per_slide}
    slide_lab = {s: slide_summary[s]["lab_total"] for s in per_slide}
    flags = flag_outliers(slide_lab, z_thresh=outlier_z)
    outlier_slides = {s for s in flags if flags[s]["outlier"]}

    def pool(exclude=frozenset()):
        acc = {k: [] for k in metric_keys}
        for s in per_slide:
            if s in exclude:
                continue
            for k in metric_keys:
                acc[k].extend(per_slide[s][k])
        n = sum(len(per_slide[s]["lab_total"]) for s in per_slide if s not in exclude)
        return {k: _mean_finite(acc[k]) for k in metric_keys}, n

    all_summary, all_n = pool()
    clean_summary, clean_n = pool(exclude=outlier_slides)

    # -- write summary CSV ----------------------------------------------
    summary_path = out_dir / "baseline_summary.csv"
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "n_crops", "lab_total", "ssim", "psnr", "mae",
                    "robust_z", "outlier"])
        w.writerow(["ALL", all_n, all_summary["lab_total"], all_summary["ssim"],
                    all_summary["psnr"], all_summary["mae"], "", ""])
        if outlier_slides:
            w.writerow(["ALL_excl_outliers", clean_n, clean_summary["lab_total"],
                        clean_summary["ssim"], clean_summary["psnr"],
                        clean_summary["mae"], "", ""])
        for s in sorted(per_slide):
            ss = slide_summary[s]
            w.writerow([s, len(per_slide[s]["lab_total"]), ss["lab_total"], ss["ssim"],
                        ss["psnr"], ss["mae"],
                        round(flags[s]["z"], 3), flags[s]["outlier"]])

    # -- stdout report ---------------------------------------------------
    print(f"\nScored {n_crops} tissue crops across {len(rows)} held-out frame pairs.")
    if n_frames_flagged:
        print(f"  {n_frames_flagged} frame(s) had low ECC (< {ecc_min}) "
              f"-- see registration_report.")

    print("\nPer-slide breakdown (RAW Aperio vs registered real Hamamatsu):")
    print(f"  {'slide':7}{'n':>6}{'lab_total':>11}{'ssim':>8}{'psnr':>8}"
          f"{'mae':>8}{'robust_z':>10}  flag")
    for s in sorted(per_slide):
        ss = slide_summary[s]
        flag = "  *** OUTLIER" if flags[s]["outlier"] else ""
        print(f"  {s:7}{len(per_slide[s]['lab_total']):>6}{ss['lab_total']:>11.2f}"
              f"{ss['ssim']:>8.3f}{ss['psnr']:>8.2f}{ss['mae']:>8.2f}"
              f"{flags[s]['z']:>10.2f}{flag}")

    print("\nAggregate:")
    print(f"  ALL ({len(per_slide)} slides)      : "
          f"lab {all_summary['lab_total']}  ssim {all_summary['ssim']}  "
          f"psnr {all_summary['psnr']}  mae {all_summary['mae']}")
    if outlier_slides:
        print(f"  excl. {len(outlier_slides)} outlier ({', '.join(sorted(outlier_slides))}): "
              f"lab {clean_summary['lab_total']}  ssim {clean_summary['ssim']}  "
              f"psnr {clean_summary['psnr']}  mae {clean_summary['mae']}")
        print(f"\n  NOTE: {', '.join(sorted(outlier_slides))} flagged as colour-gap "
              f"outlier(s) (|modified z| > {outlier_z}). Report per-slide; do not let "
              f"the pooled ALL number stand alone.")

    print(f"\nPer-crop : {per_crop_path}")
    print(f"Summary  : {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stain-normalisation metrics.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline", help="Raw held-out baseline over the inventory.")
    b.add_argument("--root", required=True)
    b.add_argument("--heldout", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--crop", type=int, default=512)
    b.add_argument("--tissue-thresh", type=float, default=0.30)
    b.add_argument("--ecc-min", type=float, default=0.30)
    b.add_argument("--outlier-z", type=float, default=3.5,
                   help="Modified z-score threshold for flagging outlier slides.")
    b.add_argument("--limit", type=int, default=0)

    p = sub.add_parser("pair", help="Score two already-aligned image files.")
    p.add_argument("--a", required=True, help="Prediction / output image.")
    p.add_argument("--b", required=True, help="Reference image.")

    args = ap.parse_args()
    if args.cmd == "baseline":
        run_baseline(Path(args.root), Path(args.heldout), Path(args.out),
                     args.crop, args.tissue_thresh, args.ecc_min,
                     args.outlier_z, args.limit)
    elif args.cmd == "pair":
        m = score_aligned_pair(read_rgb(args.a), read_rgb(args.b))
        for k, v in m.items():
            print(f"  {k:10s}: {v:.5f}")


if __name__ == "__main__":
    main()
