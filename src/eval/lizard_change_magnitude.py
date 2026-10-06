#!/usr/bin/env python3
"""
lizard_change_magnitude.py -- how much does each normaliser actually change the Lizard images,
and does the HoVer-Net Dice loss track that change? (P2-11b structure-safety arm; closes the
"5-image spot check" caveat on ParamNet's near-no-op behaviour.) CPU-only.

For every run and every one of the 130 images:
  mad_rgb     mean absolute pixel difference vs the original (0-255 scale, all channels)
  mean_dE76   mean per-pixel CIELAB Euclidean distance (colour change in perceptual units)
  dice_A / dice_B / dice_drop   per-image Dice(original,GT) / Dice(normalised,GT) / A - B
                                (from the lizard_dice.py per_image.csv files)
Per run summary: change-size distribution, Spearman(change, dice_drop), and Dice by tercile of
change size (is the damage concentrated in the images that were changed most?).

Usage
-----
    python lizard_change_magnitude.py --images-dir lizard_heldout/images --eval-root eval \
        --out eval/lizard_change_magnitude \
        --run Diffusion=lizard_normalised_images:lizard_normalised \
        --run ParamNet=lizard_normalised_images_paramnet:lizard_normalised_paramnet ...
        (--run LABEL=<normalised-images dir under eval-root>:<HoVer-Net pred tag under eval-root>)

Dependencies: numpy, scipy, scikit-image, Pillow.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from skimage.color import rgb2lab


def parse_args():
    ap = argparse.ArgumentParser(description="Per-image change magnitude of Lizard normalisers.")
    ap.add_argument("--images-dir", required=True, help="Original Lizard images.")
    ap.add_argument("--eval-root", required=True, help="Dir holding the normalised-image dirs and pred tags.")
    ap.add_argument("--original-tag", default="lizard_original", help="Pred tag of the original-image run.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=IMAGES_DIR:PRED_TAG")
    return ap.parse_args()


def load_dice(path: Path) -> dict[str, float]:
    return {r["image"]: float(r["dice"]) for r in csv.DictReader(open(path))}


def main():
    args = parse_args()
    root, images_dir, out_dir = Path(args.eval_root), Path(args.images_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dice_a = load_dice(root / args.original_tag / "per_image.csv")
    originals = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".tif", ".tiff", ".jpg"})
    assert len(originals) == len(dice_a) == 130, (len(originals), len(dice_a))

    per_rows, summ_rows = [], []
    for spec in args.run:
        label, rest = spec.split("=", 1)
        img_sub, pred_tag = rest.split(":", 1)
        norm_dir = root / img_sub
        dice_b = load_dice(root / pred_tag / "per_image.csv")
        rows = []
        for p in originals:
            a = np.asarray(Image.open(p).convert("RGB"))
            b = np.asarray(Image.open(norm_dir / f"{p.stem}.png").convert("RGB"))
            if a.shape != b.shape:
                raise SystemExit(f"{label}/{p.stem}: shape {b.shape} != original {a.shape}")
            mad = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
            de = float(np.sqrt(((rgb2lab(a) - rgb2lab(b)) ** 2).sum(-1)).mean())
            da, db = dice_a[p.stem], dice_b[p.stem]
            rows.append({"run": label, "image": p.stem, "mad_rgb": round(mad, 3), "mean_dE76": round(de, 3),
                         "dice_A": da, "dice_B": db, "dice_drop": round(da - db, 5)})
        per_rows += rows
        mad = np.array([r["mad_rgb"] for r in rows]); de = np.array([r["mean_dE76"] for r in rows])
        da = np.array([r["dice_A"] for r in rows]); db = np.array([r["dice_B"] for r in rows])
        drop = da - db
        order = np.argsort(mad)
        t = np.array_split(order, 3)
        rho_m = spearmanr(mad, drop); rho_e = spearmanr(de, drop)
        s = {"run": label, "n": len(rows),
             "mad_mean": round(mad.mean(), 2), "mad_median": round(float(np.median(mad)), 2),
             "mad_min": round(mad.min(), 2), "mad_max": round(mad.max(), 2),
             "dE76_mean": round(de.mean(), 2), "dE76_median": round(float(np.median(de)), 2),
             "spearman_mad_vs_dice_drop": round(float(rho_m.statistic), 3),
             "spearman_mad_p": round(float(rho_m.pvalue), 4),
             "spearman_dE_vs_dice_drop": round(float(rho_e.statistic), 3)}
        for name, idx in zip(("low", "mid", "high"), t):
            s[f"rel_dice_{name}_change_tercile"] = round(float(db[idx].mean() / da[idx].mean()), 4)
            s[f"mad_{name}_tercile_mean"] = round(float(mad[idx].mean()), 2)
        summ_rows.append(s)
        print(s, flush=True)

    for name, rows in (("change_magnitude_per_image.csv", per_rows), ("change_magnitude_summary.csv", summ_rows)):
        with open(out_dir / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out_dir}/change_magnitude_per_image.csv ({len(per_rows)} rows) and "
          f"change_magnitude_summary.csv ({len(summ_rows)} runs)", flush=True)


if __name__ == "__main__":
    main()
