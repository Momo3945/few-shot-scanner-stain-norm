#!/usr/bin/env python3
"""Paired bootstrap CI for Relative Dice = mean Dice(B,G) / mean Dice(A,G) (P2-11b Lizard arm).

Resamples the 130 Lizard images with replacement (same indices for A and B -> paired),
10,000 resamples, seed 0. Reads per_image_<method>.csv (pulled from eval/lizard_*/per_image.csv).
Writes relative_dice_bootstrap.csv next to this file. Run from this folder.
"""
import csv
import numpy as np
from pathlib import Path

here = Path(__file__).resolve().parent
def load(name):
    return {r["image"]: float(r["dice"]) for r in csv.DictReader(open(here / f"per_image_{name}.csv"))}

A = load("original")
rng = np.random.default_rng(0)
rows = []
for name, label in [("diffusion_a4_lcm", "Diffusion (P2-08, A4 LCM 0.20)"), ("stainnet", "StainNet"),
                    ("paramnet", "ParamNet"), ("staingan", "StainGAN"),
                    ("macenko", "Macenko"), ("reinhard", "Reinhard"), ("histogram_matching", "Histogram Matching")]:
    B = load(name)
    keys = sorted(set(A) & set(B))
    assert len(keys) == 130 == len(A) == len(B), (name, len(keys))
    a = np.array([A[k] for k in keys]); b = np.array([B[k] for k in keys])
    idx = rng.integers(0, len(keys), size=(10000, len(keys)))
    boot = b[idx].mean(1) / a[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append({"method": label, "n_images": len(keys), "dice_A": round(a.mean(), 5), "dice_B": round(b.mean(), 5),
                 "relative_dice": round(b.mean() / a.mean(), 5), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                 "p_ge_0.95": round(float((boot >= 0.95).mean()), 4),
                 "ci_excludes_0.95": bool(hi < 0.95 or lo >= 0.95)})
with open(here / "relative_dice_bootstrap.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
