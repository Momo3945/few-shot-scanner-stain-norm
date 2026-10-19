#!/usr/bin/env python3
"""Paired bootstrap CI for Relative Dice = mean Dice(B,G) / mean Dice(A,G) (P2-08b, final SD1.5 pipeline).

Same procedure as docs/results/learned_baselines/lizard/bootstrap_relative_dice.py (P2-11b): resamples the
130 Lizard images with replacement (same indices for A and B -> paired), 10,000 resamples, seed 0.
A = per_image_original.csv (copied from the P2-11b folder; = eval/lizard_original/per_image.csv).
B = per_image_inv_f3.csv / per_image_inv.csv (pulled from eval/lizard_normalised_inv{_f3}/per_image.csv,
scoring jobs 71548/71549). Also reports the paired fused-minus-unfused Dice difference.
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
rows, arrays = [], {}
for name, label in [("inv_f3", "Final pipeline: DDIM inversion + F3 fusion (sigma=8, beta=0.5)"),
                    ("inv", "DDIM inversion only (P1-11, unfused)")]:
    B = load(name)
    keys = sorted(set(A) & set(B))
    assert len(keys) == 130 == len(A) == len(B), (name, len(keys))
    a = np.array([A[k] for k in keys]); b = np.array([B[k] for k in keys])
    arrays[name] = b
    idx = rng.integers(0, len(keys), size=(10000, len(keys)))
    boot = b[idx].mean(1) / a[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append({"method": label, "n_images": len(keys), "dice_A": round(a.mean(), 5), "dice_B": round(b.mean(), 5),
                 "relative_dice": round(b.mean() / a.mean(), 5), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                 "p_ge_0.95": round(float((boot >= 0.95).mean()), 4),
                 "ci_excludes_0.95": bool(hi < 0.95 or lo >= 0.95),
                 "frac_images_B_ge_A": round(float((b >= a).mean()), 4)})
with open(here / "relative_dice_bootstrap.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)

d = arrays["inv_f3"] - arrays["inv"]
idx = rng.integers(0, len(d), size=(10000, len(d)))
lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
print(f"fused - unfused Dice: mean {d.mean():+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], "
      f"fused higher on {(d > 0).sum()}/{len(d)} images")
