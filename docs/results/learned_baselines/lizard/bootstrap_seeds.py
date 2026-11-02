#!/usr/bin/env python3
"""Relative Dice over three training seeds for the learned baselines (Lizard arm, 2026-11-02).

Seed 3407 is per_image_<method>.csv (the original run); seeds 1 and 2 are per_image_<method>_s<N>.csv
(A2H checkpoints baselines/a2h_<method>_s<N>, normalize_lizard_learned.slurm SEED=N). Same paired
bootstrap as bootstrap_relative_dice.py: 130 images resampled with replacement, 10,000 resamples, seed 0.
The 3-seed row averages Dice(B,G) per image over the three seeds before the ratio.

Pre-stated check (fixed before the seed runs): the Lizard result holds if, on every seed, ParamNet's
95% CI lower bound is >= 0.95 and StainGAN and StainNet stay below 0.95.
Writes relative_dice_seeds.csv next to this file. Run from anywhere.
"""
import csv
import numpy as np
from pathlib import Path

here = Path(__file__).resolve().parent


def load(name):
    return {r["image"]: float(r["dice"]) for r in csv.DictReader(open(here / f"per_image_{name}.csv"))}


A = load("original")
keys = sorted(A)
a = np.array([A[k] for k in keys])
rng = np.random.default_rng(0)
idx = rng.integers(0, len(keys), size=(10000, len(keys)))


def row(method, seed, b):
    boot = b[idx].mean(1) / a[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"method": method, "seed": seed, "dice_B": round(b.mean(), 5),
            "relative_dice": round(b.mean() / a.mean(), 5), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)}


rows = []
for m in ["stainnet", "paramnet", "staingan"]:
    per_seed = []
    for seed, name in [(3407, m), (1, f"{m}_s1"), (2, f"{m}_s2")]:
        B = load(name)
        assert sorted(B) == keys and len(keys) == 130, (name, len(B))
        b = np.array([B[k] for k in keys])
        per_seed.append(b)
        rows.append(row(m, seed, b))
    rows.append(row(m, "mean3", np.mean(per_seed, axis=0)))

with open(here / "relative_dice_seeds.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)

by = lambda m: [r for r in rows if r["method"] == m and r["seed"] != "mean3"]
holds = (all(r["ci95_lo"] >= 0.95 for r in by("paramnet"))
         and all(r["relative_dice"] < 0.95 for r in by("staingan") + by("stainnet")))
print("pre-stated check holds on every seed:", holds)
