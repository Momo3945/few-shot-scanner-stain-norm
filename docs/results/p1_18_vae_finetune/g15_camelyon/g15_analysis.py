#!/usr/bin/env python3
"""P1-18 G1.5: is the fine-tuned decoder (d1_pn03_best) materially worse than ft_mse off-domain?

Reads per_crop.csv from benchmark_vae_reconstruction.py --stage camelyon (copied here from
/datasets/mhoosen/stain-norm/eval/p1_18_g15_camelyon/). Pre-stated gate (2026-11-01, before the run):
pooled over all patches, mean dSSIM (d1_pn03_best - ft_mse) >= -0.01 AND mean dPSNR >= -0.5 dB.
Also reports both deltas per centre, and stock for reference, with a paired bootstrap 95% CI.
Run from this folder:  python g15_analysis.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(here / "per_crop.csv")))
by = defaultdict(dict)  # crop_id -> vae -> row
for r in rows:
    by[r["crop_id"]][r["vae"]] = r
crops = sorted(by)
assert all(set(by[c]) >= {"stock", "ft_mse", "d1_pn03_best"} for c in crops)
rng = np.random.default_rng(0)


def stats(ids, a, b, k):
    d = np.array([float(by[c][b][k]) - float(by[c][a][k]) for c in ids])
    boot = [rng.choice(d, d.size).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return d.mean(), lo, hi


centres = sorted({by[c]["ft_mse"]["slide"] for c in crops})
print(f"{len(crops)} patches, centres {centres}")
for scope, ids in [("ALL", crops)] + [(ce, [c for c in crops if by[c]["ft_mse"]["slide"] == ce]) for ce in centres]:
    means = {v: (np.mean([float(by[c][v]["ssim"]) for c in ids]), np.mean([float(by[c][v]["psnr"]) for c in ids]))
             for v in ("stock", "ft_mse", "d1_pn03_best")}
    ds = stats(ids, "ft_mse", "d1_pn03_best", "ssim")
    dp = stats(ids, "ft_mse", "d1_pn03_best", "psnr")
    print(f"{scope:9s} n={len(ids):4d}  SSIM stock/ft_mse/ft-dec {means['stock'][0]:.4f}/{means['ft_mse'][0]:.4f}/"
          f"{means['d1_pn03_best'][0]:.4f}  dSSIM {ds[0]:+.4f} [{ds[1]:+.4f},{ds[2]:+.4f}]  "
          f"PSNR {means['ft_mse'][1]:.2f}->{means['d1_pn03_best'][1]:.2f}  dPSNR {dp[0]:+.3f} [{dp[1]:+.3f},{dp[2]:+.3f}]")
    if scope == "ALL":
        gate = ds[0] >= -0.01 and dp[0] >= -0.5
        print(f"G1.5 gate (dSSIM >= -0.01 and dPSNR >= -0.5 dB, pooled): {'PASS' if gate else 'FAIL'}")
