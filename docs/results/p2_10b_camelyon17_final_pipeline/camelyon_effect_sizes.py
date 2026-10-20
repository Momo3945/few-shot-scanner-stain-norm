#!/usr/bin/env python3
"""Effect sizes + bootstrap CIs for every CAMELYON17 D_post vs D_pre run (P2-10, P3-09, P2-10b).

The proposal asks for D_post < D_pre "reported with effect sizes and bootstrap confidence intervals".
Per run: per-pair change dD = D_post - D_pre over the 10 centre pairs; mean dD and % change; pairs improved;
two-sided exact sign test; d_z = mean(dD)/sd(dD); 95% bootstrap CI of mean dD resampling the 10 pairs
(10,000 resamples, seed 0). The 10 pairs share centres, so they are not independent: the CI and sign test
are supportive only, as the proposal itself says. Reads the pairwise CSVs in prior/ and ./ ; missing runs
are skipped. Writes camelyon_effect_sizes.csv. Run from this folder.
"""
import csv
from math import comb
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
RUNS = [  # label, D_post file, D_pre file
    ("SD1.5 A4-type (P2-10)", "prior/pairwise_camelyon17_normalised.csv", "prior/pairwise_camelyon17_patches.csv"),
    ("SD1.5 inversion + F3 (P2-10b)", "pairwise_inv_f3.csv", "prior/pairwise_camelyon17_patches.csv"),
    ("SD1.5 inversion only (P2-10b)", "pairwise_inv.csv", "prior/pairwise_camelyon17_patches.csv"),
    ("SDXL 144-pair + F3 (P3-09)", "prior/pairwise_p3_09_sdxl_fused.csv", "prior/pairwise_p3_09_camelyon17_patches_1024.csv"),
    ("SDXL 144-pair unfused (P3-09)", "prior/pairwise_p3_09_sdxl_correct.csv", "prior/pairwise_p3_09_camelyon17_patches_1024.csv"),
]


def load(f):
    return {(r["centre_i"], r["centre_j"]): float(r["lab_total"]) for r in csv.DictReader(open(here / f))}


rows = []
for label, post_f, pre_f in RUNS:
    if not (here / post_f).exists():
        print(f"skip (not present yet): {label}")
        continue
    pre, post = load(pre_f), load(post_f)
    keys = sorted(pre)
    assert len(keys) == 10 and set(keys) == set(post), label
    a = np.array([pre[k] for k in keys]); b = np.array([post[k] for k in keys]); d = b - a
    n_imp = int((d < 0).sum())
    k = min(n_imp, 10 - n_imp)
    p_sign = min(1.0, 2 * sum(comb(10, i) for i in range(k + 1)) / 2 ** 10)
    rng = np.random.default_rng(0)  # per run, so a run's CI does not depend on which other runs exist
    boot = d[rng.integers(0, 10, size=(10000, 10))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append({"run": label, "D_pre": round(a.mean(), 4), "D_post": round(b.mean(), 4),
                 "mean_dD": round(d.mean(), 4), "pct_change": round(100 * d.mean() / a.mean(), 2),
                 "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4), "pairs_improved": n_imp,
                 "sign_test_p_two_sided": round(p_sign, 4), "d_z": round(d.mean() / d.std(ddof=1), 3),
                 "verdict": "PASS" if b.mean() < a.mean() else "FAIL"})
with open(here / "camelyon_effect_sizes.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
