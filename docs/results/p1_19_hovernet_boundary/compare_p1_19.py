"""P1-19: paired Canny vs HoVer-Net comparison over the same held-out crops.
Usage: python compare_p1_19.py canny_fused.csv hv_fused.csv canny_unfused.csv hv_unfused.csv
Delta = HoVer-Net - Canny. SSIM higher is better; LAB / wLAB lower is better."""
import csv
import sys

import numpy as np


def load(path, seed_filter=True):
    rows = {}
    for r in csv.DictReader(open(path, newline="")):
        if seed_filter and "seed" in r and r["seed"] != "0":
            continue
        rows[(r["slide"], r["frame"], r["x"], r["y"])] = r
    return rows


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def report(name, a, b):
    keys = sorted(set(a) & set(b))
    print(f"\n=== {name}: {len(keys)} paired crops (canny {len(a)}, hovernet {len(b)}) ===")
    scopes = {"ALL": keys, "typical": [k for k in keys if k[0] != "A06"], "A06": [k for k in keys if k[0] == "A06"]}
    for s in sorted({k[0] for k in keys}):
        scopes[s] = [k for k in keys if k[0] == s]
    print(f"{'scope':8s} {'n':>4s} | {'metric':9s} {'canny':>8s} {'hovernet':>8s} {'delta':>8s} {'95% CI':>20s} {'hv wins':>7s}")
    for scope, ks in scopes.items():
        for met in ("ssim", "wlab_mean", "lab_total"):
            ca = np.array([float(a[k][met]) for k in ks])
            hb = np.array([float(b[k][met]) for k in ks])
            mean, lo, hi = boot_ci(hb - ca)
            wins = (hb > ca).mean() if met == "ssim" else (hb < ca).mean()
            print(f"{scope:8s} {len(ks):4d} | {met:9s} {ca.mean():8.4f} {hb.mean():8.4f} {mean:+8.4f} "
                  f"[{lo:+8.4f}, {hi:+8.4f}] {100 * wins:6.1f}%")


if __name__ == "__main__":
    cf, hf, cu, hu = sys.argv[1:5]
    report("FUSED (F3)", load(cf), load(hf))
    report("UNFUSED (DDIM inversion)", load(cu), load(hu))
