"""P1-18 Stage 2a: paired per-crop comparison of the full-VAE arms against fixed baselines.
Generalises docs/results/p1_19_hovernet_boundary/compare_p1_19.py to labelled arms.

Usage: python compare_stage2.py BASE NEW [NEW ...]
  each argument is LABEL=CSV[@COL=VALUE]: a per-crop CSV (score_outputs / score_p1_10_ddim_inversion /
  benchmark_vae_reconstruction), optionally filtered to rows where COL == VALUE (e.g.
  @source_mode=correct_alt_decoder, @vae=d1_pn03_best). Seed-bearing CSVs are cut to seed 0.
  BASE is the first argument; every NEW is compared to it on the crops both contain.
Delta = NEW - BASE. SSIM higher is better; LAB / wLAB lower is better.
Paired bootstrap 95% CI over crops (10,000 resamples)."""
import csv
import sys

import numpy as np


def load(spec):
    label, rest = spec.split("=", 1)
    path, filt = (rest.split("@", 1) + [None])[:2]
    col, val = filt.split("=", 1) if filt else (None, None)
    rows = {}
    for r in csv.DictReader(open(path, newline="")):
        if "seed" in r and r["seed"] not in ("0", ""):
            continue
        if col and r.get(col) != val:
            continue
        rows[(r["slide"], r["frame"], r["x"], r["y"])] = r
    if not rows:
        raise SystemExit(f"no rows for {spec}")
    return label, rows


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def report(base, new):
    (bl, a), (nl, b) = base, new
    keys = sorted(set(a) & set(b))
    print(f"\n=== {nl} vs {bl}: {len(keys)} paired crops ({bl} {len(a)}, {nl} {len(b)}) ===")
    scopes = {"ALL": keys, "excl_A06": [k for k in keys if k[0] != "A06"]}
    for s in sorted({k[0] for k in keys}):
        scopes[s] = [k for k in keys if k[0] == s]
    print(f"{'scope':8s} {'n':>4s} | {'metric':9s} {'base':>8s} {'new':>8s} {'delta':>8s} {'95% CI':>20s} {'new wins':>8s}")
    for scope, ks in scopes.items():
        for met in ("ssim", "wlab_mean", "lab_total"):
            ca = np.array([float(a[k][met]) for k in ks])
            hb = np.array([float(b[k][met]) for k in ks])
            mean, lo, hi = boot_ci(hb - ca)
            wins = (hb > ca).mean() if met == "ssim" else (hb < ca).mean()
            print(f"{scope:8s} {len(ks):4d} | {met:9s} {ca.mean():8.4f} {hb.mean():8.4f} {mean:+8.4f} "
                  f"[{lo:+8.4f}, {hi:+8.4f}] {100 * wins:7.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    base = load(sys.argv[1])
    for spec in sys.argv[2:]:
        report(base, load(spec))
