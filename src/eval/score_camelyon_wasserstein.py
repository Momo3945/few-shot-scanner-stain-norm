#!/usr/bin/env python3
"""
score_camelyon_wasserstein.py -- P2-10: CAMELYON17 multi-centre generalisation.
Pairwise LAB-Wasserstein between all 5 centres, pooled per centre (all patients'
patches for a centre concatenated into one pixel distribution), before (D_pre)
and after (D_post) normalisation. Success = D_post < D_pre (proposal's own
criterion -- no fixed numeric threshold like H3's 0.95).

Verified directly (2026-08-18, before writing this script) against real
already-uploaded CAMELYON17 patches: metrics.py's lab_wasserstein() calls
cv2.cvtColor() internally, which requires a genuine HxWx3 image array -- a 4D
stack of shape (N, H, W, 3) raises "Bad number of channels: scn=1", not a
silent misread. Pooling must concatenate patches along axis 0 into one tall
(N*H, W, 3) "image" instead -- confirmed this produces a near-zero self-distance
(~0.25, residual from lab_wasserstein's internal random subsampling drawing
different indices for its two arguments even when they're the same array) and a
real, large cross-centre distance (~50) on two real patch sets.

Directory convention (matches camelyon_patches.py's extraction output):
    <root>/centre_<0-4>_patient_<id>/*.png

Usage
-----
    # D_pre (raw patches, no dependency on normalize_camelyon.py):
    python score_camelyon_wasserstein.py --root camelyon17_patches --out eval/camelyon17_patches

    # D_post (after normalisation), compared against the D_pre run's pairwise.csv:
    python score_camelyon_wasserstein.py --root eval/camelyon17_normalised --out eval/camelyon17_normalised \
        --against eval/camelyon17_patches/pairwise.csv

Dependencies: numpy, opencv-python-headless, scipy, Pillow.
Requires metrics.py on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
from PIL import Image

from metrics import lab_wasserstein


def parse_args():
    ap = argparse.ArgumentParser(
        description="Pairwise LAB-Wasserstein between CAMELYON17 centres, pooled per centre (P2-10).")
    ap.add_argument("--root", required=True,
                    help="Dir containing centre_<0-4>_patient_<id>/*.png (raw or normalised).")
    ap.add_argument("--out", required=True, help="Output dir for pairwise.csv / summary.csv.")
    ap.add_argument("--against", default=None,
                    help="A prior run's pairwise.csv (typically the raw/D_pre run) to diff against, "
                         "computing per-pair improvement and the overall D_post < D_pre verdict.")
    return ap.parse_args()


def load_centre_pixels(root: Path) -> dict:
    """Pool every patch for each centre into one tall (N*H, W, 3) array.

    Concatenating along axis 0 (NOT np.stack -- see module docstring) keeps the
    array a valid 2D+channel image cv2.cvtColor can consume, while
    lab_wasserstein's own reshape(-1, 3) still flattens it to one pixel
    distribution regardless of how tall it is.
    """
    by_centre: dict = {}
    patient_dirs = sorted(root.glob("centre_*_patient_*"))
    if not patient_dirs:
        raise SystemExit(f"No centre_*_patient_* directories found under {root}")
    for pdir in patient_dirs:
        centre = int(pdir.name.split("_")[1])
        paths = sorted(pdir.glob("*.png"))
        if not paths:
            print(f"  WARNING: no patches in {pdir}")
            continue
        imgs = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
        by_centre.setdefault(centre, []).extend(imgs)
        print(f"  {pdir.name}: {len(imgs)} patches")

    pooled = {}
    for centre, imgs in sorted(by_centre.items()):
        pooled[centre] = np.concatenate(imgs, axis=0)
        print(f"centre_{centre}: {len(imgs)} patches pooled, {pooled[centre].shape[0]} px rows")
    return pooled


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled = load_centre_pixels(root)
    centres = sorted(pooled)
    if len(centres) < 2:
        raise SystemExit(f"Need at least 2 centres to compute pairwise distances, found {len(centres)}")

    pairwise_path = out_dir / "pairwise.csv"
    rows = []
    with open(pairwise_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["centre_i", "centre_j", "lab_L", "lab_a", "lab_b", "lab_total"])
        for ci, cj in itertools.combinations(centres, 2):
            r = lab_wasserstein(pooled[ci], pooled[cj])
            w.writerow([ci, cj, round(r["L"], 5), round(r["a"], 5), round(r["b"], 5), round(r["total"], 5)])
            rows.append({"centre_i": ci, "centre_j": cj, "lab_total": r["total"]})
            print(f"  centre_{ci} vs centre_{cj}: lab_total={r['total']:.4f}")

    mean_lab = float(np.mean([r["lab_total"] for r in rows]))
    summary_path = out_dir / "summary.csv"
    summary_rows = [["mean_lab_total", round(mean_lab, 5), len(rows)]]

    print(f"\n{root.name}: mean pairwise LAB Wasserstein = {mean_lab:.4f} (n_pairs={len(rows)})")
    print(f"Pairwise: {pairwise_path}")

    if args.against:
        against_path = Path(args.against)
        with open(against_path, newline="") as fh:
            base_rows = {(int(r["centre_i"]), int(r["centre_j"])): float(r["lab_total"])
                        for r in csv.DictReader(fh)}
        if len(base_rows) != len(rows):
            print(f"  WARNING: --against has {len(base_rows)} pairs, this run has {len(rows)} -- "
                  f"comparison may be incomplete.")
        n_improved = n_compared = 0
        with open(out_dir / "d_pre_post_comparison.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["centre_i", "centre_j", "d_pre", "d_post", "improved"])
            for r in rows:
                key = (r["centre_i"], r["centre_j"])
                if key not in base_rows:
                    print(f"  WARNING: no D_pre row for pair {key} in {against_path} -- skipping.")
                    continue
                d_pre = base_rows[key]
                d_post = r["lab_total"]
                improved = d_post < d_pre
                n_improved += int(improved)
                n_compared += 1
                w.writerow([r["centre_i"], r["centre_j"], round(d_pre, 5), round(d_post, 5), improved])

        base_mean = float(np.mean(list(base_rows.values())))
        verdict = "PASS (D_post_mean < D_pre_mean)" if mean_lab < base_mean else "FAIL (D_post_mean >= D_pre_mean)"
        print(f"\nD_pre_mean={base_mean:.4f}  D_post_mean={mean_lab:.4f}  "
              f"{n_improved}/{n_compared} pairs improved  {verdict}")
        summary_rows.append(["d_pre_mean", round(base_mean, 5), len(base_rows)])
        summary_rows.append(["n_pairs_improved", n_improved, n_compared])
        summary_rows.append(["verdict", verdict, ""])

    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value", "n"])
        w.writerows(summary_rows)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
