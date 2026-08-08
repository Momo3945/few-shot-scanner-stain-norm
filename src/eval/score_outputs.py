#!/usr/bin/env python3
"""
score_outputs.py -- score colour-LoRA img2img outputs against held-out ground truth.

Reads the eval_manifest.csv produced by infer_colour_lora.py (output crop paired with
its registered-Hamamatsu reference crop), computes LAB Wasserstein + SSIM/PSNR/MAE per
pair, and aggregates per denoising strength and per slide. Flags outlier slides
(robust median/MAD, e.g. A06) and, if given the RAW baseline summary, reports the
recovery delta -- how much colour gap each strength closed vs do-nothing.

This is CPU-only (no model), so it runs on a cheap partition after inference.

Usage
-----
    python score_outputs.py --eval-dir eval/a2h_r8 \
        --baseline pairs/baseline_metrics/baseline_summary.csv

Dependencies: opencv-python-headless, numpy, scipy, scikit-image.
Requires metrics.py (and its siblings) on the path.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import cv2

from metrics import score_aligned_pair, flag_outliers, _mean_finite


def read_rgb_png(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_baseline_lab(path):
    """Return the RAW baseline ALL lab_total, and per-slide (lab_total, n_crops)."""
    out = {"ALL": None, "per_slide": {}}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            scope = row["scope"]
            try:
                lab = float(row["lab_total"])
            except (ValueError, KeyError):
                continue
            if scope == "ALL":
                out["ALL"] = lab
            elif scope.startswith("A"):
                try:
                    n = int(row["n_crops"])
                except (ValueError, KeyError):
                    n = None
                out["per_slide"][scope] = (lab, n)
    return out


def baseline_all_for_slides(baseline, present_slides):
    """Crop-weighted baseline ALL restricted to the slides actually present in this
    run. Comparing a partial run's pooled ALL against the full-baseline ALL (which
    may include slides this run never touched) is an apples-to-oranges scope
    mismatch -- this is what recovery_delta_lab must be computed against instead.
    Falls back to the full baseline["ALL"] only when it can't be restricted (missing
    n_crops) or when the run's slide set already matches the full baseline exactly.
    """
    per_slide = baseline["per_slide"]
    if set(present_slides) >= set(per_slide):
        return baseline["ALL"]
    weighted, total_n = 0.0, 0
    for s in present_slides:
        if s not in per_slide or per_slide[s][1] is None:
            return None
        lab, n = per_slide[s]
        weighted += lab * n
        total_n += n
    return weighted / total_n if total_n else None


def main():
    ap = argparse.ArgumentParser(description="Score colour-LoRA outputs vs ground truth.")
    ap.add_argument("--eval-dir", required=True, help="Dir containing eval_manifest.csv (from inference).")
    ap.add_argument("--baseline", default=None, help="baseline_summary.csv for recovery delta (optional).")
    ap.add_argument("--outlier-z", type=float, default=3.5)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    man_path = eval_dir / "eval_manifest.csv"
    with open(man_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"empty manifest: {man_path}")

    # metrics[strength][slide] -> list of per-crop metric dicts
    per = defaultdict(lambda: defaultdict(list))
    per_crop_path = eval_dir / "eval_per_crop.csv"
    with open(per_crop_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strength", "slide", "frame", "x", "y",
                    "lab_total", "ssim", "psnr", "mae"])
        for r in rows:
            out_rgb = read_rgb_png(eval_dir / r["output_path"])
            ref_rgb = read_rgb_png(eval_dir / r["reference_path"])
            m = score_aligned_pair(out_rgb, ref_rgb)
            per[r["strength"]][r["slide"]].append(m)
            w.writerow([r["strength"], r["slide"], r["frame"], r["x"], r["y"],
                        round(m["lab_total"], 5), round(m["ssim"], 5),
                        round(m["psnr"], 5), round(m["mae"], 5)])

    baseline = read_baseline_lab(args.baseline) if args.baseline else None

    summary_path = eval_dir / "eval_summary.csv"
    keys = ("lab_total", "ssim", "psnr", "mae")
    print("\nColour-LoRA output vs registered real Hamamatsu (per denoising strength):")
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strength", "scope", "n_crops", "lab_total", "ssim", "psnr", "mae",
                    "robust_z", "outlier", "recovery_delta_lab"])
        for strength in sorted(per):
            slides = per[strength]
            slide_lab = {s: _mean_finite([m["lab_total"] for m in slides[s]]) for s in slides}
            flags = flag_outliers(slide_lab, z_thresh=args.outlier_z)
            outliers = {s for s in flags if flags[s]["outlier"]}

            def pool(exclude=frozenset()):
                acc = {k: [] for k in keys}
                for s in slides:
                    if s in exclude:
                        continue
                    for m in slides[s]:
                        for k in keys:
                            acc[k].append(m[k])
                return {k: _mean_finite(acc[k]) for k in keys}, \
                       sum(len(slides[s]) for s in slides if s not in exclude)

            allm, alln = pool()
            base_all = baseline_all_for_slides(baseline, slides) if baseline else None
            delta = (round(base_all - allm["lab_total"], 4)
                     if base_all is not None and allm["lab_total"] is not None else "")
            partial_note = "" if (baseline and set(slides) >= set(baseline["per_slide"])) \
                else f" [baseline restricted to {','.join(sorted(slides))}]"
            print(f"\n  strength {strength}:")
            print(f"    ALL ({len(slides)} slides, n={alln}): lab {allm['lab_total']}  "
                  f"ssim {allm['ssim']}  psnr {allm['psnr']}  mae {allm['mae']}"
                  + (f"   recovery Δlab {delta}{partial_note}" if delta != "" else ""))
            w.writerow([strength, "ALL", alln, allm["lab_total"], allm["ssim"],
                        allm["psnr"], allm["mae"], "", "", delta])

            if outliers:
                cleanm, cleann = pool(exclude=outliers)
                print(f"    excl {','.join(sorted(outliers))}: lab {cleanm['lab_total']}  "
                      f"ssim {cleanm['ssim']}  psnr {cleanm['psnr']}  mae {cleanm['mae']}")
                w.writerow([strength, "ALL_excl_outliers", cleann, cleanm["lab_total"],
                            cleanm["ssim"], cleanm["psnr"], cleanm["mae"], "", "", ""])

            for s in sorted(slides):
                sm = {k: _mean_finite([m[k] for m in slides[s]]) for k in keys}
                sdelta = ""
                if baseline and s in baseline["per_slide"] and sm["lab_total"] is not None:
                    sdelta = round(baseline["per_slide"][s][0] - sm["lab_total"], 4)
                flag = "*** OUTLIER" if flags[s]["outlier"] else ""
                print(f"      {s}: lab {sm['lab_total']:.2f}  ssim {sm['ssim']:.3f}  "
                      f"mae {sm['mae']:.2f}  z={flags[s]['z']:.2f}  {flag}"
                      + (f"  Δlab {sdelta}" if sdelta != "" else ""))
                w.writerow([strength, s, len(slides[s]), sm["lab_total"], sm["ssim"],
                            sm["psnr"], sm["mae"], round(flags[s]["z"], 3),
                            flags[s]["outlier"], sdelta])

    print(f"\nPer-crop : {per_crop_path}")
    print(f"Summary  : {summary_path}")
    if baseline:
        print("Positive recovery Δlab = colour gap closed vs the raw baseline (H2).")


if __name__ == "__main__":
    main()
