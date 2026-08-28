#!/usr/bin/env python3
"""
d3_verify_p3_07.py -- P3-07 diagnostic D3: RGB vs Canny source-conditioning
comparison.

Reads the three per_crop.csv files scored by score_p1_10_ablation.py for
d3_infer_condition_split_p3_07.slurm's rgb_canny / rgb_only / canny_only
runs (same A06+A08 held-out subset, same operating point, trained colour
LoRA enabled throughout -- only which half of the 6-channel source
condition reaches the ControlNet differs), plus d1_verify_p3_07.py's
d1_summary.csv (1024-native raw baseline per slide). Prints, per condition
mode / per slide / pooled: SSIM, LAB, recovery Δlab, ΔE2000, and states the
interpretation this ticket's D3 section is looking for:

    RGB-containing conditions (rgb_canny, rgb_only): higher SSIM, more
    negative recovery -> source-RGB conditioning is preserving Aperio
    scanner appearance alongside morphology, at the cost of colour recovery.

    canny_only: lower SSIM (less structural guidance), but recovery closer
    to zero or positive -> removing the RGB half removes the appearance-
    preservation pressure.

If that pattern does not hold, the script says so explicitly rather than
forcing the H1 interpretation.

Usage
-----
    python d3_verify_p3_07.py \
        --rgb-canny-per-crop eval/p3_07_d3_rgb_canny_summary/per_crop.csv \
        --rgb-only-per-crop eval/p3_07_d3_rgb_only_summary/per_crop.csv \
        --canny-only-per-crop eval/p3_07_d3_canny_only_summary/per_crop.csv \
        --d1-summary eval/p3_07_d1_verify/d1_summary.csv \
        --out eval/p3_07_d3_summary/d3_verdict.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

MODES = ("rgb_canny", "rgb_only", "canny_only")
METRIC_KEYS = ("lab_total", "de2000_mean", "ssim")


def parse_args():
    ap = argparse.ArgumentParser(description="P3-07 D3: RGB vs Canny conditioning comparison.")
    for m in MODES:
        ap.add_argument(f"--{m.replace('_', '-')}-per-crop", required=True,
                        help=f"per_crop.csv for the {m} condition-mode run.")
    ap.add_argument("--d1-summary", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def load_per_crop(path):
    by_slide = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["source_mode"] != "correct":
                continue
            for k in METRIC_KEYS:
                by_slide[r["slide"]][k].append(float(r[k]))
    return by_slide


def main():
    args = parse_args()
    per_crop_paths = {m: getattr(args, f"{m}_per_crop") for m in MODES}

    with open(args.d1_summary, newline="") as fh:
        d1 = {r["scope"]: r for r in csv.DictReader(fh)}

    data = {m: load_per_crop(p) for m, p in per_crop_paths.items()}
    slides = sorted({s for m in MODES for s in data[m]})
    if not slides:
        raise SystemExit("No source_mode=correct rows found in any per-crop file.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []

    print("=" * 100)
    print("D3 RESULT -- RGB vs Canny source-conditioning split (A06+A08, trained colour LoRA enabled)")
    print("=" * 100)
    print(f"{'mode':11}{'slide':7}{'n':>5}{'raw_1024':>11}{'model_lab':>12}{'recovery':>11}"
          f"{'ssim':>9}{'de2000':>9}")

    pooled = {m: defaultdict(list) for m in MODES}
    for slide in slides:
        raw = float(d1[slide]["raw_1024_lab"]) if slide in d1 else None
        for m in MODES:
            vals = data[m].get(slide)
            if not vals or raw is None:
                print(f"{m:11}{slide:7}{'--':>5}  (missing)")
                continue
            n = len(vals["lab_total"])
            model_lab = statistics.mean(vals["lab_total"])
            ssim = statistics.mean(vals["ssim"])
            de2000 = statistics.mean(vals["de2000_mean"])
            recovery = raw - model_lab
            print(f"{m:11}{slide:7}{n:>5}{raw:>11.4f}{model_lab:>12.4f}{recovery:>11.4f}"
                  f"{ssim:>9.4f}{de2000:>9.4f}")
            rows_out.append([m, slide, n, raw, model_lab, recovery, ssim, de2000])
            pooled[m]["raw"].extend([raw] * n)
            pooled[m]["lab"].extend(vals["lab_total"])
            pooled[m]["ssim"].extend(vals["ssim"])
            pooled[m]["de2000"].extend(vals["de2000_mean"])

    print("-" * 100)
    pooled_summary = {}
    for m in MODES:
        if not pooled[m]["lab"]:
            continue
        raw_m = statistics.mean(pooled[m]["raw"])
        model_m = statistics.mean(pooled[m]["lab"])
        recov_m = raw_m - model_m
        ssim_m = statistics.mean(pooled[m]["ssim"])
        de_m = statistics.mean(pooled[m]["de2000"])
        n = len(pooled[m]["lab"])
        pooled_summary[m] = {"recovery": recov_m, "ssim": ssim_m, "de2000": de_m}
        print(f"{m:11}{'POOLED':7}{n:>5}{raw_m:>11.4f}{model_m:>12.4f}{recov_m:>11.4f}"
              f"{ssim_m:>9.4f}{de_m:>9.4f}")
        rows_out.append([m, "POOLED", n, raw_m, model_m, recov_m, ssim_m, de_m])

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition_mode", "slide", "n", "raw_1024_lab", "model_lab",
                    "recovery_delta_lab", "ssim", "de2000_mean"])
        w.writerows(rows_out)

    print(f"\n{'=' * 100}\nInterpretation\n{'=' * 100}")
    if all(m in pooled_summary for m in MODES):
        rgb_canny, rgb_only, canny_only = (pooled_summary[m] for m in MODES)
        rgb_conditions_higher_ssim = (rgb_canny["ssim"] > canny_only["ssim"] and
                                      rgb_only["ssim"] > canny_only["ssim"])
        rgb_conditions_more_negative = (rgb_canny["recovery"] < canny_only["recovery"] and
                                        rgb_only["recovery"] < canny_only["recovery"])
        print(f"rgb_canny : SSIM {rgb_canny['ssim']:.4f}  recovery {rgb_canny['recovery']:.4f}")
        print(f"rgb_only  : SSIM {rgb_only['ssim']:.4f}  recovery {rgb_only['recovery']:.4f}")
        print(f"canny_only: SSIM {canny_only['ssim']:.4f}  recovery {canny_only['recovery']:.4f}")
        if rgb_conditions_higher_ssim and rgb_conditions_more_negative:
            print("\n>>> H1 SUPPORTED: RGB-containing conditions (rgb_canny, rgb_only) score higher "
                  "SSIM AND more negative recovery than canny_only. At native resolution, explicit "
                  "source-RGB conditioning appears to preserve source-scanner (Aperio) appearance "
                  "as well as morphology, creating a stronger structure-colour conflict than "
                  "edge/morphology conditioning alone.")
        elif canny_only["recovery"] >= 0 and (rgb_canny["recovery"] < 0 or rgb_only["recovery"] < 0):
            print("\n>>> PARTIAL SUPPORT: canny_only recovers non-negative colour while at least one "
                  "RGB-containing condition is still negative, but the SSIM pattern doesn't cleanly "
                  "match H1's full prediction -- report both metrics, don't force the clean story.")
        else:
            print("\n>>> H1 NOT SUPPORTED as stated: the RGB-containing-conditions-preserve-colour "
                  "pattern does not hold cleanly in this data. Report the raw numbers above; "
                  "reconsider H1 relative to H2 (colour LoRA too weak generally, independent of "
                  "which conditioning channel) and H3/H4.")
    else:
        missing = [m for m in MODES if m not in pooled_summary]
        print(f"Cannot form the full comparison -- missing pooled data for: {missing}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
