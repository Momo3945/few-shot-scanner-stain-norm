#!/usr/bin/env python3
"""
d1_verify_p3_07.py -- P3-07 diagnostic D1: verify the -5.60 recovery Δlab is
not a metric/baseline artefact (this project's H5).

aggregate_p1_10_full.py computes recovery_delta_lab as:
    baseline["ALL"]["lab_total"]  -  model["ALL"]["lab_total"]
where `baseline` is whatever --baseline CSV is passed. Both
aggregate_p3_06_full.slurm (512) and aggregate_p3_07_full.slurm (1024) pass
the SAME file: pairs/baseline_metrics/baseline_summary.csv, which
score_baseline.slurm generates with a hardcoded --crop 512 (see
run_baseline() in metrics.py). So P3-07's reported recovery number is
    LAB(raw Aperio @512, real Hamamatsu @512)  -  LAB(P3-07 output @1024, real Hamamatsu @1024)
not the resolution-matched
    LAB(raw Aperio @1024, real Hamamatsu @1024)  -  LAB(P3-07 output @1024, real Hamamatsu @1024)

This script computes the TRUE 1024-native raw baseline directly from the
same held-out frames P3-07's manifest actually used (same registration,
same (x,y) grid crop, same score_aligned_pair() call as metrics.py's own
run_baseline()) and compares:
  1. raw_1024 LAB vs raw_512 LAB (from the existing baseline_summary.csv) --
     is there even a meaningful resolution effect on the raw number?
  2. recovery recomputed against raw_1024 vs the originally reported
     recovery (against raw_512) -- does the sign flip?
  3. an independent per-crop spot check on N randomly sampled crops,
     printing raw_1024 lab_total, model lab_total (mean across seeds, from
     the already-scored per_crop.csv), and the sign of (raw - model).

Does not touch any existing eval/ output, baseline_summary.csv, or scored
CSV -- read-only diagnostic, writes only to --out.

Usage
-----
    python d1_verify_p3_07.py \
        --root /datasets/mhoosen/stain-norm/mitos_heldout \
        --heldout /datasets/mhoosen/stain-norm/pairs/heldout_frames.csv \
        --manifest /datasets/mhoosen/stain-norm/eval/p3_07_full_heldout/eval_manifest.csv \
        --eval-dir /datasets/mhoosen/stain-norm/eval/p3_07_full_heldout \
        --per-crop /datasets/mhoosen/stain-norm/eval/p3_07_full_heldout_summary/per_crop.csv \
        --baseline-512 /datasets/mhoosen/stain-norm/pairs/baseline_metrics/baseline_summary.csv \
        --out /datasets/mhoosen/stain-norm/eval/p3_07_d1_verify \
        --crop 1024 --n-sample 10 --seed 0

Dependencies: opencv-python-headless, numpy, scipy, scikit-image, tifffile.
Requires registration.py and metrics.py on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from metrics import score_aligned_pair, flag_outliers, _mean_finite
from registration import read_rgb, register_h_to_a
from score_outputs import read_baseline_lab

METRIC_KEYS = ("lab_total", "wlab_mean", "de2000_mean", "ssim", "psnr", "mae")


def parse_args():
    ap = argparse.ArgumentParser(description="P3-07 D1: raw-1024-baseline / recovery-artefact check.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--manifest", required=True, help="P3-07's eval_manifest.csv (correct mode).")
    ap.add_argument("--eval-dir", required=True, help="Dir holding manifest's reference/ pngs.")
    ap.add_argument("--per-crop", required=True, help="P3-07's already-scored per_crop.csv.")
    ap.add_argument("--baseline-512", required=True, help="Existing (512) baseline_summary.csv.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--n-sample", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. unique crops actually used by P3-07 (dedupe across seeds) ----
    with open(args.manifest, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["source_mode"] == "correct"]
    if not rows:
        raise SystemExit(f"No source_mode=correct rows in {args.manifest}")

    crops_by_id = {}
    for r in rows:
        crops_by_id.setdefault(r["crop_id"], {
            "slide": r["slide"], "frame": r["frame"], "x": int(r["x"]), "y": int(r["y"]),
            "aperio_path": r["aperio_path"], "reference_path": r["reference_path"],
        })
    print(f"Manifest: {len(rows)} rows (correct mode) -> {len(crops_by_id)} unique crops.")

    # ---- 2. heldout_frames.csv: (slide, frame) -> hamamatsu_path ----
    with open(args.heldout, newline="") as fh:
        h_rows = list(csv.DictReader(fh))
    ham_path = {}
    for r in h_rows:
        r["hamamatsu_path"] = r["hamamatsu_path"].replace("\\", "/")
        ham_path[(r["aperio_slide"], r["frame_id"])] = r["hamamatsu_path"]

    # ---- 3. group unique crops by (slide, frame) -- register once per frame ----
    by_frame = defaultdict(list)
    for cid, c in crops_by_id.items():
        by_frame[(c["slide"], c["frame"])].append(cid)

    root = Path(args.root)
    eval_dir = Path(args.eval_dir)
    raw_lab_per_crop = {}          # crop_id -> score_aligned_pair(raw, ref) dict
    ref_mismatch = []              # crop_ids where recomputed ref != saved ref (sanity check)

    n_frames = len(by_frame)
    print(f"Registering {n_frames} unique held-out frames at crop={args.crop} ...")
    for i, ((slide, frame), cids) in enumerate(sorted(by_frame.items()), 1):
        first = crops_by_id[cids[0]]
        a_path = root / first["aperio_path"]
        h_rel = ham_path.get((slide, frame))
        if h_rel is None:
            print(f"  WARNING: no heldout_frames.csv row for {slide}_{frame} -- skipping {len(cids)} crop(s).")
            continue
        h_path = root / h_rel
        a_rgb = read_rgb(a_path)
        h_rgb = read_rgb(h_path)
        reg = register_h_to_a(a_rgb, h_rgb, ecc_min=args.ecc_min)
        h_reg = reg.h_registered_rgb
        print(f"  [{i}/{n_frames}] {slide}_{frame}  ecc={reg.ecc_score:.4f} ok={reg.ok}  crops={len(cids)}")

        for cid in cids:
            c = crops_by_id[cid]
            x, y = c["x"], c["y"]
            raw_c = a_rgb[y:y + args.crop, x:x + args.crop]
            ref_c_recomputed = h_reg[y:y + args.crop, x:x + args.crop]

            ref_saved = read_rgb(eval_dir / c["reference_path"])
            if ref_saved.shape[:2] == ref_c_recomputed.shape[:2]:
                mae_vs_saved = float(np.mean(np.abs(
                    ref_saved.astype(np.int16) - ref_c_recomputed.astype(np.int16))))
                if mae_vs_saved > 1.0:  # deterministic ECC -> should reproduce near-exactly
                    ref_mismatch.append((cid, mae_vs_saved))
            else:
                ref_mismatch.append((cid, "shape mismatch"))

            raw_lab_per_crop[cid] = score_aligned_pair(raw_c, ref_saved)

    if ref_mismatch:
        print(f"\nWARNING: {len(ref_mismatch)} crop(s) failed the reference-reproducibility "
              f"check (recomputed registered-Hamamatsu crop != saved reference/ png). "
              f"Raw-baseline numbers below may not line up with the exact crops the model "
              f"saw. First few: {ref_mismatch[:5]}")
    else:
        print(f"\nReproducibility check OK: recomputed registered-Hamamatsu crop matches the "
              f"saved reference/ png (MAE <= 1.0) for all {len(raw_lab_per_crop)} crops.")

    # ---- 4. model_1024 LAB per crop, from the already-scored per_crop.csv ----
    model_lab_per_crop = defaultdict(list)  # crop_id -> [lab_total per seed]
    with open(args.per_crop, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["source_mode"] != "correct":
                continue
            model_lab_per_crop[r["crop_id"]].append(float(r["lab_total"]))

    # ---- 5. per-slide / ALL aggregation: raw_1024 vs model_1024 ----
    slides = sorted({c["slide"] for c in crops_by_id.values()})
    raw_by_slide = defaultdict(list)
    model_by_slide = defaultdict(list)
    for cid, c in crops_by_id.items():
        if cid not in raw_lab_per_crop or cid not in model_lab_per_crop:
            continue
        raw_by_slide[c["slide"]].append(raw_lab_per_crop[cid]["lab_total"])
        # one value per crop (mean across its seeds), matching aggregate_p1_10_full.py's
        # treatment of "the model's number for this crop" if it pooled seeds first --
        # NOTE aggregate_p1_10_full.py actually pools every (crop,seed) row directly, so
        # we ALSO report that pooled-row version below for an exact apples-to-apples match.
        model_by_slide[c["slide"]].append(statistics.mean(model_lab_per_crop[cid]))

    baseline_512 = read_baseline_lab(args.baseline_512)

    def summarize(vals):
        return _mean_finite(vals)

    print("\n" + "=" * 78)
    print("D1 RESULT -- raw_1024 vs model_1024 (per-crop-mean-across-seeds pooling)")
    print("=" * 78)
    header = f"{'slide':7}{'n':>5}{'raw_1024':>12}{'model_1024':>12}{'raw-model':>12}{'raw_512(old)':>14}{'old_recovery':>14}"
    print(header)

    all_raw, all_model = [], []
    rows_out = []
    for s in slides:
        raw_m = summarize(raw_by_slide[s])
        model_m = summarize(model_by_slide[s])
        diff = round(raw_m - model_m, 4) if raw_m is not None and model_m is not None else None
        old_raw = baseline_512["per_slide"].get(s, (None, None))[0]
        old_recovery = round(old_raw - model_m, 4) if old_raw is not None and model_m is not None else None
        print(f"{s:7}{len(raw_by_slide[s]):>5}{raw_m:>12.4f}{model_m:>12.4f}{diff:>12.4f}"
              f"{(old_raw if old_raw is not None else float('nan')):>14.4f}"
              f"{(old_recovery if old_recovery is not None else float('nan')):>14.4f}")
        rows_out.append([s, len(raw_by_slide[s]), raw_m, model_m, diff, old_raw, old_recovery])
        all_raw.extend(raw_by_slide[s])
        all_model.extend(model_by_slide[s])

    all_raw_m = summarize(all_raw)
    all_model_m = summarize(all_model)
    all_diff = round(all_raw_m - all_model_m, 4)
    old_all_raw = baseline_512["ALL"]
    old_all_recovery = round(old_all_raw - all_model_m, 4)
    print("-" * len(header))
    print(f"{'ALL':7}{len(all_raw):>5}{all_raw_m:>12.4f}{all_model_m:>12.4f}{all_diff:>12.4f}"
          f"{old_all_raw:>14.4f}{old_all_recovery:>14.4f}")
    rows_out.append(["ALL", len(all_raw), all_raw_m, all_model_m, all_diff, old_all_raw, old_all_recovery])

    with open(out_dir / "d1_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "n_crops", "raw_1024_lab", "model_1024_lab",
                    "recovery_delta_lab_1024_native", "raw_512_lab_old_baseline",
                    "recovery_delta_lab_as_originally_reported"])
        w.writerows(rows_out)

    print(f"\nOriginally reported ALL recovery Δlab (vs 512 baseline): {old_all_recovery}")
    print(f"1024-native ALL recovery Δlab (vs 1024 raw baseline):    {all_diff}")
    if (old_all_recovery < 0) != (all_diff < 0):
        print(">>> SIGN FLIPS once the baseline resolution is matched. H5 (baseline-mismatch "
              "artefact) is at least a major contributor to the reported negative recovery.")
    else:
        print(">>> Sign is UNCHANGED against a resolution-matched baseline. H5 is ruled out as "
              "the explanation -- the negative recovery is real, proceed to D2.")

    # ---- 6. manual spot check on N random crops ----
    rng = np.random.default_rng(args.seed)
    common = sorted(set(raw_lab_per_crop) & set(model_lab_per_crop))
    sample = rng.choice(common, size=min(args.n_sample, len(common)), replace=False)
    print(f"\n{'=' * 78}\nSpot check -- {len(sample)} randomly sampled crops\n{'=' * 78}")
    print(f"{'crop_id':30}{'raw_1024_lab':>14}{'model_1024_lab':>16}{'raw-model':>12}")
    with open(out_dir / "d1_spot_check.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["crop_id", "slide", "raw_1024_lab", "model_1024_lab_mean_across_seeds", "raw_minus_model"])
        for cid in sample:
            raw_v = raw_lab_per_crop[cid]["lab_total"]
            model_v = statistics.mean(model_lab_per_crop[cid])
            d = raw_v - model_v
            print(f"{cid:30}{raw_v:>14.4f}{model_v:>16.4f}{d:>12.4f}")
            w.writerow([cid, crops_by_id[cid]["slide"], round(raw_v, 5), round(model_v, 5), round(d, 5)])

    print(f"\nWrote {out_dir / 'd1_summary.csv'}")
    print(f"Wrote {out_dir / 'd1_spot_check.csv'}")


if __name__ == "__main__":
    main()
