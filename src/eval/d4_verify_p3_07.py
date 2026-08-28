#!/usr/bin/env python3
"""
d4_verify_p3_07.py -- P3-07 diagnostic D4: controlnet_conditioning_scale
sweep comparison, on the internal validation split (A03/H03 pairs the
training run held out for val -- never the A06/A08/A09/A13/A16 held-out
test set).

Computes the raw ("do nothing") baseline directly from the 11 val pairs
(coordinate-corresponding, NOT pixel-exact -- these are training-domain
pairs, so lab_total/de2000 are valid (distributional/pixel-error metrics
that don't require registration per metrics.py), but SSIM here is noisier
than the registered held-out numbers elsewhere in this project; report it
for relative comparison ACROSS scales only, not as an absolute structural
figure comparable to the held-out tables). Then reads each scale's scored
per_crop.csv and reports recovery Δlab per scale, identifying which scale
(if any) restores non-negative recovery.

Usage
-----
    python d4_verify_p3_07.py \
        --pairs-dir pairs/train_1024 \
        --val-frames-json lora/a2h_cond_r8_sdxl_1024/pair_manifest.json \
        --scale-summaries 0.25=eval/p3_07_d4_scale_0.25_summary/per_crop.csv \
                          0.50=eval/p3_07_d4_scale_0.50_summary/per_crop.csv \
                          0.75=eval/p3_07_d4_scale_0.75_summary/per_crop.csv \
                          1.00=eval/p3_07_d4_scale_1.00_summary/per_crop.csv \
        --out eval/p3_07_d4_summary/d4_verdict.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from train_colour_translation_lora_sdxl import build_pairs  # noqa: E402

from metrics import score_aligned_pair  # noqa: E402
from score_outputs import read_rgb_png  # noqa: E402

METRIC_KEYS = ("lab_total", "de2000_mean", "ssim")


def parse_args():
    ap = argparse.ArgumentParser(description="P3-07 D4: controlnet-conditioning-scale sweep.")
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--val-frames-json", required=True)
    ap.add_argument("--scale-summaries", nargs="+", required=True,
                    help="SCALE=path/to/per_crop.csv, one per swept scale.")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    scale_paths = {}
    for entry in args.scale_summaries:
        scale, path = entry.split("=", 1)
        scale_paths[scale] = path

    with open(args.val_frames_json) as fh:
        val_frames = set(json.load(fh)["val_frames"])
    pairs = [p for p in build_pairs(args.pairs_dir) if p["frame_id"] in val_frames]
    if not pairs:
        raise SystemExit(f"No pairs matched val_frames {sorted(val_frames)} in {args.pairs_dir}")

    print(f"Internal validation baseline: {len(pairs)} pairs, frames {sorted(val_frames)}")
    raw_metrics = {k: [] for k in METRIC_KEYS}
    for p in pairs:
        a_rgb = read_rgb_png(p["aperio_path"])
        h_rgb = read_rgb_png(p["hamamatsu_path"])
        m = score_aligned_pair(a_rgb, h_rgb)
        for k in METRIC_KEYS:
            raw_metrics[k].append(m[k])
    raw_summary = {k: statistics.mean(v) for k, v in raw_metrics.items()}
    print(f"  raw (do-nothing): lab_total={raw_summary['lab_total']:.4f}  "
          f"de2000_mean={raw_summary['de2000_mean']:.4f}  ssim={raw_summary['ssim']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []

    print(f"\n{'=' * 90}\nD4 RESULT -- controlnet_conditioning_scale sweep (internal validation)\n{'=' * 90}")
    print(f"{'scale':8}{'n':>5}{'model_lab':>12}{'recovery':>11}{'ssim':>9}{'de2000':>9}")
    print(f"{'raw':8}{len(pairs):>5}{raw_summary['lab_total']:>12.4f}{'--':>11}"
          f"{raw_summary['ssim']:>9.4f}{raw_summary['de2000_mean']:>9.4f}")
    rows_out.append(["raw", len(pairs), raw_summary["lab_total"], "", raw_summary["ssim"],
                     raw_summary["de2000_mean"]])

    best_scale, best_recovery = None, float("-inf")
    for scale in sorted(scale_paths, key=float):
        path = scale_paths[scale]
        vals = {k: [] for k in METRIC_KEYS}
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["source_mode"] != "correct":
                    continue
                for k in METRIC_KEYS:
                    vals[k].append(float(r[k]))
        if not vals["lab_total"]:
            print(f"{scale:8}  WARNING: no source_mode=correct rows in {path}")
            continue
        n = len(vals["lab_total"])
        model_lab = statistics.mean(vals["lab_total"])
        ssim = statistics.mean(vals["ssim"])
        de2000 = statistics.mean(vals["de2000_mean"])
        recovery = raw_summary["lab_total"] - model_lab
        print(f"{scale:8}{n:>5}{model_lab:>12.4f}{recovery:>11.4f}{ssim:>9.4f}{de2000:>9.4f}")
        rows_out.append([scale, n, model_lab, recovery, ssim, de2000])
        if recovery > best_recovery:
            best_scale, best_recovery = scale, recovery

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scale", "n", "model_lab", "recovery_delta_lab", "ssim", "de2000_mean"])
        w.writerows(rows_out)

    print(f"\n{'=' * 90}\nInterpretation\n{'=' * 90}")
    if best_scale is not None:
        print(f"Best recovery: scale={best_scale} (Δlab={best_recovery:.4f})")
        if best_recovery >= 0:
            print(">>> A scale in this grid restores non-negative recovery on internal validation. "
                  "Worth a targeted held-out check at this scale (still not the full held-out set -- "
                  "confirm on a small diagnostic before declaring victory).")
        else:
            print(">>> No scale in {0.25, 0.50, 0.75, 1.00} restores non-negative recovery on "
                  "internal validation, even at the best point. Reducing source-conditioning "
                  "influence alone does not fix this at 1024 -- the negative colour drift appears "
                  "structural to this checkpoint/operating point, not just an over-strong "
                  "ControlNet scale. Consider H3 (harder 1024 colour-learning problem under the "
                  "same <=50-pair budget) or accepting the SSIM/colour trade-off as documented.")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
