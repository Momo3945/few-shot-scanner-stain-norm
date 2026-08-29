#!/usr/bin/env python3
"""
d5_verify_p3_07.py -- P3-07 diagnostic D5: colour-LoRA-scale sweep
comparison, on the internal validation split (same 11 A03/H03 pairs D4
used -- never the A06/A08/A09/A13/A16 held-out test set).

Computes the raw ("do nothing") baseline directly from the val pairs
(same method as d4_verify_p3_07.py), then reads each lora_scale's scored
per_crop.csv and reports SSIM / LAB total / windowed LAB / ΔE2000 /
recovery Δlab per scale. Applies this ticket's selection rule: first
require recovery Δlab > 0; among positive-recovery configurations, select
the highest-SSIM one. If no configuration is positive, says so explicitly
and recommends stopping inference-only tuning (per-instruction: propose the
bounded P3-07b >50-pair training follow-up instead, not more inference
grids). Also reports whether the trend across scales moved consistently
toward zero/positive, which is the trigger for the second (strength x
lora_scale) grid.

Usage
-----
    python d5_verify_p3_07.py \
        --pairs-dir pairs/train_1024 \
        --val-frames-json lora/a2h_cond_r8_sdxl_1024/pair_manifest.json \
        --scale-summaries 1.0=eval/p3_07_d5_lora_1.0_summary/per_crop.csv \
                          1.25=eval/p3_07_d5_lora_1.25_summary/per_crop.csv \
                          1.5=eval/p3_07_d5_lora_1.5_summary/per_crop.csv \
                          2.0=eval/p3_07_d5_lora_2.0_summary/per_crop.csv \
        --out eval/p3_07_d5_summary/d5_verdict.csv
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

METRIC_KEYS = ("lab_total", "wlab_mean", "de2000_mean", "ssim")


def parse_args():
    ap = argparse.ArgumentParser(description="P3-07 D5: colour-LoRA-scale sweep.")
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--val-frames-json", required=True)
    ap.add_argument("--scale-summaries", nargs="+", required=True,
                    help="LORA_SCALE=path/to/per_crop.csv, one per swept scale.")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def compute_raw_baseline(pairs_dir, val_frames_json):
    with open(val_frames_json) as fh:
        val_frames = set(json.load(fh)["val_frames"])
    pairs = [p for p in build_pairs(pairs_dir) if p["frame_id"] in val_frames]
    if not pairs:
        raise SystemExit(f"No pairs matched val_frames {sorted(val_frames)} in {pairs_dir}")
    acc = {k: [] for k in METRIC_KEYS}
    for p in pairs:
        a_rgb = read_rgb_png(p["aperio_path"])
        h_rgb = read_rgb_png(p["hamamatsu_path"])
        m = score_aligned_pair(a_rgb, h_rgb)
        for k in METRIC_KEYS:
            acc[k].append(m[k])
    return {k: statistics.mean(v) for k, v in acc.items()}, len(pairs), sorted(val_frames)


def main():
    args = parse_args()
    scale_paths = {}
    for entry in args.scale_summaries:
        scale, path = entry.split("=", 1)
        scale_paths[scale] = path

    raw_summary, n_pairs, val_frames = compute_raw_baseline(args.pairs_dir, args.val_frames_json)
    print(f"Internal validation baseline: {n_pairs} pairs, frames {val_frames}")
    print(f"  raw (do-nothing): lab_total={raw_summary['lab_total']:.4f}  "
          f"wlab_mean={raw_summary['wlab_mean']:.4f}  de2000_mean={raw_summary['de2000_mean']:.4f}  "
          f"ssim={raw_summary['ssim']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []
    results = {}  # scale (float) -> dict of metrics + recovery

    print(f"\n{'=' * 100}\nD5 RESULT -- colour-LoRA-scale sweep (internal validation)\n{'=' * 100}")
    print(f"{'lora_scale':11}{'n':>5}{'model_lab':>12}{'recovery':>11}{'ssim':>9}"
          f"{'wlab':>9}{'de2000':>9}")
    print(f"{'raw':11}{n_pairs:>5}{raw_summary['lab_total']:>12.4f}{'--':>11}"
          f"{raw_summary['ssim']:>9.4f}{raw_summary['wlab_mean']:>9.4f}{raw_summary['de2000_mean']:>9.4f}")
    rows_out.append(["raw", n_pairs, raw_summary["lab_total"], "", raw_summary["ssim"],
                     raw_summary["wlab_mean"], raw_summary["de2000_mean"]])

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
            print(f"{scale:11}  WARNING: no source_mode=correct rows in {path}")
            continue
        n = len(vals["lab_total"])
        model_lab = statistics.mean(vals["lab_total"])
        ssim = statistics.mean(vals["ssim"])
        wlab = statistics.mean(vals["wlab_mean"])
        de2000 = statistics.mean(vals["de2000_mean"])
        recovery = raw_summary["lab_total"] - model_lab
        print(f"{scale:11}{n:>5}{model_lab:>12.4f}{recovery:>11.4f}{ssim:>9.4f}"
              f"{wlab:>9.4f}{de2000:>9.4f}")
        rows_out.append([scale, n, model_lab, recovery, ssim, wlab, de2000])
        results[float(scale)] = {"recovery": recovery, "ssim": ssim, "n": n}

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lora_scale", "n", "model_lab", "recovery_delta_lab", "ssim",
                    "wlab_mean", "de2000_mean"])
        w.writerows(rows_out)

    print(f"\n{'=' * 100}\nSelection rule + interpretation\n{'=' * 100}")
    sorted_scales = sorted(results)
    recoveries = [results[s]["recovery"] for s in sorted_scales]
    trend_str = " -> ".join(f"{s}:{results[s]['recovery']:.2f}" for s in sorted_scales)
    print(f"Recovery by scale: {trend_str}")

    positive = {s: results[s] for s in sorted_scales if results[s]["recovery"] > 0}
    if positive:
        best_scale = max(positive, key=lambda s: positive[s]["ssim"])
        print(f"\n>>> {len(positive)} configuration(s) achieve positive recovery: "
              f"{sorted(positive.keys())}.")
        print(f">>> SELECTED (highest SSIM among positive-recovery configs): "
              f"lora_scale={best_scale}  recovery={positive[best_scale]['recovery']:.4f}  "
              f"ssim={positive[best_scale]['ssim']:.4f}")
    else:
        print("\n>>> NO configuration achieves positive recovery on internal validation.")
        monotone_improving = all(recoveries[i] <= recoveries[i + 1] for i in range(len(recoveries) - 1))
        net_improved = recoveries[-1] > recoveries[0] if len(recoveries) > 1 else False
        if monotone_improving and net_improved:
            print(">>> Trend IS consistently toward zero/positive (monotone non-decreasing, "
                  f"{recoveries[0]:.2f} -> {recoveries[-1]:.2f}) but does not cross zero within "
                  "this grid. Per the diagnostic plan, this is the trigger to run the second "
                  "grid: strength in {0.50, 0.60, 0.70} x lora_scale in {1.0, 1.5, 2.0}, "
                  "controlnet_conditioning_scale=1.0 and guidance=2.0 fixed.")
        else:
            print(">>> Trend is NOT consistently toward zero/positive (non-monotone, or net "
                  f"movement is not toward positive: {recoveries[0]:.2f} -> {recoveries[-1]:.2f}). "
                  "Per instruction, STOP inference-only tuning here. The remaining hypothesis is "
                  "insufficient native-1024 paired data (H3) -- propose a bounded P3-07b training "
                  "experiment using all 96 available non-overlapping A03/H03 1024 pairs, everything "
                  "else fixed, explicitly labelled as a supplementary >50-pair follow-up and NOT "
                  "evidence for the formal <=50-pair H1 claim. Do not add slides, rank sweeps, new "
                  "losses, or overlapping crops.")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
