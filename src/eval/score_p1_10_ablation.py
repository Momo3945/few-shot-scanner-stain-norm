#!/usr/bin/env python3
"""
score_p1_10_ablation.py -- P1-10's mandatory source-conditioning ablation /
VAE-only floor scorer. Deliberately NOT a change to score_outputs.py (that
script is validated, in-use code elsewhere, and its manifest schema is
strength-shaped, not source_mode/seed-shaped) -- this is a small, dedicated
scorer for this ticket's specific comparisons only.

Reads one eval_manifest.csv per --eval-dir (each produced by a separate
infer_colour_translation.py run -- one dir per source_mode, since the
launcher runs correct/zero/shuffled as separate jobs), scores every crop with
the same metrics.py machinery every other run in this project uses, and
reports PER SOURCE_MODE aggregates: mean +/- spread ACROSS SEEDS (the ticket's
seed protocol requirement), not a single-seed point estimate.

Also reports a per-crop PAIRED win-rate: for each (slide, frame, x, y) crop
that appears in more than one source_mode's manifest, which mode scores best
on SSIM (this is the ticket's actual acceptance bar -- "correct must win on
structural/content metrics", not just have a better independent mean).

Usage
-----
    python score_p1_10_ablation.py \
        --eval-dirs eval/p1_10_ablation_correct eval/p1_10_ablation_zero eval/p1_10_ablation_shuffled \
        --out eval/p1_10_ablation_summary

Dependencies: numpy, opencv-python-headless. Requires metrics.py and
score_outputs.py's read_rgb_png on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from metrics import score_aligned_pair
from score_outputs import read_rgb_png

METRIC_KEYS = ("lab_total", "wlab_mean", "de2000_mean", "ssim", "psnr", "mae")


def parse_args():
    ap = argparse.ArgumentParser(
        description="P1-10 source-conditioning ablation / VAE-only floor scorer.")
    ap.add_argument("--eval-dirs", nargs="+", required=True,
                    help="One or more eval dirs, each with its own eval_manifest.csv "
                         "(typically one per source_mode: correct/zero/shuffled, or a "
                         "single vae_only dir).")
    ap.add_argument("--out", required=True, help="Output dir for per_crop.csv / summary.csv.")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_crop_path = out_dir / "per_crop.csv"
    per_crop_fh = open(per_crop_path, "w", newline="")
    pc_w = csv.writer(per_crop_fh)
    pc_w.writerow(["source_mode", "seed", "crop_id", "slide", "frame", "x", "y", *METRIC_KEYS])

    # mode -> seed -> metric_key -> [values across crops for that one seed] --
    # keyed by seed FIRST so the summary can aggregate crops within a seed before
    # comparing across seeds, not pool crop and seed variance together.
    by_mode_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # (slide, frame, x, y) -> mode -> [ssim values across seeds] -- for the paired win-rate
    by_crop = defaultdict(lambda: defaultdict(list))
    # mode -> {(slide, frame, x, y, seed)} -- inventory check below confirms every
    # mode covers the same crop x seed set. A silent filename collision (like the
    # one caught and fixed in the pairs-dir tag construction) would otherwise show
    # up only as a slightly-off aggregate, not an obvious failure.
    inventory = defaultdict(set)

    n_rows = 0
    for eval_dir in args.eval_dirs:
        eval_dir = Path(eval_dir)
        man_path = eval_dir / "eval_manifest.csv"
        if not man_path.exists():
            print(f"  WARNING: no manifest at {man_path} -- skipping.")
            continue
        with open(man_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            out_rgb = read_rgb_png(eval_dir / r["output_path"])
            ref_rgb = read_rgb_png(eval_dir / r["reference_path"])
            m = score_aligned_pair(out_rgb, ref_rgb)
            mode = r["source_mode"]
            seed = r["seed"]
            pc_w.writerow([mode, seed, r["crop_id"], r["slide"], r["frame"], r["x"], r["y"],
                          *(round(m[k], 5) for k in METRIC_KEYS)])
            for k in METRIC_KEYS:
                by_mode_seed[mode][seed][k].append(m[k])
            # crop_id (not (slide,frame,x,y)) -- pairs-dir mode writes x=y=0 for every
            # crop (they're pre-cropped pair images, not sub-crops of a frame), so
            # multiple distinct pairs from the same frame would otherwise collide onto
            # one crop_key and get silently conflated here even though the output
            # files themselves no longer collide (see tag_id fix in
            # infer_colour_translation.py). crop_id is unique per real crop always.
            crop_key = r["crop_id"]
            by_crop[crop_key][mode].append(m["ssim"])
            inventory[mode].add((crop_key, seed))
            n_rows += 1
    per_crop_fh.close()

    # ---- inventory check: every mode must cover the identical (crop, seed) set ----
    modes = sorted(inventory)
    if len(modes) > 1:
        reference = inventory[modes[0]]
        for mode in modes[1:]:
            missing = reference - inventory[mode]
            extra = inventory[mode] - reference
            if missing or extra:
                print(f"  WARNING: inventory mismatch for mode '{mode}' vs '{modes[0]}' -- "
                      f"{len(missing)} missing, {len(extra)} extra (crop,seed) keys. "
                      f"Comparison below may not be apples-to-apples.")
        counts = {m: len(inventory[m]) for m in modes}
        print(f"Inventory: {counts}" + ("  (all modes match)" if len(set(counts.values())) == 1
              and all(inventory[m] == reference for m in modes) else ""))

    if n_rows == 0:
        raise SystemExit("No rows scored -- check --eval-dirs paths.")

    # ---- per-mode summary: mean +/- spread ACROSS SEEDS, per the ticket's seed
    # protocol -- NOT pooled (crop, seed) stdev, which would measure tissue-crop
    # variation (large, uninteresting) rather than seed-to-seed variation (small,
    # the actual thing being checked). For each mode: first aggregate all crops
    # within a single seed into one number per seed, THEN take mean+/-stdev across
    # those per-seed aggregates (at most len(seeds) values). ----
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source_mode", "n_seeds", "n_rows", *[f"{k}_mean" for k in METRIC_KEYS],
                    *[f"{k}_stdev_across_seeds" for k in METRIC_KEYS]])
        for mode in sorted(by_mode_seed):
            seed_vals = by_mode_seed[mode]
            n_rows_mode = sum(len(seed_vals[s]["ssim"]) for s in seed_vals)
            n_seeds = len(seed_vals)
            per_seed_means = {k: [statistics.mean(seed_vals[s][k]) for s in sorted(seed_vals)]
                              for k in METRIC_KEYS}
            means = [round(statistics.mean(per_seed_means[k]), 5) for k in METRIC_KEYS]
            stdevs = [round(statistics.stdev(per_seed_means[k]), 5) if n_seeds > 1 else 0.0
                     for k in METRIC_KEYS]
            w.writerow([mode, n_seeds, n_rows_mode, *means, *stdevs])
            print(f"{mode}: n_seeds={n_seeds} n_rows={n_rows_mode}  " +
                  "  ".join(f"{k}={m:.4f}+/-{s:.4f}(across seeds)"
                  for k, m, s in zip(METRIC_KEYS, means, stdevs)))

    # ---- paired win-rate: for crops scored under >1 mode, which mode has the
    # best mean SSIM (across its seeds) for that specific crop -- this is the
    # ticket's actual acceptance bar, not just independent means ----
    multi_mode_crops = {k: v for k, v in by_crop.items() if len(v) > 1}
    if multi_mode_crops:
        win_counts = defaultdict(int)
        for crop_key, mode_vals in multi_mode_crops.items():
            best_mode = max(mode_vals, key=lambda m: statistics.mean(mode_vals[m]))
            win_counts[best_mode] += 1
        print(f"\nPaired win-rate (best mean SSIM per crop, {len(multi_mode_crops)} crops "
              f"scored under >1 mode):")
        for mode in sorted(win_counts):
            print(f"  {mode}: wins {win_counts[mode]}/{len(multi_mode_crops)} crops")
        with open(out_dir / "paired_win_rate.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["source_mode", "wins", "n_crops_compared"])
            for mode in sorted(win_counts):
                w.writerow([mode, win_counts[mode], len(multi_mode_crops)])
    else:
        print("\nNo crops scored under more than one source_mode -- skipping paired win-rate "
              "(pass multiple --eval-dirs covering the same crops for this check).")

    print(f"\nPer-crop : {per_crop_path}")
    print(f"Summary  : {summary_path}")


if __name__ == "__main__":
    main()
