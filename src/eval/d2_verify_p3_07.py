#!/usr/bin/env python3
"""
d2_verify_p3_07.py -- P3-07 diagnostic D2: colour-LoRA-disabled comparison.

Reads the per_crop.csv scored by score_p1_10_ablation.py for the D2 no-LoRA
inference run (infer_colour_translation_sdxl.py --no-lora, A06+A08 held-out
subset) and d1_verify_p3_07.py's d1_summary.csv (which already has, per
slide: the 1024-native raw baseline, and the WITH-LoRA model LAB/recovery
from the full P3-07 held-out run). Prints, per slide and pooled:

    raw_1024 | model_with_lora | recovery_with_lora | model_no_lora | recovery_no_lora

and states which case applies:
    Case A: no-LoRA recovery is already strongly negative (roughly matching
            the with-LoRA figure) -> the frozen SDXL base + trained
            ControlNet + source conditioning is what drifts colour negative;
            the colour LoRA is too weak to overcome it.
    Case B: no-LoRA recovery is near zero or positive -> the 1024 colour
            LoRA itself learned the wrong mapping; the base+ControlNet path
            is not the culprit.

Usage
-----
    python d2_verify_p3_07.py \
        --no-lora-per-crop eval/p3_07_d2_no_lora_summary/per_crop.csv \
        --d1-summary eval/p3_07_d1_verify/d1_summary.csv \
        --out eval/p3_07_d2_no_lora_summary/d2_verdict.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="P3-07 D2: colour-LoRA-disabled comparison.")
    ap.add_argument("--no-lora-per-crop", required=True)
    ap.add_argument("--d1-summary", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()

    with open(args.d1_summary, newline="") as fh:
        d1 = {r["scope"]: r for r in csv.DictReader(fh)}

    by_slide = defaultdict(list)
    with open(args.no_lora_per_crop, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["source_mode"] != "correct":
                continue
            by_slide[r["slide"]].append(float(r["lab_total"]))

    if not by_slide:
        raise SystemExit(f"No source_mode=correct rows found in {args.no_lora_per_crop}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["slide", "n", "raw_1024", "model_with_lora", "recovery_with_lora",
              "model_no_lora", "recovery_no_lora"]
    rows_out = []

    print("=" * 92)
    print("D2 RESULT -- colour LoRA DISABLED vs the full P3-07 (WITH LoRA) run")
    print("=" * 92)
    print(f"{'slide':7}{'n':>5}{'raw_1024':>11}{'model_w/lora':>14}{'recov_w/lora':>14}"
          f"{'model_no_lora':>15}{'recov_no_lora':>15}")

    pooled_raw, pooled_no_lora = [], []
    for slide in sorted(by_slide):
        if slide not in d1:
            print(f"  WARNING: {slide} not in {args.d1_summary} -- skipping.")
            continue
        raw = float(d1[slide]["raw_1024_lab"])
        model_w = float(d1[slide]["model_1024_lab"])
        recov_w = float(d1[slide]["recovery_delta_lab_1024_native"])
        model_no = statistics.mean(by_slide[slide])
        recov_no = raw - model_no
        n = len(by_slide[slide])
        print(f"{slide:7}{n:>5}{raw:>11.4f}{model_w:>14.4f}{recov_w:>14.4f}"
              f"{model_no:>15.4f}{recov_no:>15.4f}")
        rows_out.append([slide, n, raw, model_w, recov_w, model_no, recov_no])
        pooled_raw.extend([raw] * n)
        pooled_no_lora.extend(by_slide[slide])

    pooled_raw_m = statistics.mean(pooled_raw)
    pooled_no_lora_m = statistics.mean(pooled_no_lora)
    pooled_recov_no = pooled_raw_m - pooled_no_lora_m
    all_row = d1.get("ALL")
    all_recov_w = float(all_row["recovery_delta_lab_1024_native"]) if all_row else None
    print("-" * 92)
    print(f"{'POOLED':7}{len(pooled_no_lora):>5}{pooled_raw_m:>11.4f}{'':>14}"
          f"{(all_recov_w if all_recov_w is not None else float('nan')):>14.4f}"
          f"{pooled_no_lora_m:>15.4f}{pooled_recov_no:>15.4f}")
    rows_out.append(["POOLED(A06+A08)", len(pooled_no_lora), pooled_raw_m, "",
                     all_recov_w, pooled_no_lora_m, pooled_recov_no])

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows_out)

    print(f"\nPooled A06+A08 recovery: WITH-lora (from D1) = {all_recov_w}, "
          f"NO-lora = {round(pooled_recov_no, 4)}")
    if pooled_recov_no < -2.0:
        print(">>> CASE A: no-LoRA recovery is already strongly negative -- the frozen SDXL "
              "base + trained ControlNet + source conditioning is what drifts colour negative. "
              "The colour LoRA is too weak to overcome it (H1/H2 supported over H2-as-sole-cause; "
              "proceed to D3 to test whether source-RGB conditioning specifically is responsible).")
    elif pooled_recov_no > 0:
        print(">>> CASE B: no-LoRA recovery is positive/near-zero -- the 1024 colour LoRA itself "
              "learned the wrong mapping. The base+ControlNet path is not the culprit "
              "(re-examine the LoRA training run/data, not source-conditioning balance).")
    else:
        print(">>> AMBIGUOUS: no-LoRA recovery is negative but smaller in magnitude than the "
              "with-LoRA figure -- both the base path and the LoRA are contributing. Report both "
              "numbers; do not force a single-cause verdict.")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
