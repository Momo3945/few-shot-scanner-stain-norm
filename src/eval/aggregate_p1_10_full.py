#!/usr/bin/env python3
"""
aggregate_p1_10_full.py -- P1-10 full held-out evaluation: per-slide/ALL/
ALL_excl_outliers aggregation + recovery-delta-vs-baseline, matching
score_outputs.py's exact methodology, but reading from
score_p1_10_ablation.py's per_crop.csv (source_mode/seed-shaped) instead of
score_outputs.py's own strength-shaped manifest -- the two scripts' schemas
differ (see infer_colour_translation.slurm's header), so this is a small
dedicated aggregator rather than a change to either existing scorer.

All 3 seeds are pooled together per slide (not averaged-then-spread) for the
headline comparison table, since every comparison point (A4/A5@0.20, Macenko,
Reinhard, histogram matching) is a single-seed run -- pooling gives a lower-
variance estimate and is the most directly comparable presentation. Seed-to-
seed spread was already separately validated at the smoke-gate stage.

Usage
-----
    python aggregate_p1_10_full.py \
        --per-crop eval/p1_10_full_heldout_summary/per_crop.csv \
        --baseline pairs/baseline_metrics/baseline_summary.csv \
        --out eval/p1_10_full_heldout_summary/eval_summary_final.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from metrics import flag_outliers, _mean_finite
from score_outputs import read_baseline_lab, baseline_all_for_slides

KEYS = ("lab_total", "wlab_mean", "de2000_mean", "ssim", "psnr", "mae")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-crop", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outlier-z", type=float, default=3.5)
    args = ap.parse_args()

    slides = defaultdict(list)  # slide -> list of per-crop metric dicts (all seeds pooled)
    with open(args.per_crop, newline="") as fh:
        for r in csv.DictReader(fh):
            m = {k: float(r[k]) for k in KEYS}
            slides[r["slide"]].append(m)

    baseline = read_baseline_lab(args.baseline) if args.baseline else None

    slide_lab = {s: _mean_finite([m["lab_total"] for m in slides[s]]) for s in slides}
    flags = flag_outliers(slide_lab, z_thresh=args.outlier_z)
    outliers = {s for s in flags if flags[s]["outlier"]}

    def pool(exclude=frozenset()):
        acc = {k: [] for k in KEYS}
        for s in slides:
            if s in exclude:
                continue
            for m in slides[s]:
                for k in KEYS:
                    acc[k].append(m[k])
        return {k: _mean_finite(acc[k]) for k in KEYS}, \
               sum(len(slides[s]) for s in slides if s not in exclude)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "n_crops", "lab_total", "wlab_mean", "de2000_mean",
                    "ssim", "psnr", "mae", "robust_z", "outlier", "recovery_delta_lab"])

        allm, alln = pool()
        base_all = baseline_all_for_slides(baseline, slides) if baseline else None
        delta = (round(base_all - allm["lab_total"], 4)
                 if base_all is not None and allm["lab_total"] is not None else "")
        print(f"ALL ({len(slides)} slides, n={alln}): lab {allm['lab_total']:.2f}  "
              f"wlab {allm['wlab_mean']:.2f}  de2000 {allm['de2000_mean']:.2f}  "
              f"ssim {allm['ssim']:.4f}  psnr {allm['psnr']:.2f}  mae {allm['mae']:.2f}"
              + (f"  recovery Δlab {delta}" if delta != "" else ""))
        w.writerow(["ALL", alln, allm["lab_total"], allm["wlab_mean"], allm["de2000_mean"],
                    allm["ssim"], allm["psnr"], allm["mae"], "", "", delta])

        if outliers:
            cleanm, cleann = pool(exclude=outliers)
            base_clean = baseline_all_for_slides(
                baseline, {s: slides[s] for s in slides if s not in outliers}) if baseline else None
            cdelta = (round(base_clean - cleanm["lab_total"], 4)
                      if base_clean is not None and cleanm["lab_total"] is not None else "")
            print(f"ALL_excl_outliers ({','.join(sorted(outliers))} excluded, n={cleann}): "
                  f"lab {cleanm['lab_total']:.2f}  wlab {cleanm['wlab_mean']:.2f}  "
                  f"de2000 {cleanm['de2000_mean']:.2f}  ssim {cleanm['ssim']:.4f}  "
                  f"psnr {cleanm['psnr']:.2f}  mae {cleanm['mae']:.2f}"
                  + (f"  recovery Δlab {cdelta}" if cdelta != "" else ""))
            w.writerow(["ALL_excl_outliers", cleann, cleanm["lab_total"], cleanm["wlab_mean"],
                        cleanm["de2000_mean"], cleanm["ssim"], cleanm["psnr"], cleanm["mae"],
                        "", "", cdelta])

        for s in sorted(slides):
            sm = {k: _mean_finite([m[k] for m in slides[s]]) for k in KEYS}
            sdelta = ""
            if baseline and s in baseline["per_slide"] and sm["lab_total"] is not None:
                sdelta = round(baseline["per_slide"][s][0] - sm["lab_total"], 4)
            flag = "*** OUTLIER" if flags[s]["outlier"] else ""
            print(f"  {s}: lab {sm['lab_total']:.2f}  wlab {sm['wlab_mean']:.2f}  "
                  f"de2000 {sm['de2000_mean']:.2f}  ssim {sm['ssim']:.4f}  "
                  f"psnr {sm['psnr']:.2f}  mae {sm['mae']:.2f}  z={flags[s]['z']:.2f}  {flag}"
                  + (f"  Δlab {sdelta}" if sdelta != "" else ""))
            w.writerow([s, len(slides[s]), sm["lab_total"], sm["wlab_mean"], sm["de2000_mean"],
                        sm["ssim"], sm["psnr"], sm["mae"], round(flags[s]["z"], 3),
                        flags[s]["outlier"], sdelta])

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
