#!/usr/bin/env python3
"""Crop-weighted aggregate over the four typical held-out slides (A08/A09/A13/A16) plus
the A06 outlier row, for every classical + learned baseline, A2H and H2A (P2-11/P2-11b).

Uses one fixed slide set for every method (the per-method `ALL_excl_outliers` rows in
eval_summary.csv use per-method robust-z flags and are not comparable across methods).
Reads each method's eval_summary.csv; writes aggregate_typical_slides.csv next to this file.
Run from docs/results/:  python learned_baselines/aggregate_typical_slides.py
"""
import csv
from pathlib import Path

TYPICAL = ("A08", "A09", "A13", "A16")
RUNS = [
    ("A2H", "Macenko", "classical_baselines/macenko"),
    ("A2H", "Reinhard", "classical_baselines/reinhard"),
    ("A2H", "Histogram Matching", "classical_baselines/histogram_matching"),
    ("A2H", "StainNet", "learned_baselines/stainnet"),
    ("A2H", "ParamNet", "learned_baselines/paramnet"),
    ("A2H", "StainGAN", "learned_baselines/staingan"),
    ("H2A", "Macenko", "classical_baselines/h2a_macenko"),
    ("H2A", "Reinhard", "classical_baselines/h2a_reinhard"),
    ("H2A", "Histogram Matching", "classical_baselines/h2a_histogram_matching"),
    ("H2A", "StainNet", "learned_baselines/h2a_stainnet"),
    ("H2A", "ParamNet", "learned_baselines/h2a_paramnet"),
    ("H2A", "StainGAN", "learned_baselines/h2a_staingan"),
]
METRICS = ("lab_total", "wlab_mean", "ssim", "recovery_delta_lab")

root = Path(__file__).resolve().parents[1]
out_rows = []
for direction, method, folder in RUNS:
    p = root / folder / "eval_summary.csv"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    rows = {r["scope"]: r for r in csv.DictReader(open(p))}
    n = sum(int(rows[s]["n_crops"]) for s in TYPICAL)
    agg = {m: sum(float(rows[s][m]) * int(rows[s]["n_crops"]) for s in TYPICAL) / n for m in METRICS}
    out_rows.append({"direction": direction, "method": method,
                     "n_typical": n, **{f"typical_{m}": round(agg[m], 3) for m in METRICS},
                     "a06_lab_total": round(float(rows["A06"]["lab_total"]), 3),
                     "a06_recovery_delta_lab": round(float(rows["A06"]["recovery_delta_lab"]), 3),
                     "all_n": rows["ALL"]["n_crops"],
                     "all_lab_total": round(float(rows["ALL"]["lab_total"]), 3),
                     "all_ssim": round(float(rows["ALL"]["ssim"]), 3),
                     "all_recovery_delta_lab": round(float(rows["ALL"]["recovery_delta_lab"]), 3)})

dest = Path(__file__).with_name("aggregate_typical_slides.csv")
with open(dest, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
    w.writeheader(); w.writerows(out_rows)
for r in out_rows:
    print(r)
print("wrote", dest)
