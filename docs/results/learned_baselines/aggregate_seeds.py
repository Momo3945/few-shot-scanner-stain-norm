#!/usr/bin/env python3
"""Three-seed summary of the A2H learned baselines (P2-11b seeds, 2026-11).

Seed 3407 is the original run (<method>/), seeds 1 and 2 are <method>_s1/, <method>_s2/ (eval_summary.csv
copied from /datasets/mhoosen/stain-norm/eval/<method>_s<N>/). For each run: LAB Wasserstein and SSIM
pooled (ALL), on the four typical slides (crop-weighted, as aggregate_typical_slides.py) and on A06, plus
per-slide typical recovery. Then mean, SD and min-max over seeds, and the pre-stated ranking check:
on every seed, ParamNet and StainGAN each beat the fused SD1.5 pipeline on pooled SSIM (> 0.729) and on
typical LAB (< 18.11). Writes aggregate_seeds.csv next to this file.
Run from docs/results/:  python learned_baselines/aggregate_seeds.py
"""
import csv
import statistics as st
from pathlib import Path

TYPICAL = ("A08", "A09", "A13", "A16")
METHODS = ("stainnet", "paramnet", "staingan")
SEEDS = {3407: "{m}", 1: "{m}_s1", 2: "{m}_s2"}
FUSED_SSIM_ALL, FUSED_LAB_TYPICAL = 0.729, 18.11

here = Path(__file__).resolve().parent


def run_metrics(folder):
    rows = {r["scope"]: r for r in csv.DictReader(open(here / folder / "eval_summary.csv"))}
    n = sum(int(rows[s]["n_crops"]) for s in TYPICAL)
    typ = lambda k: sum(float(rows[s][k]) * int(rows[s]["n_crops"]) for s in TYPICAL) / n
    out = {"all_lab": float(rows["ALL"]["lab_total"]), "typical_lab": typ("lab_total"),
           "a06_lab": float(rows["A06"]["lab_total"]), "all_ssim": float(rows["ALL"]["ssim"]),
           "typical_ssim": typ("ssim"), "a06_ssim": float(rows["A06"]["ssim"]),
           "typical_dlab": typ("recovery_delta_lab"), "n_all": int(rows["ALL"]["n_crops"])}
    for s in TYPICAL:
        out[f"dlab_{s}"] = float(rows[s]["recovery_delta_lab"])
    return out


per_run, summary = [], []
for m in METHODS:
    runs = {seed: run_metrics(f.format(m=m)) for seed, f in SEEDS.items()}
    for seed, r in runs.items():
        assert r["n_all"] == 496, (m, seed, r["n_all"])
        per_run.append({"method": m, "seed": seed, **{k: round(v, 4) for k, v in r.items()}})
    keys = [k for k in runs[3407] if k != "n_all"]
    agg = {"method": m}
    for k in keys:
        vals = [runs[s][k] for s in SEEDS]
        agg[f"{k}_mean"] = round(st.mean(vals), 4)
        agg[f"{k}_sd"] = round(st.stdev(vals), 4)
        agg[f"{k}_min"] = round(min(vals), 4)
        agg[f"{k}_max"] = round(max(vals), 4)
    summary.append(agg)
    print(f"{m:9s} " + "  ".join(f"{k}={agg[k + '_mean']:.3f}±{agg[k + '_sd']:.3f} [{agg[k + '_min']:.3f},{agg[k + '_max']:.3f}]"
                                for k in ("all_lab", "typical_lab", "a06_lab", "all_ssim", "typical_ssim", "a06_ssim")))

print("\nPre-stated ranking check (every seed: pooled SSIM > 0.729 and typical LAB < 18.11):")
for m in ("paramnet", "staingan"):
    for r in (x for x in per_run if x["method"] == m):
        ok = r["all_ssim"] > FUSED_SSIM_ALL and r["typical_lab"] < FUSED_LAB_TYPICAL
        print(f"  {m} seed {r['seed']}: SSIM {r['all_ssim']:.3f}, typical LAB {r['typical_lab']:.2f} -> {'holds' if ok else 'FAILS'}")
print("Per-slide typical recovery (ParamNet), all seeds:",
      sorted(round(r[f"dlab_{s}"], 1) for r in per_run if r["method"] == "paramnet" for s in TYPICAL))

with open(here / "aggregate_seeds.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(per_run[0]))
    w.writeheader(); w.writerows(per_run)
with open(here / "aggregate_seeds_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0]))
    w.writeheader(); w.writerows(summary)
print("wrote aggregate_seeds.csv and aggregate_seeds_summary.csv")
