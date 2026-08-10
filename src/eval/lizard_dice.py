#!/usr/bin/env python3
"""
lizard_dice.py -- score HoVer-Net predictions (from hovernet_wrapper.py) against
Lizard's human-validated ground-truth nuclear masks (P2-08, Structural Safety).

Primary metric: binary, pixel-pooled Dice -- HoVer-Net's own get_dice_1()/
"standard DICE", and the same convention the Lizard paper's own authors report
as their headline "Binary Dice" (see tickets/PHASE2-TICKETS.md P2-08 for why
this was chosen over instance- or class-matched Dice). Secondary metrics named
in the proposal: IoU, a simple greedy centroid-matched object-level F1, and
nuclear count consistency.

Two-stage use, matching the proposal's own validation-before-trust requirement:
    1. Run on HoVer-Net(original Lizard patch) vs ground truth -> Dice(A,G).
       This is the validation gate -- HoVer-Net must look sane here before its
       output means anything as a measurement instrument.
    2. Run on HoVer-Net(normalised patch) vs ground truth, with --against
       pointing at run 1's summary.csv, to get
       Relative Dice = Dice(B,G) / Dice(A,G) >= 0.95 (proposal's threshold).

Deliberately does NOT import tiatoolbox or joblib -- runs in the plain
`stainnorm` env (only needs numpy/scipy), unlike hovernet_wrapper.py which
must run in the separate stainnorm-hovernet env. hovernet_wrapper.py
re-serialises its instance dicts with stdlib pickle for exactly this reason.

Usage
-----
    python lizard_dice.py --pred-dir eval/lizard_original \
        --labels-dir lizard_heldout/labels --out eval/lizard_original

    python lizard_dice.py --pred-dir eval/lizard_normalised \
        --labels-dir lizard_heldout/labels --out eval/lizard_normalised \
        --against eval/lizard_original/summary.csv

Dependencies: numpy, scipy.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from progress import progress


def binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * inter / denom) if denom else 1.0


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union else 1.0


def object_f1(pred_centroids: list, gt_centroids: list, match_radius: float = 12.0) -> float:
    """Greedy nearest-centroid object-level F1.

    Simplified convention, documented as such: greedy nearest-centroid matching
    within `match_radius` pixels, not Hungarian-optimal assignment or IoU-based
    matching. This is the simplest defensible choice for a secondary metric --
    the primary pass/fail metric is binary_dice() above, not this. Revisit only
    if stricter matching is specifically needed.
    """
    if not gt_centroids:
        return 1.0 if not pred_centroids else 0.0
    if not pred_centroids:
        return 0.0
    gt_arr = np.asarray(gt_centroids, dtype=np.float64)
    used = np.zeros(len(gt_arr), dtype=bool)
    tp = 0
    for px, py in pred_centroids:
        d = np.hypot(gt_arr[:, 0] - px, gt_arr[:, 1] - py)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= match_radius:
            used[j] = True
            tp += 1
    fp = len(pred_centroids) - tp
    fn = len(gt_centroids) - tp
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


def load_gt(mat_path: Path) -> tuple[np.ndarray, list]:
    """Return (binary foreground mask, [(x, y), ...] centroids) from a Lizard .mat label.

    Field layout verified directly (not just from Lizard's README) against a
    real label file during P2-08 scoping: inst_map int32 HxW (0=background),
    centroid (N,2) ordered (x, y) -- see tickets/PHASE2-TICKETS.md P2-08.
    """
    d = loadmat(str(mat_path))
    mask = d["inst_map"] > 0
    centroids = [tuple(c) for c in np.asarray(d["centroid"], dtype=np.float64)]
    return mask, centroids


def load_pred_centroids(inst_dict: dict) -> list:
    return [tuple(v["centroid"]) for v in inst_dict.values()]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Score HoVer-Net predictions against Lizard ground truth (P2-08).")
    ap.add_argument("--pred-dir", required=True,
                    help="Output dir from hovernet_wrapper.py (has instances/, masks/, manifest.csv).")
    ap.add_argument("--labels-dir", required=True, help="Lizard .mat ground-truth label dir.")
    ap.add_argument("--out", required=True, help="Output dir for per_image.csv / summary.csv.")
    ap.add_argument("--against", default=None,
                    help="Another run's summary.csv (the 'A'/original run) to compute "
                         "Relative Dice = Dice(this)/Dice(against) against.")
    ap.add_argument("--match-radius", type=float, default=12.0,
                    help="Centroid-match radius in pixels for object-level F1.")
    return ap.parse_args()


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    mask_dir = pred_dir / "masks"
    inst_dir = pred_dir / "instances"
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(mask_dir.glob("*.npy"))
    if not mask_paths:
        raise SystemExit(f"No predicted masks found in {mask_dir} -- run hovernet_wrapper.py first.")

    rows = []
    per_image_path = out_dir / "per_image.csv"
    with open(per_image_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "dice", "iou", "object_f1", "n_pred_nuclei", "n_gt_nuclei", "count_ratio"])
        for mp in progress(mask_paths, desc="scoring"):
            stem = mp.stem
            mat_path = labels_dir / f"{stem}.mat"
            if not mat_path.exists():
                print(f"  WARNING: no ground truth for {stem} ({mat_path}) -- skipping.")
                continue

            pred_mask = np.load(mp)
            gt_mask, gt_centroids = load_gt(mat_path)
            if pred_mask.shape != gt_mask.shape:
                print(f"  WARNING: shape mismatch for {stem}: pred {pred_mask.shape} "
                      f"vs gt {gt_mask.shape} -- skipping.")
                continue

            inst_path = inst_dir / f"{stem}.pkl"
            if inst_path.exists():
                with open(inst_path, "rb") as ifh:
                    inst_dict = pickle.load(ifh)
            else:
                inst_dict = {}
            pred_centroids = load_pred_centroids(inst_dict)

            dice = binary_dice(pred_mask, gt_mask)
            iou = binary_iou(pred_mask, gt_mask)
            f1 = object_f1(pred_centroids, gt_centroids, match_radius=args.match_radius)
            n_pred, n_gt = len(pred_centroids), len(gt_centroids)
            count_ratio = (n_pred / n_gt) if n_gt else None

            rows.append({"image": stem, "dice": dice, "iou": iou, "object_f1": f1,
                        "n_pred_nuclei": n_pred, "n_gt_nuclei": n_gt, "count_ratio": count_ratio})
            w.writerow([stem, round(dice, 5), round(iou, 5), round(f1, 5),
                       n_pred, n_gt, round(count_ratio, 4) if count_ratio is not None else ""])

    if not rows:
        raise SystemExit("No images scored -- check --pred-dir / --labels-dir paths.")

    mean_dice = float(np.mean([r["dice"] for r in rows]))
    mean_iou = float(np.mean([r["iou"] for r in rows]))
    mean_f1 = float(np.mean([r["object_f1"] for r in rows]))
    ratios = [r["count_ratio"] for r in rows if r["count_ratio"] is not None]
    mean_count_ratio = float(np.mean(ratios)) if ratios else float("nan")

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value", "n_images"])
        w.writerow(["dice", round(mean_dice, 5), len(rows)])
        w.writerow(["iou", round(mean_iou, 5), len(rows)])
        w.writerow(["object_f1", round(mean_f1, 5), len(rows)])
        w.writerow(["count_ratio", round(mean_count_ratio, 5), len(rows)])

    print(f"\n{pred_dir.name}: mean Dice={mean_dice:.4f}  IoU={mean_iou:.4f}  "
          f"object-F1={mean_f1:.4f}  count_ratio={mean_count_ratio:.4f}  (n={len(rows)} images)")
    print(f"Per-image: {per_image_path}")
    print(f"Summary  : {summary_path}")

    if args.against:
        against_path = Path(args.against)
        with open(against_path, newline="") as fh:
            base = {row["metric"]: float(row["value"]) for row in csv.DictReader(fh)}
        base_dice = base.get("dice")
        if base_dice:
            relative_dice = mean_dice / base_dice
            verdict = "PASS (>= 0.95)" if relative_dice >= 0.95 else "FAIL (< 0.95)"
            print(f"\nRelative Dice = Dice(this)/Dice({against_path}) "
                  f"= {mean_dice:.4f}/{base_dice:.4f} = {relative_dice:.4f}  {verdict}")
            with open(out_dir / "relative_dice.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["this_dice", "baseline_dice", "relative_dice", "threshold", "pass"])
                w.writerow([round(mean_dice, 5), round(base_dice, 5),
                           round(relative_dice, 5), 0.95, relative_dice >= 0.95])
        else:
            print(f"WARNING: no 'dice' row found in {against_path}, cannot compute Relative Dice.")


if __name__ == "__main__":
    main()
