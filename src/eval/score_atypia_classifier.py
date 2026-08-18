#!/usr/bin/env python3
"""
score_atypia_classifier.py -- score the trained atypia classifier against every
already-computed method's crop outputs (P2-09, Clinical Utility).

Per the proposal: "recovery delta = performance(Proposed) - performance(Raw
Hamamatsu)". This script runs the classifier ONCE (inference only, no retraining)
over every method's existing eval_manifest.csv outputs, joins predictions back to
the true atypia score via the testing-root manifest built by
build_atypia_manifest.py, and reports accuracy/macro-F1/macro-AUC per method --
mirroring score_outputs.py's per-slide + ALL/ALL_excl_outliers aggregation
convention (metrics.flag_outliers, same robust median/MAD outlier rule, not
mean/std, per CLAUDE.md).

Each existing eval_manifest.csv (infer_colour_lora.py / infer_baseline.py, byte-
identical schema: strength,slide,frame,x,y,output_path,reference_path,aperio_path)
is treated as one or more "methods": one method per distinct strength value for
the diffusion rungs (e.g. a3_s0.20, a3_s0.30 -- the manifest's own granularity,
not collapsed to a single headline strength here), and a single method for the
classical baselines (strength == "na"). Two additional synthetic methods are
derived without any new inference:
  - raw_hamamatsu: the registered-Hamamatsu reference_path column, deduplicated
    (the reference crop is strength-independent -- reused across every strength
    row in the source manifest, so naive iteration would triple/quadruple-count it).
  - raw_aperio (sanity check, not proposal-required): the ORIGINAL raw Aperio
    frame re-cropped at the manifest's own (x, y, crop) -- should score highest,
    since it's literally the classifier's training domain; a low raw_aperio score
    would flag a broken join rather than a real finding.
Both are read from exactly one --tags entry (first one processed) to avoid
re-deriving the same reference crop N times across every other tag's manifest.

Usage
-----
    python score_atypia_classifier.py \
        --checkpoint /datasets/mhoosen/stain-norm/classifier/atypia_r18/best.pt \
        --testing-manifest pairs/atypia_test_manifest.csv \
        --testing-root /datasets/mhoosen/stain-norm/mitos_heldout/mitos_atypia_2014_testing_aperio \
        --eval-root /datasets/mhoosen/stain-norm/eval \
        --out eval/atypia_classifier_scores

Dependencies: torch, torchvision, numpy, opencv-python-headless, scikit-learn,
tifffile. Requires metrics.py and registration.py on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

DEFAULT_TAGS = ["a0", "a1", "a2h_r4", "a2h_r8", "a3", "a4", "a5",
                "macenko", "reinhard", "histogram_matching"]


def parse_args():
    ap = argparse.ArgumentParser(description="Score the atypia classifier against every method's outputs.")
    ap.add_argument("--checkpoint", required=True, help="Trained classifier checkpoint (best.pt).")
    ap.add_argument("--testing-manifest", required=True,
                    help="atypia_manifest.csv built from the testing/held-out root (ground truth).")
    ap.add_argument("--testing-root", required=True,
                    help="Aperio root the testing manifest's image_path is relative to "
                         "(only used for --with-raw-aperio's re-crop).")
    ap.add_argument("--eval-root", required=True, help="Dir containing eval/<tag>/eval_manifest.csv per method.")
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--out", required=True, help="Output dir for per_crop.csv / summary.csv.")
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--crop", type=int, default=512, help="Crop size used for --with-raw-aperio's re-crop.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--outlier-z", type=float, default=3.5)
    ap.add_argument("--no-raw-aperio", action="store_true",
                    help="Skip the raw_aperio sanity-check method (on by default).")
    return ap.parse_args()


def load_testing_labels(path):
    with open(path, newline="") as fh:
        return {(r["slide"], r["frame_id"]): int(r["atypia_score"]) for r in csv.DictReader(fh)}


def main():
    args = parse_args()

    import numpy as np
    import torch
    import cv2
    from PIL import Image
    from torchvision.models import resnet18
    from sklearn.metrics import f1_score, roc_auc_score

    from metrics import flag_outliers, _mean_finite
    from registration import read_rgb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = load_testing_labels(args.testing_manifest)
    if not labels:
        raise SystemExit(f"Empty testing manifest: {args.testing_manifest}")

    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device); model.eval()
    print(f"Loaded checkpoint {args.checkpoint} (step {ckpt.get('step')}, "
          f"val_acc {ckpt.get('val_acc')})")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def preprocess(rgb):
        img = Image.fromarray(rgb).resize((args.resize, args.resize), Image.LANCZOS)
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (arr - mean) / std

    @torch.no_grad()
    def classify_batch(rgb_list):
        x = torch.stack([preprocess(r) for r in rgb_list]).to(device)
        probs = []
        for i in range(0, len(x), args.batch_size):
            probs.append(torch.softmax(model(x[i:i + args.batch_size]), dim=1).cpu())
        return torch.cat(probs).numpy()

    def read_png(path):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Build the (method -> list of (slide, frame, x, y, image-loader)) work list
    # ------------------------------------------------------------------
    eval_root = Path(args.eval_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = defaultdict(list)  # method_label -> [(slide, frame, x, y, loader_fn)]
    derived_done = False
    for tag in args.tags:
        man_path = eval_root / tag / "eval_manifest.csv"
        if not man_path.exists():
            print(f"  WARNING: no manifest at {man_path} -- skipping tag {tag}.")
            continue
        tag_dir = eval_root / tag
        with open(man_path, newline="") as fh:
            rows = list(csv.DictReader(fh))

        for r in rows:
            strength = r["strength"]
            label = tag if strength == "na" else f"{tag}_s{strength}"
            out_path = tag_dir / r["output_path"]
            methods[label].append((r["slide"], r["frame"], r["x"], r["y"],
                                   lambda p=out_path: read_png(p)))

        if not derived_done:
            seen_ref = set()
            for r in rows:
                ref_path = tag_dir / r["reference_path"]
                key = (r["slide"], r["frame"], r["x"], r["y"])
                if key in seen_ref:
                    continue
                seen_ref.add(key)
                methods["raw_hamamatsu"].append((r["slide"], r["frame"], r["x"], r["y"],
                                                 lambda p=ref_path: read_png(p)))
                if not args.no_raw_aperio:
                    aperio_path = Path(args.testing_root) / r["aperio_path"]
                    x, y, crop = int(r["x"]), int(r["y"]), args.crop
                    methods["raw_aperio"].append((r["slide"], r["frame"], r["x"], r["y"],
                                                  lambda p=aperio_path, x=x, y=y, c=crop:
                                                  read_rgb(p)[y:y + c, x:x + c]))
            derived_done = True

    if not methods:
        raise SystemExit(f"No usable eval_manifest.csv found under {eval_root} for tags {args.tags}")

    # ------------------------------------------------------------------
    # Classify + score each method
    # ------------------------------------------------------------------
    per_crop_path = out_dir / "per_crop.csv"
    per_crop_fh = open(per_crop_path, "w", newline="")
    pc_w = csv.writer(per_crop_fh)
    pc_w.writerow(["method", "slide", "frame", "x", "y",
                   "true_score", "pred_score", "correct", "prob_1", "prob_2", "prob_3"])

    summary_rows = []  # (method, scope, n, accuracy, macro_f1, macro_auc, robust_z, outlier)
    method_accuracy = {}  # for recovery_delta, filled after all methods scored

    for method, items in sorted(methods.items()):
        rgb_list, keys = [], []
        n_unlabelled = 0
        for slide, frame, x, y, loader in items:
            if (slide, frame) not in labels:
                n_unlabelled += 1
                continue
            keys.append((slide, frame, x, y))
            rgb_list.append(loader())
        if n_unlabelled:
            print(f"  {method}: {n_unlabelled} crops skipped (no atypia label for their frame).")
        if not rgb_list:
            print(f"  {method}: no labelled crops -- skipping.")
            continue

        probs = classify_batch(rgb_list)
        preds = probs.argmax(1) + 1  # back to 1-3 scale
        trues = np.array([labels[(s, f)] for s, f, _, _ in keys])

        by_slide = defaultdict(list)  # slide -> list[(true, pred, probs)]
        for (slide, frame, x, y), t, p, pr in zip(keys, trues, preds, probs):
            by_slide[slide].append((t, p, pr))
            pc_w.writerow([method, slide, frame, x, y, t, p, int(t == p),
                          round(float(pr[0]), 5), round(float(pr[1]), 5), round(float(pr[2]), 5)])

        slide_acc = {s: sum(t == p for t, p, _ in v) / len(v) for s, v in by_slide.items()}
        flags = flag_outliers(slide_acc, z_thresh=args.outlier_z)
        outliers = {s for s in flags if flags[s]["outlier"]}

        def pool(exclude=frozenset()):
            t_all, p_all, pr_all = [], [], []
            for s, v in by_slide.items():
                if s in exclude:
                    continue
                for t, p, pr in v:
                    t_all.append(t); p_all.append(p); pr_all.append(pr)
            if not t_all:
                return None
            t_all, p_all, pr_all = np.array(t_all), np.array(p_all), np.array(pr_all)
            acc = float((t_all == p_all).mean())
            f1 = f1_score(t_all, p_all, average="macro", zero_division=0)
            try:
                auc = roc_auc_score(t_all, pr_all, multi_class="ovr", average="macro", labels=[1, 2, 3])
            except ValueError:
                auc = None  # e.g. a class entirely absent from this scope's true labels
            return acc, f1, auc, len(t_all)

        allm = pool()
        method_accuracy[method] = allm[0]
        summary_rows.append([method, "ALL", allm[3], round(allm[0], 5), round(allm[1], 5),
                            round(allm[2], 5) if allm[2] is not None else "", "", ""])
        print(f"\n{method}: ALL n={allm[3]} acc={allm[0]:.4f} macroF1={allm[1]:.4f} "
              f"macroAUC={allm[2] if allm[2] is None else round(allm[2], 4)}")

        if outliers:
            cleanm = pool(exclude=outliers)
            if cleanm:
                summary_rows.append([method, "ALL_excl_outliers", cleanm[3], round(cleanm[0], 5),
                                    round(cleanm[1], 5),
                                    round(cleanm[2], 5) if cleanm[2] is not None else "", "", ""])

        for s in sorted(by_slide):
            v = by_slide[s]
            acc = sum(t == p for t, p, _ in v) / len(v)
            flag = "*** OUTLIER" if flags[s]["outlier"] else ""
            print(f"    {s}: n={len(v)} acc={acc:.3f}  z={flags[s]['z']:.2f}  {flag}")
            summary_rows.append([method, s, len(v), round(acc, 5), "", "",
                                round(flags[s]["z"], 3), flags[s]["outlier"]])

    per_crop_fh.close()

    # ------------------------------------------------------------------
    # Recovery delta = accuracy(method) - accuracy(raw_hamamatsu), on the ALL scope
    # ------------------------------------------------------------------
    base_acc = method_accuracy.get("raw_hamamatsu")
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "scope", "n", "accuracy", "macro_f1", "macro_auc",
                    "robust_z", "outlier", "recovery_delta"])
        for row in summary_rows:
            method, scope = row[0], row[1]
            delta = ""
            if scope == "ALL" and base_acc is not None and row[3] != "":
                delta = round(row[3] - base_acc, 5)
            w.writerow(row + [delta])

    print(f"\nPer-crop : {per_crop_path}")
    print(f"Summary  : {summary_path}")
    if base_acc is not None:
        print(f"raw_hamamatsu ALL accuracy = {base_acc:.4f} (recovery_delta baseline)")


if __name__ == "__main__":
    main()
