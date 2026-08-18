#!/usr/bin/env python3
"""
train_atypia_classifier.py -- fine-tune a standard diagnostic classifier (ResNet18)
on raw Aperio x20 atypia-score patches (P2-09, Clinical Utility).

Per the proposal (sec:experiments "Clinical Utility"): a standard diagnostic
classifier trained on raw Aperio (x20) patches, later evaluated (inference only,
this script is not re-run per method) on raw Hamamatsu + every baseline/diffusion
output (score_atypia_classifier.py). Primary metric: atypia score accuracy at x20.

Each 512x512 tissue crop (same grid_offsets/tissue_fraction tiling convention as
infer_colour_lora.py/extract_pairs.py -- kept apples-to-apples with every other
crop in this project) inherits its parent frame's atypia score as a WEAK,
frame-level label -- MITOS-ATYPIA-14 assesses atypia per ROI, not per sub-tile, so
this is the standard, defensible assumption in histopathology patch classification
(same one implicit in treating a whole frame as one diagnostic unit). This is also
what lets score_atypia_classifier.py skip any frame-level aggregation step: it can
score each existing eval crop directly against its frame's true label.

Design decision not in the original P2-09 plan, added here: the atypia-score
distribution is heavily imbalanced (training manifest: roughly 1:23, 2:222, 3:52
frames in this dataset) -- CrossEntropyLoss is class-weighted (inverse frequency)
to avoid a degenerate "always predict 2" classifier, which would still report a
misleadingly high raw accuracy.

Train/val split is BY SLIDE, not by crop, to avoid leaking crops from the same
frame across the split (default: the two alphabetically-last training slides are
held out for validation unless --val-slides is given explicitly).

Usage
-----
    # smoke test first:
    python train_atypia_classifier.py --manifest pairs/atypia_train_manifest.csv \
        --root /datasets/mhoosen/stain-norm/mitos_atypia_train_aperio \
        --output-dir /datasets/mhoosen/stain-norm/classifier/atypia_r18/smoke --smoke

    # real run:
    python train_atypia_classifier.py --manifest pairs/atypia_train_manifest.csv \
        --root /datasets/mhoosen/stain-norm/mitos_atypia_train_aperio \
        --output-dir /datasets/mhoosen/stain-norm/classifier/atypia_r18

Dependencies: torch, torchvision, numpy, Pillow, tifffile, opencv-python-headless.
Requires registration.py (src/eval/) on the path for read_rgb (TIFF frame loading).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# registration.py lives in src/eval, a sibling of this file's parent (src/train)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))


def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune a ResNet18 atypia-score classifier.")
    ap.add_argument("--manifest", required=True, help="atypia_manifest.csv (training root).")
    ap.add_argument("--root", required=True, help="Aperio root the manifest's image_path is relative to.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--val-slides", default="",
                    help="Comma-separated slide names to hold out for validation. "
                         "Empty -> auto-pick the 2 alphabetically-last slides in the manifest.")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--max-crops-per-frame", type=int, default=4, help="0 = all tissue crops.")
    ap.add_argument("--resize", type=int, default=224, help="Resize crop to this before ResNet18.")
    ap.add_argument("--train-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true",
                    help="5-step dry run: overrides train-steps/save-every for a quick env+loop check.")
    return ap.parse_args()


def load_manifest(path):
    with open(path, newline="") as fh:
        return [dict(r, atypia_score=int(r["atypia_score"])) for r in csv.DictReader(fh)]


def main():
    args = parse_args()
    if args.smoke:
        args.train_steps = 5
        args.save_every = 5
        args.log_every = 1

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision.models import resnet18, ResNet18_Weights
    from PIL import Image

    from registration import read_rgb

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device found -- training on CPU will be unusably slow.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)

    # ------------------------------------------------------------------
    # Tissue + crop helpers -- identical convention to infer_colour_lora.py
    # ------------------------------------------------------------------
    def tissue_fraction(rgb):
        a = rgb.astype(np.int16); mx = a.max(2); mn = a.min(2)
        return float(((mx < 235) & ((mx - mn) > 12)).mean())

    def grid_offsets(length, crop):
        if length <= crop:
            return [0]
        offs = list(range(0, length - crop + 1, crop))
        if offs[-1] != length - crop:
            offs.append(length - crop)
        return sorted(set(offs))

    # ------------------------------------------------------------------
    # Manifest -> train/val split by slide -> tissue crops
    # ------------------------------------------------------------------
    rows = load_manifest(args.manifest)
    if not rows:
        raise SystemExit(f"Empty manifest: {args.manifest}")
    slides = sorted({r["slide"] for r in rows})
    if args.val_slides:
        val_slides = {s.strip() for s in args.val_slides.split(",") if s.strip()}
    else:
        val_slides = set(slides[-2:]) if len(slides) > 2 else set()
    train_rows = [r for r in rows if r["slide"] not in val_slides]
    val_rows = [r for r in rows if r["slide"] in val_slides]
    print(f"Slides: {len(slides)} total, val={sorted(val_slides)} "
          f"({len(val_rows)} frames), train={len(slides) - len(val_slides)} slides "
          f"({len(train_rows)} frames).")

    def crops_for(frame_rows, tag):
        items = []  # (image_path, x, y, label)
        for r in frame_rows:
            frame_path = root / r["image_path"]
            rgb = read_rgb(frame_path)
            H, W = rgb.shape[:2]
            n = 0
            for y in grid_offsets(H, args.crop):
                for x in grid_offsets(W, args.crop):
                    if args.max_crops_per_frame and n >= args.max_crops_per_frame:
                        break
                    crop = rgb[y:y + args.crop, x:x + args.crop]
                    if tissue_fraction(crop) < args.tissue_thresh:
                        continue
                    items.append((str(frame_path), x, y, r["atypia_score"] - 1))
                    n += 1
        print(f"  {tag}: {len(frame_rows)} frames -> {len(items)} tissue crops")
        return items

    train_items = crops_for(train_rows, "train")
    val_items = crops_for(val_rows, "val") if val_rows else []
    if not train_items:
        raise SystemExit("No training tissue crops selected -- check --tissue-thresh / manifest paths.")

    class_counts = Counter(it[3] for it in train_items)
    print(f"Train class distribution (0/1/2 = score 1/2/3): {dict(sorted(class_counts.items()))}")
    weight = torch.tensor(
        [1.0 / max(class_counts.get(c, 1), 1) for c in range(3)], dtype=torch.float32)
    weight = weight / weight.sum() * 3.0  # keep loss scale comparable to unweighted CE

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    class CropDataset(Dataset):
        # frame images are cached per-worker (small manifest, frames re-read across
        # crops of the same frame otherwise) -- simple dict cache keyed by path.
        _cache: dict = {}

        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, x, y, label = self.items[i]
            rgb = self._cache.get(path)
            if rgb is None:
                rgb = read_rgb(path)
                if len(self._cache) < 64:  # small bound, avoid unbounded worker memory growth
                    self._cache[path] = rgb
            crop = rgb[y:y + args.crop, x:x + args.crop]
            img = Image.fromarray(crop).resize((args.resize, args.resize), Image.LANCZOS)
            arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
            arr = (arr - mean) / std
            return arr, label

    train_loader = DataLoader(CropDataset(train_items), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              drop_last=(len(train_items) >= args.batch_size), pin_memory=True)
    val_loader = (DataLoader(CropDataset(val_items), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)
                 if val_items else None)

    # ------------------------------------------------------------------
    # Model: ResNet18, ImageNet-pretrained, 3-class head
    # ------------------------------------------------------------------
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.to(device)
    weight = weight.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def evaluate(loader):
        if loader is None:
            return None
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()
        model.train()
        return correct / total if total else None

    # ------------------------------------------------------------------
    # Training loop (step-based, matching train_colour_lora.py's convention)
    # ------------------------------------------------------------------
    log_path = out_dir / "loss_log.csv"
    log_fh = open(log_path, "w", newline="")
    log_w = csv.writer(log_fh); log_w.writerow(["step", "loss", "train_acc", "val_acc", "sec"])

    print(f"Training for {args.train_steps} steps (batch {args.batch_size}, lr {args.lr}) ...")
    model.train()
    step = 0
    t0 = time.time()
    running_loss = running_correct = running_n = 0
    best_val = -1.0
    done = False
    while not done:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            running_loss += loss.item()
            running_correct += (logits.argmax(1) == y).sum().item()
            running_n += y.numel()

            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                train_acc = running_correct / running_n if running_n else 0.0
                running_loss = running_correct = running_n = 0
                sec = time.time() - t0
                val_acc = ""
                print(f"  step {step:5d}/{args.train_steps}  loss {avg_loss:.4f}  "
                      f"train_acc {train_acc:.3f}  ({sec:.1f}s)")
                log_w.writerow([step, f"{avg_loss:.6f}", f"{train_acc:.4f}", val_acc, f"{sec:.1f}"])
                log_fh.flush()

            if step % args.save_every == 0 or step >= args.train_steps:
                val_acc = evaluate(val_loader)
                if val_acc is not None:
                    print(f"    val_acc {val_acc:.3f}")
                    log_w.writerow([step, "", "", f"{val_acc:.4f}", f"{time.time()-t0:.1f}"])
                    log_fh.flush()
                ckpt_name = "final.pt" if step >= args.train_steps else f"checkpoint-{step}.pt"
                torch.save({"model": model.state_dict(), "step": step, "val_acc": val_acc},
                          out_dir / ckpt_name)
                print(f"  saved -> {out_dir / ckpt_name}")
                if val_acc is not None and val_acc >= best_val:
                    best_val = val_acc
                    torch.save({"model": model.state_dict(), "step": step, "val_acc": val_acc},
                              out_dir / "best.pt")
                    print(f"  new best -> {out_dir / 'best.pt'} (val_acc {val_acc:.3f})")

            if step >= args.train_steps:
                done = True
                break

    log_fh.close()
    if val_loader is None:
        # no val split (e.g. --smoke on a tiny manifest) -- final.pt is also the best
        # checkpoint we have, so downstream scoring can always rely on best.pt existing.
        import shutil
        shutil.copy(out_dir / "final.pt", out_dir / "best.pt")

    with open(out_dir / "training_config.json", "w") as fh:
        json.dump(vars(args) | {"val_slides": sorted(val_slides), "n_train_crops": len(train_items),
                                "n_val_crops": len(val_items), "best_val_acc": best_val,
                                "class_counts": dict(class_counts)}, fh, indent=2)
    print(f"\nDone. Best checkpoint: {out_dir / 'best.pt'} (val_acc {best_val:.3f})")


if __name__ == "__main__":
    main()
