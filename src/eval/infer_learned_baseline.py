#!/usr/bin/env python3
"""
infer_learned_baseline.py -- run a learned stain-normalisation baseline (P2-11b:
StainNet / ParamNet / StainGAN) over the held-out set, writing an eval_manifest.csv in
the same schema as infer_baseline.py / infer_colour_lora.py, so score_outputs.py (and
the atypia/Lizard/CAMELYON scorers) score it with ZERO changes.

Mirrors infer_baseline.py's held-out loop deliberately (same registration, crop grid,
tissue/eccentricity filters, max-crops-per-frame) so every method is scored on the
identical crops. Networks are fully convolutional -> full 512 px crop in one pass.

Usage
-----
    python infer_learned_baseline.py --method stainnet --ckpt baselines/a2h_stainnet/stainnet_a2h.pt \
        --root mitos_heldout --heldout pairs/heldout_frames.csv --out eval/stainnet [--direction A2H]

Requires registration.py, metrics.py (siblings) and src/baselines/models.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ -> `baselines` package
from baselines.models import ParamNet, ResnetGenerator, StainNet  # noqa: E402
from registration import read_rgb, register_h_to_a  # noqa: E402
from metrics import grid_offsets, tissue_fraction  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Run a learned stain-norm baseline (P2-11b).")
    ap.add_argument("--method", choices=["stainnet", "paramnet", "staingan"], required=True)
    ap.add_argument("--ckpt", required=True, help="<method>_<direction>.pt from train_learned_baseline.py.")
    ap.add_argument("--root", required=True, help="Dataset root (heldout paths are relative to this).")
    ap.add_argument("--heldout", required=True, help="heldout_frames.csv.")
    ap.add_argument("--direction", choices=["A2H", "H2A"], default="A2H")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help="Max held-out frames (0 = all).")
    ap.add_argument("--max-crops-per-frame", type=int, default=4, help="0 = all tissue crops.")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu (StainNet is fine on CPU).")
    return ap.parse_args()


def build_model(method: str, ck: dict) -> nn.Module:
    arch = ck["arch"]
    if method == "stainnet":
        model = StainNet(3, 3, arch["n_layer"], arch["n_channel"])
    elif method == "paramnet":
        model = ParamNet(arch["resample_size"], arch["channels"], arch["layers"])
    else:
        model = ResnetGenerator(3, 3, 64, nn.InstanceNorm2d, n_blocks=arch["n_blocks"])
    model.load_state_dict(ck["model"])
    return model.eval()


def make_transform(model: nn.Module, device: torch.device):
    @torch.no_grad()
    def transform(rgb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)
        x = (x - 127.5) / 127.5
        y = model(x).clamp(-1, 1) * 127.5 + 127.5
        return y.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).byte().cpu().numpy()
    return transform


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu" and args.method != "stainnet":
        raise SystemExit(f"ABORT: {args.method} needs a GPU (torch.cuda.is_available() is "
                         f"{torch.cuda.is_available()}); refusing to CPU-crawl.")

    ck = torch.load(args.ckpt, map_location="cpu")
    if ck["method"] != args.method or ck["direction"] != args.direction:
        raise SystemExit(f"Checkpoint is {ck['method']}/{ck['direction']}, "
                         f"asked for {args.method}/{args.direction}.")
    model = build_model(args.method, ck).to(device)
    transform = make_transform(model, device)
    print(f"Method={args.method}  Direction={args.direction}  Ckpt={args.ckpt}  "
          f"steps={ck['steps']}  device={device}", flush=True)

    out_dir = Path(args.out)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)

    with open(args.heldout, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    for r in rows:  # heldout_frames.csv has Windows backslashes
        r["aperio_path"] = r["aperio_path"].replace("\\", "/")
        r["hamamatsu_path"] = r["hamamatsu_path"].replace("\\", "/")

    man = open(out_dir / "eval_manifest.csv", "w", newline="")
    mw = csv.writer(man)
    mw.writerow(["strength", "slide", "frame", "x", "y", "output_path", "reference_path", "aperio_path"])

    n_out = 0
    for r in rows:
        a_rgb = read_rgb(Path(args.root) / r["aperio_path"])
        h_rgb = read_rgb(Path(args.root) / r["hamamatsu_path"])
        reg = register_h_to_a(a_rgb, h_rgb, ecc_min=args.ecc_min)
        h_reg = reg.h_registered_rgb
        H, W = a_rgb.shape[:2]
        src_frame, ref_frame = (a_rgb, h_reg) if args.direction == "A2H" else (h_reg, a_rgb)

        crops_done = 0
        for y in grid_offsets(H, args.crop):
            for x in grid_offsets(W, args.crop):
                if args.max_crops_per_frame and crops_done >= args.max_crops_per_frame:
                    break
                src_c = src_frame[y:y + args.crop, x:x + args.crop]
                ref_c = ref_frame[y:y + args.crop, x:x + args.crop]
                if (ref_c.max(2) < 6).mean() > 0.10:
                    continue
                if tissue_fraction(src_c) < args.tissue_thresh:
                    continue

                tag = f"{r['aperio_slide']}_{r['frame_id']}_x{x}_y{y}"
                ref_path = out_dir / "reference" / f"{tag}.png"
                Image.fromarray(ref_c).save(ref_path)
                out_path = out_dir / "outputs" / f"{tag}.png"
                Image.fromarray(transform(src_c)).save(out_path)
                mw.writerow(["na", r["aperio_slide"], r["frame_id"], x, y,
                             os.path.relpath(out_path, out_dir),
                             os.path.relpath(ref_path, out_dir),
                             r["aperio_path"]])
                n_out += 1
                crops_done += 1
        print(f"  {r['aperio_slide']}_{r['frame_id']}: {crops_done} crops", flush=True)

    man.close()
    with open(out_dir / "run_metadata.json", "w") as fh:
        json.dump({"script": "infer_learned_baseline.py", "method": args.method,
                   "direction": args.direction, "ckpt": args.ckpt, "train_steps": ck["steps"]},
                  fh, indent=2)
    print(f"\nWrote {n_out} output crops (method={args.method}).", flush=True)
    print(f"Manifest: {out_dir / 'eval_manifest.csv'}")


if __name__ == "__main__":
    main()
