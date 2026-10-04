#!/usr/bin/env python3
"""
fuse_camelyon_patches.py -- P3-09: P1-16/P3-08 F3 fusion over CAMELYON17 patches.
See tickets/PHASE3-TICKETS.md P3-09.

Simpler than fuse_source_detail.py's manifest-driven paths: CAMELYON17 patches
need no registration or manifests, so the three inputs are matched by relative
path (centre_<i>_patient_<id>/<name>.png) across three directories:
  --raw       the upscaled 1024px source patches (A_raw)
  --vae-only  normalize_camelyon_sdxl.py --vae-only output (A_V)
  --pred      normalize_camelyon_sdxl.py correct-mode output (H_pred)
F3 = LAB(A_raw) + beta * G_sigma(LAB(H_pred) - LAB(A_V)), imported unchanged
from fuse_source_detail.py. Frozen config sigma=8, beta=0.50 (P3-08), no
re-tuning. Resumable (skips outputs that already exist).

Usage
-----
    python fuse_camelyon_patches.py --raw camelyon17_patches_1024 \
        --vae-only eval/p3_09_sdxl_vae_only --pred eval/p3_09_sdxl_correct \
        --out eval/p3_09_sdxl_fused
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

SIGMA, BETA = 8.0, 0.50


def parse_args():
    ap = argparse.ArgumentParser(description="P3-09: F3 fusion over CAMELYON17 patches.")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--vae-only", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--workers", type=int, default=4)
    return ap.parse_args()


def _fuse_one(job):
    from PIL import Image
    from fuse_source_detail import fuse_f3
    from score_outputs import read_rgb_png
    rel, raw_dir, vae_dir, pred_dir, out_dir, sigma, beta = job
    dest = Path(out_dir) / rel
    if dest.exists():
        return rel, "exists"
    v, p = Path(vae_dir) / rel, Path(pred_dir) / rel
    if not v.exists() or not p.exists():
        return rel, "missing"
    fused = fuse_f3(read_rgb_png(Path(raw_dir) / rel), read_rgb_png(p), read_rgb_png(v),
                    sigma=sigma, beta=beta)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(fused).save(dest)
    return rel, "ok"


def main():
    args = parse_args()
    raw = Path(args.raw)
    rels = sorted(str(p.relative_to(raw)) for p in raw.glob("centre_*_patient_*/*.png"))
    if not rels:
        raise SystemExit(f"No centre_*_patient_*/*.png under {raw}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(r, args.raw, args.vae_only, args.pred, args.out, args.sigma, args.beta) for r in rels]
    counts = {"ok": 0, "exists": 0, "missing": 0}
    with Pool(args.workers) as pool:
        for i, (rel, status) in enumerate(pool.imap_unordered(_fuse_one, jobs, chunksize=4), 1):
            counts[status] += 1
            if status == "missing":
                print(f"  MISSING vae-only/pred for {rel}")
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} {counts}")
    print(f"\nFused {counts['ok']} (+{counts['exists']} existed), {counts['missing']} missing, "
          f"sigma={args.sigma:g} beta={args.beta:.2f} -> {args.out}")
    if counts["missing"]:
        raise SystemExit(f"ABORT: {counts['missing']} patches missing from vae-only/pred dirs.")


if __name__ == "__main__":
    main()
