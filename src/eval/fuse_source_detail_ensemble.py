#!/usr/bin/env python3
"""
fuse_source_detail_ensemble.py -- combines P1-11's future-work idea #2
(genuine self-ensembling, tickets/PHASE1-TICKETS.md P1-11) with P1-16's
post-hoc colour-residual fusion (F3), to test whether the two techniques'
SSIM gains compound. Neither technique alone crosses classical baselines
from P1-11's own line (ensembling: 0.496 -> 0.543); P1-16's fusion alone
already does (0.496 -> 0.729) by acting on a DIFFERENT signal (raw source
detail), so this checks whether fusing the ENSEMBLED prediction (rather
than a single deterministic sample) pushes further still, stays flat, or
regresses (not guaranteed additive -- both techniques touch related
signal, so this is a genuine open question, not assumed).

Deliberately standalone -- does NOT edit fuse_source_detail.py (P1-16's
validated script) or infer_colour_source_ddim_inversion_ensemble.py (P1-11
future-work #2's own validated script). Imports `fuse_f3` directly from
fuse_source_detail.py (a pure, stateless function -- safe to reuse without
risking that script's own validated behaviour) rather than duplicating the
fusion maths.

Why a new script instead of fuse_source_detail.py's own --identity-manifest
path unmodified: that path keys the A_V (VAE-only reconstruction) lookup by
(crop_id, seed), expecting an exact seed match against the identity-mode
manifest's own seed column. The ensemble script's manifest uses the
sentinel seed value "ensemble" for its one averaged-output row per crop,
which will never match any of the identity manifest's real numeric seeds
(0, in the canonical full run) -- every ensemble row would silently
`n_skipped` under the unmodified path. This script instead keys A_V by
crop_id ONLY, the exact convention fuse_source_detail.py's own P3-08/SDXL
branch already uses, for the same underlying reason already documented
there: VAE-only reconstruction has no randomness, so the SAME A_V is
correct for every one of H_pred's seeds (or pseudo-seeds) for a given crop
-- not a new assumption, just applying that script's own existing
reasoning to this specific manifest shape.

Inputs
------
  --identity-manifest   A full-scale P1-11 --mode identity run's
                        eval_manifest.csv (e.g. eval/p1_10_ddim_inversion/
                        identity_full_heldout/eval_manifest.csv) -- supplies
                        A_raw (reference_path, any row, keyed by crop_id)
                        and A_V (method=="vae_only" row's output_path,
                        keyed by crop_id only).
  --ensemble-manifest   The self-ensembling script's own eval_manifest.csv
                        (e.g. eval/p1_11_ensemble/eta03_n3_full/
                        eval_manifest.csv) -- supplies H_pred from its
                        method=="ddim_inversion_ensemble" rows (source_mode
                        ending in "_ensemble") ONLY, never the individual
                        "_member" rows.

Usage
-----
    python fuse_source_detail_ensemble.py \
        --identity-manifest eval/p1_10_ddim_inversion/identity_full_heldout/eval_manifest.csv \
        --ensemble-manifest eval/p1_11_ensemble/eta03_n3_full/eval_manifest.csv \
        --sigma 8 --beta 0.50 \
        --out eval/p1_16_fusion_ensemble/f3_s8_b0.50_eta03

Dependencies: numpy, opencv-python-headless, scipy, scikit-image.
Requires score_outputs.py (read_rgb_png) and fuse_source_detail.py (fuse_f3)
on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

from PIL import Image

from fuse_source_detail import fuse_f3
from score_outputs import read_rgb_png


def parse_args():
    ap = argparse.ArgumentParser(
        description="Fuse P1-16's F3 colour-residual onto P1-11's ensembled (not single-sample) prediction.")
    ap.add_argument("--identity-manifest", required=True,
                    help="eval_manifest.csv from a full-scale P1-11 --mode identity run. "
                         "Supplies A_raw (reference/) and A_V (method=vae_only output), both "
                         "keyed by crop_id only.")
    ap.add_argument("--ensemble-manifest", required=True,
                    help="eval_manifest.csv from infer_colour_source_ddim_inversion_ensemble.py. "
                         "Supplies H_pred from its method=ddim_inversion_ensemble rows only.")
    ap.add_argument("--sigma", type=float, default=8.0, help="Gaussian blur sigma (px). Matches P1-16's frozen F3 config by default.")
    ap.add_argument("--beta", type=float, default=0.50, help="F3 colour-residual weight. Matches P1-16's frozen F3 config by default.")
    ap.add_argument("--out", required=True, help="Output dir for fused crops + manifest.")
    return ap.parse_args()


def load_rows(manifest_path):
    base = Path(manifest_path).parent
    with open(manifest_path, newline="") as fh:
        return base, list(csv.DictReader(fh))


def main():
    args = parse_args()

    id_base, id_rows = load_rows(args.identity_manifest)
    a_raw_by_crop = {r["crop_id"]: id_base / r["reference_path"] for r in id_rows}
    a_vae_by_crop = {r["crop_id"]: id_base / r["output_path"]
                     for r in id_rows if r.get("method") == "vae_only"}
    if not a_vae_by_crop:
        raise SystemExit(f"--identity-manifest {args.identity_manifest} has no method==vae_only "
                         f"rows -- wrong file?")

    ens_base, ens_rows = load_rows(args.ensemble_manifest)
    h_pred_rows = [r for r in ens_rows if r.get("method") == "ddim_inversion_ensemble"]
    if not h_pred_rows:
        raise SystemExit(f"--ensemble-manifest {args.ensemble_manifest} has no "
                         f"method==ddim_inversion_ensemble rows -- wrong file, or generated "
                         f"before the ensemble row was added?")

    out_dir = Path(args.out)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "eval_manifest.csv"
    man = open(man_path, "w", newline="")
    mw = csv.writer(man)
    mw.writerow(["strength", "seed", "slide", "frame", "x", "y",
                "output_path", "reference_path", "aperio_path"])

    config_tag = f"f3_s{args.sigma:g}_b{args.beta:.2f}_ensemble"
    n_out = n_skipped = 0
    for r in h_pred_rows:
        crop_id = r["crop_id"]
        if crop_id not in a_raw_by_crop or crop_id not in a_vae_by_crop:
            n_skipped += 1
            continue
        a_raw_rgb = read_rgb_png(a_raw_by_crop[crop_id])
        a_vae_rgb = read_rgb_png(a_vae_by_crop[crop_id])
        h_pred_rgb = read_rgb_png(ens_base / r["output_path"])

        fused = fuse_f3(a_raw_rgb, h_pred_rgb, a_vae_rgb, sigma=args.sigma, beta=args.beta)
        out_path = out_dir / "outputs" / f"{crop_id}_{config_tag}.png"
        Image.fromarray(fused).save(out_path)

        # Reference for scoring = the ensemble manifest's own reference_path
        # (real registered Hamamatsu) -- never read for fusion construction
        # above, only copied through for score_outputs.py to score against,
        # same convention as fuse_source_detail.py's own output.
        ref_src = ens_base / r["reference_path"]
        ref_dst = out_dir / "reference" / f"{crop_id}.png"
        ref_dst.parent.mkdir(parents=True, exist_ok=True)
        if not ref_dst.exists():
            shutil.copyfile(ref_src, ref_dst)

        mw.writerow([config_tag, "ensemble", r["slide"], r["frame"], r["x"], r["y"],
                    os.path.relpath(out_path, out_dir),
                    os.path.relpath(ref_dst, out_dir),
                    r["aperio_path"]])
        man.flush()
        n_out += 1

    man.close()
    if n_skipped:
        print(f"WARNING: skipped {n_skipped} rows with no matching identity-manifest crop_id -- "
             f"identity and ensemble manifests may not cover the same crops.")
    print(f"\nWrote {n_out} fused crops (method=f3, config={config_tag}).")
    print(f"Manifest: {man_path}")
    print("Next: score with score_outputs.py (unmodified) -- schema matches infer_colour_translation.py exactly.")


if __name__ == "__main__":
    main()
