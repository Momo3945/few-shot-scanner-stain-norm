#!/usr/bin/env python3
"""
fuse_source_detail.py -- P1-16: raw-source-detail / learned colour-residual
fusion. See tickets/PHASE1-TICKETS.md P1-16 and
tickets/P1-16_source_detail_colour_residual_fusion.md for the full spec.

Motivation (ticket): P1-11 (DDIM inversion) strongly improves scanner-colour
recovery but remains below classical methods on SSIM, because it resynthesises
the ENTIRE crop through the VAE + denoising path rather than recolouring the
original pixels. This script tests whether the learned colour transformation
can be projected back onto the untouched raw Aperio source, so diffusion
supplies only the A->H colour/style change while the source supplies the fine
tissue structure -- deterministic post-processing on ALREADY-GENERATED
outputs, no retraining, no new diffusion run (per the ticket's "No Training
Initially" section).

INTERFACE DEVIATION FROM THE TICKET (documented, not silent): the ticket
suggests flat --source-dir/--prediction-dir/--vae-reconstruction-dir
directories matched by filename. This project's actual P1-11 outputs
(infer_colour_source_ddim_inversion.py) are NOT flat per-role directories --
one eval_manifest.csv maps crop_id+seed to output_path/reference_path/method
across MULTIPLE roles in a single run dir (vae_only / img2img_baseline /
ddim_inversion in --mode identity; ddim_inversion in --mode translate). A
filename-matching join would be fragile and could silently pair the wrong
crops. Instead this script is manifest-driven: it reads TWO existing P1-11
eval_manifest.csv files (one --mode identity run, one --mode translate run,
both against the SAME internal-validation crops/seeds) and joins their rows
by (crop_id, seed) -- the exact columns P1-11's manifest already carries for
this purpose.

Inputs, each read ONLY from an existing P1-11 identity/translate run
(generate them first with infer_colour_source_ddim_inversion.py if missing --
this script itself runs no diffusion):
  A_raw  = the identity-mode run's reference/<crop_id>.png (identity mode's
           reference IS the untouched source crop -- see that script's own
           `ref_img = c["src"] if args.mode == "identity" else c["ref"]`).
  A_V    = the identity-mode run's method="vae_only" row's output_path (pure
           VAE encode/decode of A_raw, no UNet).
  H_pred = the translate-mode run's method="ddim_inversion", source_mode=
           "correct" row's output_path (the actual A->H learned prediction).
Real Hamamatsu ground truth is NEVER read by this script for fusion
construction (only usable afterward, by the separate scoring step, per the
ticket's "Hard Guardrail -- No Target Leakage").

LAB convention note: uses skimage.color.rgb2lab/lab2rgb (real CIELAB range,
L in [0,100], a/b roughly [-128,127], float) for ALL fusion arithmetic --
NOT metrics.py's lab_wasserstein(), which uses cv2.cvtColor(...,
COLOR_RGB2LAB) (uint8-quantized, L/a/b all packed into [0,255], OpenCV's
storage convention; ciede2000_stats() in that same file already uses
skimage's rgb2lab, so this project already has both conventions in use).
Float CIELAB avoids the banding/clipping a uint8 round-trip would introduce
mid-computation (Gaussian blur, channel arithmetic, addition of a residual).
Scoring afterward goes through the existing score_outputs.py unchanged (RGB
in/out only -- no LAB convention crosses that boundary).

Three variants (ticket's F1/F2/F3, LAB channels as (L, a, b)):
  F1  Source-luminance diagnostic: F1 = (L_A, a_Hpred, b_Hpred). NOT the
      final method -- how much of the SSIM penalty is regenerated
      luminance/detail vs the learned chromatic transform alone.
  F2  High-frequency luminance reinjection: low-pass H_pred's luminance,
      high-pass A_raw's luminance, recombine -- L_F = G_sigma(L_Hpred) +
      alpha * (L_A - G_sigma(L_A)); F2 = (L_F, a_Hpred, b_Hpred).
  F3  VAE-relative learned colour-residual projection (primary): estimate
      the learned change relative to the model's OWN reconstruction of the
      source (cancels shared VAE reconstruction characteristics before
      transferring the change onto the untouched source) --
      Delta = LAB(H_pred) - LAB(A_V), Delta_colour = G_sigma(Delta),
      F3 = LAB(A_raw) + beta * Delta_colour (all 3 channels).

Usage
-----
    python fuse_source_detail.py \
        --identity-manifest <out>/p1_11_identity/eval_manifest.csv \
        --translate-manifest <out>/p1_11_translate_correct/eval_manifest.csv \
        --method f3 --sigma 8 --beta 0.75 \
        --out eval/p1_16_fusion/f3_sigma8_beta0.75

Dependencies: numpy, opencv-python-headless, scipy, scikit-image.
Requires score_outputs.py (read_rgb_png) on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import lab2rgb, rgb2lab

from score_outputs import read_rgb_png


def to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    return rgb2lab(rgb_uint8.astype(np.float64) / 255.0)


def to_rgb(lab: np.ndarray) -> np.ndarray:
    rgb = lab2rgb(lab)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def blur(channel: np.ndarray, sigma: float) -> np.ndarray:
    return channel if sigma <= 0 else gaussian_filter(channel, sigma=sigma)


def fuse_f1(a_raw_rgb: np.ndarray, h_pred_rgb: np.ndarray, **_) -> np.ndarray:
    lab_a, lab_h = to_lab(a_raw_rgb), to_lab(h_pred_rgb)
    out = lab_h.copy()
    out[..., 0] = lab_a[..., 0]
    return to_rgb(out)


def fuse_f2(a_raw_rgb: np.ndarray, h_pred_rgb: np.ndarray, *, sigma: float, alpha: float, **_) -> np.ndarray:
    lab_a, lab_h = to_lab(a_raw_rgb), to_lab(h_pred_rgb)
    L_A, L_H = lab_a[..., 0], lab_h[..., 0]
    L_A_hf = L_A - blur(L_A, sigma)
    L_H_lf = blur(L_H, sigma)
    out = lab_h.copy()
    out[..., 0] = L_H_lf + alpha * L_A_hf
    return to_rgb(out)


def fuse_f3(a_raw_rgb: np.ndarray, h_pred_rgb: np.ndarray, a_vae_rgb: np.ndarray,
           *, sigma: float, beta: float, **_) -> np.ndarray:
    lab_a_raw, lab_h, lab_a_vae = to_lab(a_raw_rgb), to_lab(h_pred_rgb), to_lab(a_vae_rgb)
    delta = lab_h - lab_a_vae
    delta_colour = np.stack([blur(delta[..., c], sigma) for c in range(3)], axis=-1)
    return to_rgb(lab_a_raw + beta * delta_colour)


FUSERS = {"f1": fuse_f1, "f2": fuse_f2, "f3": fuse_f3}


def parse_args():
    ap = argparse.ArgumentParser(description="P1-16: raw-source-detail / learned colour-residual fusion.")
    ap.add_argument("--identity-manifest", required=True,
                    help="eval_manifest.csv from a P1-11 --mode identity run (same internal-"
                         "validation crops/seeds as --translate-manifest). Supplies A_raw "
                         "(reference/) and A_V (method=vae_only output).")
    ap.add_argument("--translate-manifest", required=True,
                    help="eval_manifest.csv from a P1-11 --mode translate --source-mode correct "
                         "run. Supplies H_pred (method=ddim_inversion output). Its own "
                         "reference_path (real Hamamatsu) is NOT read by this script.")
    ap.add_argument("--method", choices=["f1", "f2", "f3"], required=True)
    ap.add_argument("--sigma", type=float, default=4.0, help="Gaussian blur sigma (px). F2/F3.")
    ap.add_argument("--alpha", type=float, default=1.0, help="F2 high-frequency reinjection weight.")
    ap.add_argument("--beta", type=float, default=1.0, help="F3 colour-residual weight.")
    ap.add_argument("--out", required=True, help="Output dir for fused crops + manifest.")
    return ap.parse_args()


def load_rows(manifest_path):
    base = Path(manifest_path).parent
    with open(manifest_path, newline="") as fh:
        return base, list(csv.DictReader(fh))


def main():
    args = parse_args()
    fuse = FUSERS[args.method]
    if args.method == "f2" and not (0.0 <= args.alpha):
        raise SystemExit("--alpha must be >= 0 for --method f2.")
    if args.method == "f3" and not (0.0 <= args.beta):
        raise SystemExit("--beta must be >= 0 for --method f3.")

    id_base, id_rows = load_rows(args.identity_manifest)
    tr_base, tr_rows = load_rows(args.translate_manifest)

    # A_raw + A_V: any row in the identity manifest carries the shared
    # reference_path (= A_raw) for its crop_id; only method=="vae_only" rows
    # carry A_V.
    a_raw_by_crop = {r["crop_id"]: id_base / r["reference_path"] for r in id_rows}
    a_vae_by_key = {(r["crop_id"], r["seed"]): id_base / r["output_path"]
                    for r in id_rows if r["method"] == "vae_only"}

    # H_pred: translate-mode rows, method==ddim_inversion, source_mode==correct
    # (the primary A->H prediction -- zero/shuffled ablation rows, if present
    # in the same manifest, are deliberately excluded).
    h_pred_rows = [r for r in tr_rows if r["method"] == "ddim_inversion" and r["source_mode"] == "correct"]
    if not h_pred_rows:
        raise SystemExit(
            "No method=ddim_inversion, source_mode=correct rows found in --translate-manifest -- "
            "wrong file, or the run used a different --source-mode.")

    out_dir = Path(args.out)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "eval_manifest.csv"
    man = open(man_path, "w", newline="")
    mw = csv.writer(man)
    mw.writerow(["strength", "seed", "slide", "frame", "x", "y",
                "output_path", "reference_path", "aperio_path"])

    config_tag = {"f1": "f1", "f2": f"f2_s{args.sigma:g}_a{args.alpha:.2f}",
                  "f3": f"f3_s{args.sigma:g}_b{args.beta:.2f}"}[args.method]

    n_out = n_skipped = 0
    for r in h_pred_rows:
        crop_id, seed = r["crop_id"], r["seed"]
        if crop_id not in a_raw_by_crop:
            n_skipped += 1
            continue
        a_raw_path = a_raw_by_crop[crop_id]
        h_pred_path = tr_base / r["output_path"]
        a_raw_rgb = read_rgb_png(a_raw_path)
        h_pred_rgb = read_rgb_png(h_pred_path)

        kwargs = {"sigma": args.sigma, "alpha": args.alpha, "beta": args.beta}
        if args.method == "f3":
            key = (crop_id, seed)
            if key not in a_vae_by_key:
                n_skipped += 1
                continue
            kwargs["a_vae_rgb"] = read_rgb_png(a_vae_by_key[key])

        fused = fuse(a_raw_rgb, h_pred_rgb, **kwargs)
        out_path = out_dir / "outputs" / f"{crop_id}_seed{seed}_{config_tag}.png"
        from PIL import Image
        Image.fromarray(fused).save(out_path)

        # Reference for scoring = the TRANSLATE manifest's own reference_path
        # (real registered Hamamatsu) -- never read for fusion construction
        # above, only copied through here so score_outputs.py can score
        # against it afterward, same as every other eval dir in this project.
        ref_src = tr_base / r["reference_path"]
        ref_dst = out_dir / "reference" / f"{crop_id}.png"
        ref_dst.parent.mkdir(parents=True, exist_ok=True)
        if not ref_dst.exists():
            import shutil
            shutil.copyfile(ref_src, ref_dst)

        mw.writerow([config_tag, seed, r["slide"], r["frame"], r["x"], r["y"],
                    os.path.relpath(out_path, out_dir),
                    os.path.relpath(ref_dst, out_dir),
                    r["aperio_path"]])
        n_out += 1

    man.close()
    if n_skipped:
        print(f"WARNING: skipped {n_skipped} rows with no matching identity-manifest crop_id/"
              f"vae_only row -- identity and translate manifests may not cover the same crops/seeds.")
    print(f"\nWrote {n_out} fused crops (method={args.method}, config={config_tag}).")
    print(f"Manifest: {man_path}")
    print("Next: score with score_outputs.py (unmodified) -- schema matches infer_colour_translation.py exactly.")


if __name__ == "__main__":
    main()
