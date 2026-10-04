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

Inputs -- two supported conventions, controlled by which of
--identity-manifest/--vae-only-manifest is passed (this script itself runs
no diffusion in either case):

  SD1.5/P1-11 convention (--identity-manifest, generate first with
  infer_colour_source_ddim_inversion.py if missing):
    A_raw  = the identity-mode run's reference/<crop_id>.png (identity
             mode's reference IS the untouched source crop -- see that
             script's own `ref_img = c["src"] if args.mode == "identity"
             else c["ref"]`).
    A_V    = the identity-mode run's method="vae_only" row's output_path
             (pure VAE encode/decode of A_raw, no UNet).

  P3-08/SDXL convention (--vae-only-manifest, added for backbones with no
  "identity mode" -- e.g. infer_colour_translation_sdxl.py):
    A_raw  = read directly from --translate-manifest's own aperio_path
             column (always the true untouched Aperio crop, direction-
             invariant -- no identity run needed at all).
    A_V    = a separate plain --vae-only run's (no checkpoint needed)
             source_mode=="vae_only" row's output_path.

  Both conventions:
  H_pred = the translate-mode run's method="ddim_inversion" (or no
           "method" column at all -- some manifests in this project use a
           simpler schema where every row in a single-source-mode run
           already IS the prediction), source_mode="correct" row's
           output_path (the actual A->H learned prediction).
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
    # SD1.5/P1-11 convention:
    python fuse_source_detail.py \
        --identity-manifest <out>/p1_11_identity/eval_manifest.csv \
        --translate-manifest <out>/p1_11_translate_correct/eval_manifest.csv \
        --method f3 --sigma 8 --beta 0.75 \
        --out eval/p1_16_fusion/f3_sigma8_beta0.75

    # P3-08/SDXL convention:
    python fuse_source_detail.py \
        --vae-only-manifest <out>/p3_07c_vae_only/eval_manifest.csv \
        --translate-manifest <out>/p3_07c_full_heldout/eval_manifest.csv \
        --method f3 --sigma 8 --beta 0.50 \
        --out eval/p3_08_fusion/f3_s8_b0.50

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
    ap.add_argument("--identity-manifest", default=None,
                    help="eval_manifest.csv from a P1-11 --mode identity run (same internal-"
                         "validation crops/seeds as --translate-manifest). Supplies A_raw "
                         "(reference/) and A_V (method=vae_only output). SD1.5/P1-11 convention -- "
                         "mutually exclusive with --vae-only-manifest (P3-08/SDXL convention, see "
                         "that flag's help).")
    ap.add_argument("--vae-only-manifest", default=None,
                    help="P3-08/SDXL alternative to --identity-manifest: SDXL's "
                         "infer_colour_translation_sdxl.py has no 'identity mode' and never saves "
                         "the raw Aperio crop as its own file (only reference_path=Hamamatsu and "
                         "output_path=generated/reconstructed are saved; aperio_path is a "
                         "provenance pointer to the ORIGINAL full-resolution TIFF frame, not a "
                         "loadable crop PNG -- confirmed empirically, job 61392). When this flag is "
                         "given instead of --identity-manifest: A_raw is RE-DERIVED directly "
                         "(registration + grid-crop, same routine as infer_colour_translation_sdxl.py/"
                         "benchmark_vae_reconstruction.py's own --stage heldout gathering -- requires "
                         "--aperio-root/--heldout-csv/--crop below), and A_V is read from THIS "
                         "manifest's source_mode==vae_only rows (a plain infer_colour_translation_"
                         "sdxl.py --vae-only run, no checkpoint needed).")
    ap.add_argument("--aperio-root", default=None,
                    help="P3-08/SDXL only: mitos_heldout dataset root, for re-deriving A_raw crops. "
                         "Required with --vae-only-manifest.")
    ap.add_argument("--direction", choices=["A2H", "H2A"], default="A2H",
                    help="P3-08/SDXL only: which direction's raw source crop to re-derive -- must "
                         "match the --direction used for both --translate-manifest and "
                         "--vae-only-manifest's own infer_colour_translation_sdxl.py runs. Default "
                         "A2H preserves every existing invocation's behaviour unchanged.")
    ap.add_argument("--heldout-csv", default=None,
                    help="P3-08/SDXL only: heldout_frames.csv. Required with --vae-only-manifest.")
    ap.add_argument("--crop", type=int, default=1024,
                    help="P3-08/SDXL only: crop size, must match the translate/vae-only runs' own "
                         "resolution (P3-07/P3-07c use 1024).")
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--translate-manifest", required=True,
                    help="eval_manifest.csv from a P1-11 --mode translate --source-mode correct "
                         "run (or, for P3-08/SDXL, an infer_colour_translation_sdxl.py "
                         "--source-mode correct run -- same manifest schema). Supplies H_pred. Its "
                         "own reference_path (real Hamamatsu) is NOT read by this script.")
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


def _rederive_aperio_crops(root, heldout_csv, crop, tissue_thresh, ecc_min, direction="A2H"):
    """P3-08/SDXL only: reproduce infer_colour_translation_sdxl.py's own
    crop selection exactly (registration + grid tiling + both its filters),
    since that script never saves the raw source crop to disk. Returns
    {tag_id: rgb_uint8_array}, tag_id format f"{slide}_{frame}_x{x}_y{y}"
    matching that script's own tag_id exactly (direction-invariant --
    always keyed by the Aperio slide/frame id, per that script's own
    `crops.append({"slide": r["aperio_slide"], "frame": r["frame_id"], ...})`).

    direction="A2H" (default, matches every P3-06/P3-07/P3-07c run so far):
    src_frame is the UNREGISTERED raw Aperio frame, per that script's own
    `src_frame, ref_frame = (a_rgb, h_reg) if direction == "A2H" ...`.
    direction="H2A" (P3-07 H2A/P3-08 H2A extension): src_frame is the
    REGISTERED Hamamatsu frame (h_reg) instead -- the mirror image of the
    same conditional in that script.
    """
    from registration import read_rgb, register_h_to_a

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

    with open(heldout_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["aperio_path"] = r["aperio_path"].replace("\\", "/")
        r["hamamatsu_path"] = r["hamamatsu_path"].replace("\\", "/")

    out = {}
    for r in rows:
        a_rgb = read_rgb(Path(root) / r["aperio_path"])
        h_rgb = read_rgb(Path(root) / r["hamamatsu_path"])
        reg = register_h_to_a(a_rgb, h_rgb, ecc_min=ecc_min)
        h_reg = reg.h_registered_rgb
        H, W = a_rgb.shape[:2]
        src_frame, ref_frame = (a_rgb, h_reg) if direction == "A2H" else (h_reg, a_rgb)

        crops_done = 0
        for y in grid_offsets(H, crop):
            for x in grid_offsets(W, crop):
                src_c = src_frame[y:y + crop, x:x + crop]
                ref_c = ref_frame[y:y + crop, x:x + crop]
                if (ref_c.max(2) < 6).mean() > 0.10:
                    continue
                if tissue_fraction(src_c) < tissue_thresh:
                    continue
                tag = f"{r['aperio_slide']}_{r['frame_id']}_x{x}_y{y}"
                out[tag] = src_c
                crops_done += 1
        print(f"  [A_raw re-derive] {r['aperio_slide']}_{r['frame_id']}: {crops_done} crops")
    return out


def main():
    args = parse_args()
    fuse = FUSERS[args.method]
    if args.method == "f2" and not (0.0 <= args.alpha):
        raise SystemExit("--alpha must be >= 0 for --method f2.")
    if args.method == "f3" and not (0.0 <= args.beta):
        raise SystemExit("--beta must be >= 0 for --method f3.")

    if not args.identity_manifest and not args.vae_only_manifest:
        raise SystemExit("Provide either --identity-manifest (SD1.5/P1-11 convention) or "
                         "--vae-only-manifest (P3-08/SDXL convention) -- see their --help text.")
    if args.identity_manifest and args.vae_only_manifest:
        raise SystemExit("--identity-manifest and --vae-only-manifest are mutually exclusive "
                         "(two different ways of supplying A_raw/A_V) -- pass only one.")

    tr_base, tr_rows = load_rows(args.translate_manifest)

    def is_vae_only(r):
        # Two manifest shapes exist for a vae_only row in this project: a
        # mixed-method infer_colour_source_ddim_inversion.py --mode identity
        # manifest (method=="vae_only", source_mode=="identity_vae_only"), or
        # a dedicated infer_colour_translation.py/infer_colour_translation_
        # sdxl.py --vae-only manifest (no "method" column at all,
        # source_mode=="vae_only" directly). Accept either so a full-scale
        # --vae-only run (e.g. the existing project-wide VAE-only floor
        # check) can supply A_V without a new GPU job.
        return r.get("method") == "vae_only" or r.get("source_mode") == "vae_only"

    if args.identity_manifest:
        # SD1.5/P1-11 convention: A_raw + A_V both come from the identity
        # manifest -- reference_path IS the untouched source crop under
        # --mode identity (A->A), by construction of that special mode.
        id_base, id_rows = load_rows(args.identity_manifest)
        a_raw_by_crop = {r["crop_id"]: id_base / r["reference_path"] for r in id_rows}
        a_vae_by_key = {(r["crop_id"], r["seed"]): id_base / r["output_path"]
                        for r in id_rows if is_vae_only(r)}
    else:
        # P3-08/SDXL convention: no identity-mode run exists. A_raw comes
        # directly from the TRANSLATE manifest's own aperio_path column
        # (always the true Aperio crop, direction-invariant -- confirmed by
        # direct read of infer_colour_translation_sdxl.py: written verbatim
        # regardless of --direction, unlike reference_path). A_V comes from
        # a separate plain --vae-only run's manifest.
        #
        # infer_colour_translation_sdxl.py's own --vae-only mode always
        # writes exactly ONE seed (`seeds = [0] if args.vae_only else
        # args.seeds` -- confirmed by direct read; also confirmed empirically,
        # job 61319: "495 output crops ... x 1 seed(s)" against P3-07c's own
        # 1485 = 495x3 seeds). This is semantically correct, not a shortfall
        # -- VAE encode/decode has no randomness, so the SAME reconstruction
        # is the right A_V for every one of H_pred's 3 seeds. Key by crop_id
        # ONLY here (not (crop_id, seed), unlike the identity-manifest
        # branch above) so every H_pred seed finds the one real A_V rather
        # than being wrongly skipped for 2 of every 3 rows.
        # A_raw is NOT retrievable from any saved file (confirmed empirically,
        # job 61392: infer_colour_translation_sdxl.py never writes the raw
        # source crop to disk, only reference_path=Hamamatsu and output_path;
        # aperio_path is a provenance pointer to the original full-resolution
        # TIFF frame, not a crop). Re-derive it directly, duplicating the
        # EXACT same registration + grid-crop + filter routine
        # infer_colour_translation_sdxl.py itself uses (own small copy, same
        # convention already established by benchmark_vae_reconstruction.py)
        # so the crop_id SET matches exactly -- a mismatch here would silently
        # under-cover the fusion, not just misalign a few rows.
        if not (args.aperio_root and args.heldout_csv):
            raise SystemExit("--aperio-root and --heldout-csv are required with --vae-only-manifest "
                             "(A_raw must be re-derived -- see that flag's --help).")
        a_raw_by_crop = _rederive_aperio_crops(
            args.aperio_root, args.heldout_csv, args.crop, args.tissue_thresh, args.ecc_min,
            direction=args.direction)
        vae_base, vae_rows = load_rows(args.vae_only_manifest)
        a_vae_by_crop_only = {r["crop_id"]: vae_base / r["output_path"]
                              for r in vae_rows if is_vae_only(r)}
        if not a_vae_by_crop_only:
            raise SystemExit(f"--vae-only-manifest {args.vae_only_manifest} has no "
                             f"source_mode==vae_only rows -- wrong file?")
        a_vae_by_key = None  # signals the crop_id-only lookup path below

    # H_pred: translate-mode rows, method==ddim_inversion (or the column is
    # absent -- some existing full-scale translate-mode manifests in this
    # project were trimmed to a simpler schema with no "method" column at
    # all, since every row in a single-source-mode translate run already IS
    # ddim_inversion; .get() with a matching default handles both shapes),
    # source_mode==correct (the primary A->H prediction -- zero/shuffled
    # ablation rows, if present in the same manifest, are deliberately
    # excluded).
    h_pred_rows = [r for r in tr_rows
                   if r.get("method", "ddim_inversion") == "ddim_inversion" and r["source_mode"] == "correct"]
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
        a_raw_entry = a_raw_by_crop[crop_id]
        # P3-08/SDXL convention: a_raw_by_crop holds already-decoded ndarrays
        # (re-derived, never saved to disk -- see _rederive_aperio_crops).
        # SD1.5/identity-manifest convention: holds Path objects, unchanged.
        a_raw_rgb = a_raw_entry if isinstance(a_raw_entry, np.ndarray) else read_rgb_png(a_raw_entry)
        h_pred_path = tr_base / r["output_path"]
        h_pred_rgb = read_rgb_png(h_pred_path)

        kwargs = {"sigma": args.sigma, "alpha": args.alpha, "beta": args.beta}
        if args.method == "f3":
            # crop_id-only lookup for the P3-08/SDXL convention (a_vae_by_key
            # is None there -- see its construction above); (crop_id, seed)
            # for the SD1.5/identity-manifest convention, unchanged.
            a_vae_lookup = a_vae_by_crop_only if a_vae_by_key is None else a_vae_by_key
            key = crop_id if a_vae_by_key is None else (crop_id, seed)
            if key not in a_vae_lookup:
                n_skipped += 1
                continue
            kwargs["a_vae_rgb"] = read_rgb_png(a_vae_lookup[key])

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
        man.flush()
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
