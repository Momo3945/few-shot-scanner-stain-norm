#!/usr/bin/env python3
"""
benchmark_vae_reconstruction.py -- P1-14: pure VAE self-reconstruction
benchmark, stock SD1.5 VAE vs stabilityai/sd-vae-ft-mse. Isolates
decoder/autoencoder reconstruction quality from everything else in the
pipeline -- no UNet, no ControlNet, no LoRA, no noise, no DDIM, no scanner
translation. Deterministic (encode via .mode(), never .sample()). See
tickets/P1-14_vae_reconstruction_benchmark.md and
tickets/PHASE1-TICKETS.md P1-14.

Deliberately a standalone script -- does NOT edit
infer_colour_translation.py or any other validated P1-10/P1-11 script, per
the ticket's explicit requirement. Its --vae-only branch's exact
preprocessing/scaling convention is mirrored here (deterministic .mode()
encode, /127.5-1.0 normalisation, vae.config.scaling_factor -- read from
the LOADED model, never assumed from an absent config.json key, since
NEITHER stock-SD1.5's nor sd-vae-ft-mse's cached config.json explicitly
serialises scaling_factor -- confirmed by inspecting both directly before
writing this script) -- but this script measures TRUE self-reconstruction
(encode a crop, decode it, compare the reconstruction to the SAME crop),
not the cross-domain "identity-transform ceiling" job 44515's VAE-only
floor check used. That is a deliberate, ticket-specified difference: P1-14
asks "does the VAE reconstruct Aperio crops faithfully, and Hamamatsu crops
faithfully" as two SEPARATE questions, not "how close does encode-then-
decode(Aperio) get to real Hamamatsu".

Stage A (--stage internal): P1-10's own non-held-out training pairs
(pairs/train, A03/H03 only, via build_pairs() -- same leak-check-safe
pattes used everywhere else in this project), both scanner domains scored
independently.

Stage B (--stage heldout): the canonical 496-crop held-out set, using the
EXACT same crop-selection pipeline (registration + tissue-fraction filter
+ grid tiling) as infer_colour_translation.py, so the crop SET is provably
identical to the existing stock-VAE control -- registration is needed only
to reproduce that exact crop selection (one of its filters depends on the
registered Hamamatsu frame), not for scoring, since scoring here is
Aperio-vs-itself only ("use exactly the same canonical Aperio crops").

Usage
-----
    # Stage A, both VAEs, internal validation pairs:
    python benchmark_vae_reconstruction.py --stage internal --vae both \
        --pairs-dir <data>/pairs/train --out <data>/eval/p1_14_vae_stage_a

    # Stage B, both VAEs, canonical held-out set (only if Stage A gates open):
    python benchmark_vae_reconstruction.py --stage heldout --vae both \
        --root <data>/mitos_heldout --heldout <data>/pairs/heldout_frames.csv \
        --out <data>/eval/p1_14_vae_stage_b

Dependencies: torch, diffusers, numpy, Pillow, opencv-python-headless,
scikit-image. Requires metrics.py/registration.py on the path (same
folder) and train_colour_translation_lora.py (sibling src/train/ dir) for
build_pairs() in --stage internal.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

VAE_REPOS = {
    "stock": ("stable-diffusion-v1-5/stable-diffusion-v1-5", "vae"),
    "ft_mse": ("stabilityai/sd-vae-ft-mse", None),
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="P1-14: pure VAE self-reconstruction benchmark, stock vs sd-vae-ft-mse.")
    ap.add_argument("--vae", choices=["stock", "ft_mse", "both"], default="both")
    ap.add_argument("--stage", choices=["internal", "heldout"], required=True)
    ap.add_argument("--pairs-dir", default=None, help="Stage internal: e.g. pairs/train.")
    ap.add_argument("--root", default=None, help="Stage heldout: dataset root.")
    ap.add_argument("--heldout", default=None, help="Stage heldout: heldout_frames.csv.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help="Max held-out frames (0 = all).")
    ap.add_argument("--max-crops-per-frame", type=int, default=4)
    ap.add_argument("--bootstrap-n", type=int, default=2000,
                    help="Bootstrap resamples for the paired SSIM-delta CI (both-VAE runs only).")
    ap.add_argument("--qualitative-slides", nargs="*", default=[],
                    help="Stage heldout only: for one representative crop per named slide (the "
                         "first crop encountered), save a side-by-side panel (original | stock "
                         "reconstruction | ft_mse reconstruction) plus an absolute-error heatmap "
                         "per VAE, to <out>/qualitative/. Requires --vae both. E.g. A06 A08.")
    ap.add_argument("--qualitative-only", action="store_true",
                    help="Skip the full per-crop reconstruction+scoring pass entirely -- only "
                         "gather crops (cheap: registration, no reconstruction) and generate the "
                         "--qualitative-slides panels. For adding panels to an already-completed "
                         "scoring run without repeating the expensive part.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--online", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.stage == "internal" and not args.pairs_dir:
        raise SystemExit("--pairs-dir is required for --stage internal.")
    if args.stage == "heldout" and not (args.root and args.heldout):
        raise SystemExit("--root and --heldout are required for --stage heldout.")

    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import diffusers
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL

    from metrics import score_aligned_pair

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("No CUDA device -- this script is meant for a GPU node.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vae_names = ["stock", "ft_mse"] if args.vae == "both" else [args.vae]

    # ------------------------------------------------------------------
    # Load requested VAE(s) + log the ticket-required config inspection
    # (architecture, param count, dtype, and the ACTUAL scaling factor
    # read from the loaded model -- never assumed from the config.json,
    # since neither VAE's cached config explicitly serialises it).
    # ------------------------------------------------------------------
    vaes = {}
    vae_info = {}
    for name in vae_names:
        repo, subfolder = VAE_REPOS[name]
        kwargs = {"subfolder": subfolder} if subfolder else {}
        vae = AutoencoderKL.from_pretrained(repo, torch_dtype=torch.float16, **kwargs).to(device)
        vae.eval()
        vaes[name] = vae
        info = {
            "repo": repo, "subfolder": subfolder,
            "diffusers_version": diffusers.__version__,
            "latent_channels": vae.config.latent_channels,
            "block_out_channels": list(vae.config.block_out_channels),
            "down_block_types": list(vae.config.down_block_types),
            "up_block_types": list(vae.config.up_block_types),
            "sample_size": vae.config.sample_size,
            "param_count": sum(p.numel() for p in vae.parameters()),
            "dtype": str(vae.dtype),
            "scaling_factor": vae.config.scaling_factor,
        }
        vae_info[name] = info
        print(f"VAE '{name}': {json.dumps(info, indent=2)}")
    with open(out_dir / "vae_config_inspection.json", "w") as fh:
        json.dump(vae_info, fh, indent=2)

    def encode_decode(vae, rgb_uint8):
        """Deterministic self-reconstruction: encode via .mode() (never
        .sample()), decode, return uint8 RGB. Same normalisation/scaling
        convention as infer_colour_translation.py's --vae-only branch."""
        scaling = vae.config.scaling_factor
        arr = torch.from_numpy(rgb_uint8.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        arr = arr.unsqueeze(0).to(device, dtype=torch.float16)
        with torch.no_grad():
            latents = vae.encode(arr).latent_dist.mode() * scaling
            decoded = vae.decode(latents / scaling).sample
        out = ((decoded[0].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return out

    # ------------------------------------------------------------------
    # Crop gathering
    # ------------------------------------------------------------------
    crops = []  # each: {slide, frame, x, y, tag_id, aperio, hamamatsu}
    if args.stage == "internal":
        from train_colour_translation_lora import build_pairs
        pairs = build_pairs(args.pairs_dir)
        for p in pairs:
            aperio = np.asarray(Image.open(p["aperio_path"]).convert("RGB"))
            hamamatsu = np.asarray(Image.open(p["hamamatsu_path"]).convert("RGB"))
            crops.append({"slide": p["slide"], "frame": p["frame_id"], "x": 0, "y": 0,
                         "tag_id": p["pair_id"], "aperio": aperio, "hamamatsu": hamamatsu})
        print(f"Stage internal: {len(crops)} pairs from {args.pairs_dir}.")
    else:
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

        with open(args.heldout, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if args.limit:
            rows = rows[: args.limit]
        for r in rows:
            r["aperio_path"] = r["aperio_path"].replace("\\", "/")
            r["hamamatsu_path"] = r["hamamatsu_path"].replace("\\", "/")

        for r in rows:
            a_rgb = read_rgb(Path(args.root) / r["aperio_path"])
            h_rgb = read_rgb(Path(args.root) / r["hamamatsu_path"])
            reg = register_h_to_a(a_rgb, h_rgb, ecc_min=args.ecc_min)
            h_reg = reg.h_registered_rgb
            H, W = a_rgb.shape[:2]

            crops_done = 0
            for y in grid_offsets(H, args.crop):
                for x in grid_offsets(W, args.crop):
                    if args.max_crops_per_frame and crops_done >= args.max_crops_per_frame:
                        break
                    a_c = a_rgb[y:y + args.crop, x:x + args.crop]
                    h_c = h_reg[y:y + args.crop, x:x + args.crop]
                    # Same two filters as infer_colour_translation.py, so the
                    # crop SET is provably identical to the existing control.
                    if (h_c.max(2) < 6).mean() > 0.10:
                        continue
                    if tissue_fraction(a_c) < args.tissue_thresh:
                        continue
                    crops.append({"slide": r["aperio_slide"], "frame": r["frame_id"],
                                 "x": x, "y": y,
                                 "tag_id": f"{r['aperio_slide']}_{r['frame_id']}_x{x}_y{y}",
                                 "aperio": a_c, "hamamatsu": h_c})
                    crops_done += 1
            print(f"  {r['aperio_slide']}_{r['frame_id']}: {crops_done} crops")
        print(f"Stage heldout: {len(crops)} crops (canonical Aperio-crop selection).")
    if not crops:
        raise SystemExit("No crops selected -- check --pairs-dir / --root+--heldout.")

    # ------------------------------------------------------------------
    # Self-reconstruction + scoring, both domains, each requested VAE.
    # Skipped entirely under --qualitative-only (adding panels to an
    # already-completed scoring run without repeating the expensive part).
    # ------------------------------------------------------------------
    domains = ["aperio", "hamamatsu"] if args.stage == "internal" else ["aperio"]
    metric_keys = ["lab_total", "wlab_mean", "de2000_mean", "ssim", "psnr", "mae"]
    scores = {name: {d: [] for d in domains} for name in vae_names}
    if not args.qualitative_only:
        per_crop_path = out_dir / "per_crop.csv"
        per_crop_fh = open(per_crop_path, "w", newline="")
        pc_w = csv.writer(per_crop_fh)
        pc_w.writerow(["vae", "domain", "crop_id", "slide", "frame", "x", "y", *metric_keys])

        for name in vae_names:
            vae = vaes[name]
            for c in crops:
                for domain in domains:
                    original = c[domain]
                    recon = encode_decode(vae, original)
                    m = score_aligned_pair(recon, original)
                    scores[name][domain].append(m)
                    pc_w.writerow([name, domain, c["tag_id"], c["slide"], c["frame"], c["x"], c["y"],
                                  *(round(m[k], 5) for k in metric_keys)])
        per_crop_fh.close()

    # ------------------------------------------------------------------
    # Qualitative panel (Stage heldout only): one representative crop per
    # requested slide -- original | stock reconstruction | ft_mse
    # reconstruction, plus a per-VAE absolute-error heatmap. Ticket's
    # ("Qualitative Check") explicit ask -- inspect colour drift, blur,
    # ringing, checkerboarding, nuclear/chromatin/stromal detail by eye,
    # not automated.
    # ------------------------------------------------------------------
    if args.qualitative_slides:
        if args.vae != "both":
            print("WARNING: --qualitative-slides requested but --vae != both -- skipping panel.")
        else:
            qual_dir = out_dir / "qualitative"
            qual_dir.mkdir(parents=True, exist_ok=True)
            for slide in args.qualitative_slides:
                c = next((c for c in crops if c["slide"] == slide), None)
                if c is None:
                    print(f"WARNING: no crop found for slide '{slide}' -- skipping.")
                    continue
                original = c["aperio"]
                recon_stock = encode_decode(vaes["stock"], original)
                recon_ftmse = encode_decode(vaes["ft_mse"], original)
                panel = np.concatenate([original, recon_stock, recon_ftmse], axis=1)
                Image.fromarray(panel).save(qual_dir / f"{slide}_{c['tag_id']}_panel_orig_stock_ftmse.png")
                for vae_name, recon in [("stock", recon_stock), ("ft_mse", recon_ftmse)]:
                    err = np.abs(original.astype(np.int16) - recon.astype(np.int16)).astype(np.uint8)
                    err_gray = err.max(axis=2)  # per-pixel worst-channel abs error, 0-255
                    Image.fromarray(err_gray).save(
                        qual_dir / f"{slide}_{c['tag_id']}_{vae_name}_abs_error_heatmap.png")
                print(f"  Qualitative panel saved for slide '{slide}' (crop {c['tag_id']}) -> {qual_dir}")

    if args.qualitative_only:
        print("\n--qualitative-only: skipping scoring/summary/paired-stats -- panels only.")
        return

    def mean_of(dicts, key):
        return sum(d[key] for d in dicts) / len(dicts)

    summary = {}
    for name in vae_names:
        summary[name] = {}
        for domain in domains:
            summary[name][domain] = {k: mean_of(scores[name][domain], k) for k in metric_keys}
            m = summary[name][domain]
            print(f"[{name}][{domain}] ssim={m['ssim']:.4f} lab_total={m['lab_total']:.2f} "
                  f"wlab_mean={m['wlab_mean']:.2f} psnr={m['psnr']:.2f} mae={m['mae']:.2f}")

    # ------------------------------------------------------------------
    # Paired stock-vs-ft_mse statistics (only meaningful when both ran).
    # ------------------------------------------------------------------
    paired = {}
    if args.vae == "both":
        rng = np.random.default_rng(args.seed)
        for domain in domains:
            stock_scores = scores["stock"][domain]
            ftmse_scores = scores["ft_mse"][domain]
            deltas = {k: [b[k] - a[k] for a, b in zip(stock_scores, ftmse_scores)] for k in metric_keys}
            ssim_deltas = np.asarray(deltas["ssim"])
            boot_means = [rng.choice(ssim_deltas, size=len(ssim_deltas), replace=True).mean()
                         for _ in range(args.bootstrap_n)]
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
            paired[domain] = {
                "mean_delta_ssim": float(np.mean(deltas["ssim"])),
                "median_delta_ssim": float(np.median(deltas["ssim"])),
                "mean_delta_psnr": float(np.mean(deltas["psnr"])),
                "mean_delta_mae": float(np.mean(deltas["mae"])),
                "mean_delta_lab_total": float(np.mean(deltas["lab_total"])),
                "ssim_delta_bootstrap_ci95": [float(ci_lo), float(ci_hi)],
                "n_crops": len(ssim_deltas),
            }
            p = paired[domain]
            print(f"[paired ft_mse-stock][{domain}] mean_dSSIM={p['mean_delta_ssim']:+.4f} "
                  f"(95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}])  median_dSSIM={p['median_delta_ssim']:+.4f}  "
                  f"mean_dPSNR={p['mean_delta_psnr']:+.3f}  mean_dMAE={p['mean_delta_mae']:+.3f}")

    with open(out_dir / "summary.json", "w") as fh:
        json.dump({"vae_info": vae_info, "summary": summary, "paired": paired,
                   "stage": args.stage, "n_crops": len(crops)}, fh, indent=2)
    print(f"\nWritten: {out_dir / 'summary.json'}  and  {per_crop_path}")


if __name__ == "__main__":
    main()
