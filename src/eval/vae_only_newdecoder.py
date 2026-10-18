#!/usr/bin/env python3
"""
vae_only_newdecoder.py -- P1-18 G1.4 prerequisite: the VAE-only reconstruction A_V of each held-out
source crop, encoded with the STOCK SD1.5 VAE and decoded with a different decoder (a P1-18
fine-tuned VAE dir). P1-16's F3 fusion subtracts A_V from the prediction H_pred
(Delta = LAB(H_pred) - LAB(A_V)) precisely so that decoder-specific characteristics cancel, so when
H_pred is decoded by the new decoder, A_V must be too.

Reads an existing P1-11 --mode identity manifest (e.g. eval/p1_10_ddim_inversion/identity_full_heldout,
the checksum-verified full-496 run P1-16 used): its reference_path IS the untouched source crop, and its
method=="vae_only" rows give the (crop_id, seed) set. Reconstructs each crop once (deterministic
.mode() encode, no randomness, same normalisation/scaling/fp16 convention as
infer_colour_source_ddim_inversion_alt_decoder.py's encode_latent/decode_with) and writes a manifest
in the SAME identity-manifest schema fuse_source_detail.py --identity-manifest already reads, so the
fusion script needs no change beyond its --pred-method flag.

No registration, no UNet, no diffusion: fast. Run via slurm/vae_only_newdecoder.slurm (GPU).

Usage
-----
    python vae_only_newdecoder.py \
        --identity-manifest <data>/eval/p1_10_ddim_inversion/identity_full_heldout/eval_manifest.csv \
        --decoder-path <data>/vae/d1_pn03/best_val \
        --out <data>/eval/p1_18_g14/vae_only_d1_pn03
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity-manifest", required=True)
    ap.add_argument("--decoder-path", required=True, help="Local AutoencoderKL dir (save_pretrained).")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="Max crops (0 = all).")
    ap.add_argument("--online", action="store_true")
    args = ap.parse_args()
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL

    if not torch.cuda.is_available():
        raise SystemExit("ABORT: no CUDA device -- this script is meant for a GPU node.")
    device, dtype = torch.device("cuda"), torch.float16

    stock = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype).to(device).eval()
    new = AutoencoderKL.from_pretrained(args.decoder_path, torch_dtype=dtype).to(device).eval()
    s_stock, s_new = stock.config.scaling_factor, new.config.scaling_factor
    print(f"stock scaling={s_stock}  new decoder scaling={s_new}  decoder={args.decoder_path}")
    if abs(s_stock - s_new) > 1e-9 or stock.config.latent_channels != new.config.latent_channels:
        raise SystemExit("ABORT: incompatible stock/new VAE (latent channels or scaling factor differ).")

    man_path = Path(args.identity_manifest)
    base = man_path.parent
    rows = list(csv.DictReader(open(man_path, newline="")))
    vo = [r for r in rows if r.get("method") == "vae_only" or r.get("source_mode") == "vae_only"]
    if not vo:
        raise SystemExit("No vae_only rows in the identity manifest -- wrong file?")
    crops = {}
    for r in vo:
        crops.setdefault(r["crop_id"], r["reference_path"])  # reference_path == untouched source crop
    ids = sorted(crops)
    if args.limit:
        ids = ids[: args.limit]
        vo = [r for r in vo if r["crop_id"] in set(ids)]
    print(f"{len(vo)} vae_only rows over {len(ids)} unique crops")

    out_dir = Path(args.out)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)

    done = {}
    with torch.no_grad():
        for n, cid in enumerate(ids, 1):
            ref = base / crops[cid]
            rgb = np.asarray(Image.open(ref).convert("RGB"))
            x = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
            x = x.to(device, dtype=dtype)
            z = stock.encode(x).latent_dist.mode() * s_stock       # stock encoder, always
            dec = new.decode(z / s_new).sample
            out = ((dec[0].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            p = out_dir / "outputs" / f"{cid}_vae_only.png"
            Image.fromarray(out).save(p)
            done[cid] = p
            if n % 50 == 0 or n == len(ids):
                print(f"  {n}/{len(ids)}")

    man = open(out_dir / "eval_manifest.csv", "w", newline="")
    w = csv.writer(man)
    w.writerow(["seed", "source_mode", "crop_id", "slide", "frame", "x", "y", "output_path",
                "reference_path", "aperio_path", "method"])
    for r in vo:
        p = done[r["crop_id"]]
        ref_abs = base / crops[r["crop_id"]]
        w.writerow([r["seed"], "identity_vae_only", r["crop_id"], r.get("slide", ""), r.get("frame", ""),
                    r.get("x", ""), r.get("y", ""), os.path.relpath(p, out_dir),
                    os.path.relpath(ref_abs, out_dir), r.get("aperio_path", ""), "vae_only"])
    man.close()
    print(f"Wrote {len(vo)} manifest rows, {len(done)} reconstructions -> {out_dir}")


if __name__ == "__main__":
    main()
