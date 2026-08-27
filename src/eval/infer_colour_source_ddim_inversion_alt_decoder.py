#!/usr/bin/env python3
"""
infer_colour_source_ddim_inversion_alt_decoder.py -- P1-15: decoder-only
swap on top of P1-11's validated DDIM-inversion translation path. See
tickets/P1-15_p1_11_alternate_decoder.md and tickets/PHASE1-TICKETS.md
P1-15.

Stage 1's own critical isolation rule: stock SD1.5 ENCODER, P1-11's exact
DDIM-inversion trajectory (f=1.00 default, correct source, same guidance/
steps/seeds), P1-10 frozen colour LoRA + source-conditioning ControlNet all
held fixed -- ONLY the final VAE decode call changes, from the stock
decoder to P1-14's winning alternate decoder (stabilityai/sd-vae-ft-mse).
Swapping the encoder is explicitly out of scope (P1-10 was never trained
under an altered encoder's latent distribution) -- encode_latent() below
always uses the one fixed stock `vae` object, structurally, never the
alternate one.

Deliberately a standalone script -- does NOT edit
infer_colour_source_ddim_inversion.py (P1-11's validated script). The
crop-loading / predict_noise / encode_latent / invert / reconstruct logic
below is copied verbatim from that script (same precedent as P1-13's own
standalone PairDataset copy elsewhere in this project), so this script has
zero runtime dependency on the original ever changing underneath it. Only
`--mode translate` is implemented -- P1-11's own `--mode identity`
(vae_only/img2img_baseline comparison arms) was validation scaffolding for
THAT script, not requested anywhere in the P1-15 ticket.

Exact-latent A/B (ticket's own mandatory first test), built into every
invocation, not a separate mode: the reconstruction trajectory
(invert -> reconstruct) runs EXACTLY ONCE per crop/seed -- never two
separate denoising trajectories for the two decoder arms -- and the single
resulting latent is decoded through BOTH the stock VAE and the alternate
VAE, producing two manifest rows (`method=stock_decoder` /
`method=alt_decoder`) that share the identical upstream latent by
construction.

Compatibility verification (ticket's own checklist) is logged at startup:
latent_channels, scaling_factor (read from the LOADED model's config, not
assumed -- same convention already verified in P1-14's
benchmark_vae_reconstruction.py), dtype, param count, diffusers version,
for both VAEs -- and asserted equal before any translation runs.

Usage
-----
    # Stage 1 smoke test, A06+A08 subset, f=1.00, correct source:
    python infer_colour_source_ddim_inversion_alt_decoder.py \
        --lora <ckpt> --controlnet <ckpt>/controlnet \
        --root <mitos_heldout> --heldout pairs/heldout_frames.csv \
        --out eval/p1_15_smoke --source-mode correct --limit 20 --seeds 0 1 2

    # Source-conditioning sanity check:
    python infer_colour_source_ddim_inversion_alt_decoder.py ... \
        --out eval/p1_15_smoke_shuffled --source-mode shuffled --limit 20 --seeds 0 1 2

Dependencies: torch, diffusers>=0.39 (DDIMInverseScheduler), numpy, Pillow,
opencv-python-headless, tifffile. Requires canny.py and registration.py (or
train_colour_translation_lora.py for --pairs-dir) on the path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="P1-15: decoder-only swap on top of P1-11's DDIM-inversion translation.")
    ap.add_argument("--lora", required=True,
                    help="Path to the trained P1-10 LoRA checkpoint dir (e.g. lora/a2h_cond_r8/best).")
    ap.add_argument("--controlnet", required=True,
                    help="Path to the trained P1-10 ControlNet dir (e.g. lora/a2h_cond_r8/best/controlnet).")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--alt-decoder-repo", default="stabilityai/sd-vae-ft-mse",
                    help="P1-14's winning alternate decoder.")
    ap.add_argument("--root", default=None, help="Dataset root (heldout paths are relative to this). "
                    "Mutually exclusive with --pairs-dir.")
    ap.add_argument("--heldout", default=None, help="heldout_frames.csv. Mutually exclusive with --pairs-dir.")
    ap.add_argument("--pairs-dir", default=None,
                    help="Run against training pairs instead of held-out frames. Mutually exclusive "
                         "with --root/--heldout.")
    ap.add_argument("--overfit-n", type=int, default=0)
    ap.add_argument("--out", required=True, help="Output dir for crops + manifest + run_metadata.json.")
    ap.add_argument("--direction", choices=["A2H", "H2A"], default="A2H")
    ap.add_argument("--source-mode", choices=["correct", "zero", "shuffled"], default="correct",
                    help="Stage 1's fixed setting is 'correct'. 'shuffled' is the ticket's "
                         "Source-Conditioning Sanity Check; 'zero' retained for parity.")
    ap.add_argument("--inversion-condition", choices=["none", "source"], default="source")
    ap.add_argument("--inversion-fraction", type=float, default=1.0,
                    help="Stage 1's fixed f=1.00 (P1-11's own best/canonical setting).")
    ap.add_argument("--inversion-steps", type=int, default=50)
    ap.add_argument("--translation-steps", type=int, default=50)
    ap.add_argument("--inversion-guidance", type=float, default=1.0)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--controlnet-scale", type=float, default=1.0)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help="Max held-out frames (0 = all).")
    ap.add_argument("--max-crops-per-frame", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--debug-inversion", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not (0.0 < args.inversion_fraction <= 1.0):
        raise SystemExit(f"--inversion-fraction must satisfy 0 < f <= 1, got {args.inversion_fraction}.")

    using_pairs_dir = bool(args.pairs_dir)
    using_heldout = bool(args.root or args.heldout)
    if using_pairs_dir and using_heldout:
        raise SystemExit("--pairs-dir is mutually exclusive with --root/--heldout.")
    if not using_pairs_dir and not (args.root and args.heldout):
        raise SystemExit("Either --pairs-dir, or both --root and --heldout, must be given.")

    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import diffusers
    import numpy as np
    import torch
    from PIL import Image

    from canny import extract_canny_control_image
    if using_pairs_dir:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
        from train_colour_translation_lora import build_pairs
    else:
        from registration import read_rgb, register_h_to_a

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("No CUDA device -- inference must run on a GPU node.")
    dtype = torch.float16

    out_dir = Path(args.out)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Crop loading -- byte-for-byte copy of
    # infer_colour_source_ddim_inversion.py's own (which itself copies
    # infer_colour_translation.py's) -- zero runtime dependency on either
    # ever changing underneath this script.
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

    crops = []
    if using_pairs_dir:
        pairs = build_pairs(args.pairs_dir)
        if args.overfit_n:
            pairs = pairs[: args.overfit_n]
        if args.limit:
            pairs = pairs[: args.limit]
        src_key, ref_key = ("aperio_path", "hamamatsu_path") if args.direction == "A2H" \
            else ("hamamatsu_path", "aperio_path")
        for p in pairs:
            src_c = np.asarray(Image.open(p[src_key]).convert("RGB"))
            ref_c = np.asarray(Image.open(p[ref_key]).convert("RGB"))
            crops.append({"slide": p["slide"], "frame": p["frame_id"], "x": 0, "y": 0,
                         "tag_id": p["pair_id"], "src": src_c, "ref": ref_c, "aperio_path": p["aperio_path"]})
        print(f"  {args.pairs_dir}: {len(crops)} training pairs (overfit_n={args.overfit_n or 'all'})")
    else:
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
                    crops.append({"slide": r["aperio_slide"], "frame": r["frame_id"],
                                 "x": x, "y": y,
                                 "tag_id": f"{r['aperio_slide']}_{r['frame_id']}_x{x}_y{y}",
                                 "src": src_c, "ref": ref_c, "aperio_path": r["aperio_path"]})
                    crops_done += 1
            print(f"  {r['aperio_slide']}_{r['frame_id']}: {crops_done} crops")
    if not crops:
        raise SystemExit("No tissue crops selected -- check --tissue-thresh / heldout / pairs-dir inventory.")

    rng = np.random.default_rng(0)
    shuffled_idx = rng.permutation(len(crops))
    for i in range(len(crops)):
        if shuffled_idx[i] == i:
            j = (i + 1) % len(crops)
            shuffled_idx[i], shuffled_idx[j] = shuffled_idx[j], shuffled_idx[i]

    def control_source_for(i):
        if args.source_mode == "correct":
            return crops[i]["src"]
        if args.source_mode == "zero":
            return None
        return crops[shuffled_idx[i]]["src"]

    def build_control_tensor(rgb):
        canny = extract_canny_control_image(rgb)
        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        canny_t = torch.from_numpy(canny.astype(np.float32) / 255.0).permute(2, 0, 1)
        return torch.cat([rgb_t, canny_t], dim=0).unsqueeze(0)

    def control_tensor_or_zero(rgb):
        t = (torch.zeros(1, 6, args.crop, args.crop, dtype=dtype)
             if rgb is None else build_control_tensor(rgb).to(dtype=dtype))
        return t.to(device)

    def seed_for(pair_key: str, seed: int) -> int:
        h = hashlib.sha256(f"{pair_key}|{seed}".encode()).hexdigest()
        return int(h[:8], 16)

    # ------------------------------------------------------------------
    # Pipeline + schedulers -- identical construction to P1-11's script.
    # ------------------------------------------------------------------
    from diffusers import (AutoencoderKL, ControlNetModel, DDIMInverseScheduler,
                           DDIMScheduler, StableDiffusionControlNetImg2ImgPipeline)

    print(f"Loading P1-10 pipeline: LoRA={args.lora}  ControlNet={args.controlnet}  "
          f"source_mode={args.source_mode}  f={args.inversion_fraction} ...")
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.model, controlnet=controlnet, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False)
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = ddim_scheduler
    if ddim_scheduler.config.prediction_type != "epsilon":
        raise SystemExit(
            f"P1-10 (lora/a2h_cond_r8) was trained with epsilon prediction, but the resolved "
            f"scheduler config says prediction_type={ddim_scheduler.config.prediction_type!r}. "
            f"Refusing to silently run inversion/translation under a mismatched prediction type.")
    inverse_scheduler = DDIMInverseScheduler.from_config(ddim_scheduler.config)
    pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    vae, unet, text_encoder, tokenizer = pipe.vae, pipe.unet, pipe.text_encoder, pipe.tokenizer
    scaling = vae.config.scaling_factor

    # ------------------------------------------------------------------
    # Alternate decoder + compatibility verification (ticket's own
    # checklist -- logged, not assumed).
    # ------------------------------------------------------------------
    print(f"Loading alternate decoder: {args.alt_decoder_repo} ...")
    alt_vae = AutoencoderKL.from_pretrained(args.alt_decoder_repo, torch_dtype=dtype).to(device)
    alt_vae.eval()
    alt_scaling = alt_vae.config.scaling_factor

    def vae_info(name, v, scale):
        return {"name": name, "latent_channels": v.config.latent_channels,
               "scaling_factor": scale, "dtype": str(v.dtype),
               "param_count": sum(p.numel() for p in v.parameters()),
               "diffusers_version": diffusers.__version__}

    stock_info = vae_info("stock", vae, scaling)
    alt_info = vae_info("alt", alt_vae, alt_scaling)
    print(f"Compatibility verification:\n  stock: {json.dumps(stock_info)}\n  alt:   {json.dumps(alt_info)}")
    if stock_info["latent_channels"] != alt_info["latent_channels"]:
        raise SystemExit(f"ABORT: latent_channels mismatch (stock={stock_info['latent_channels']}, "
                         f"alt={alt_info['latent_channels']}) -- not a compatible decoder swap.")
    if abs(stock_info["scaling_factor"] - alt_info["scaling_factor"]) > 1e-9:
        raise SystemExit(f"ABORT: scaling_factor mismatch (stock={stock_info['scaling_factor']}, "
                         f"alt={alt_info['scaling_factor']}) -- not a compatible decoder swap.")
    print("Confirmed: alt decoder never used for encoding (encode_latent always uses the fixed "
         "stock `vae` object below) -- structural guarantee, not just an assertion.")

    with torch.no_grad():
        def encode_prompt(text):
            tok = tokenizer(text, padding="max_length", truncation=True,
                            max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
            return text_encoder(tok.input_ids)[0].to(dtype)
        cond_embeds = encode_prompt(args.prompt)
        uncond_embeds = encode_prompt("")

    # ------------------------------------------------------------------
    # Core diffusion machinery -- identical to P1-11's script.
    # ------------------------------------------------------------------
    def predict_noise(latent, t, control, guidance_scale):
        if guidance_scale is None or guidance_scale <= 1.0:
            down_res, mid_res = controlnet(
                latent, t, encoder_hidden_states=cond_embeds, controlnet_cond=control,
                conditioning_scale=args.controlnet_scale, return_dict=False)
            return unet(latent, t, encoder_hidden_states=cond_embeds,
                       down_block_additional_residuals=down_res,
                       mid_block_additional_residual=mid_res).sample
        latent_in = torch.cat([latent, latent], dim=0)
        embeds_in = torch.cat([uncond_embeds, cond_embeds], dim=0)
        control_in = torch.cat([control, control], dim=0)
        down_res, mid_res = controlnet(
            latent_in, t, encoder_hidden_states=embeds_in, controlnet_cond=control_in,
            conditioning_scale=args.controlnet_scale, return_dict=False)
        noise_pred = unet(latent_in, t, encoder_hidden_states=embeds_in,
                          down_block_additional_residuals=down_res,
                          mid_block_additional_residual=mid_res).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)
        return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    def encode_latent(rgb):
        # ALWAYS the stock `vae` -- the alt decoder is never used for
        # encoding, by construction (Stage 1's critical isolation rule).
        arr = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        arr = arr.unsqueeze(0).to(device, dtype=dtype)
        return vae.encode(arr).latent_dist.mode() * scaling

    def decode_with(v, scale, latent):
        decoded = v.decode(latent / scale).sample
        if not torch.isfinite(decoded).all():
            raise RuntimeError("NaN/Inf in decoded output -- aborting.")
        return ((decoded[0].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    def invert(latent0, control, guidance_scale, n_steps_total, fraction, tag=""):
        inverse_scheduler.set_timesteps(n_steps_total, device=device)
        timesteps = inverse_scheduler.timesteps
        k = max(1, round(n_steps_total * fraction))
        used = timesteps[:k]
        latent = latent0
        for i, t in enumerate(used):
            noise_pred = predict_noise(latent, t, control, guidance_scale)
            latent = inverse_scheduler.step(noise_pred, t, latent, return_dict=True).prev_sample
            if args.debug_inversion:
                print(f"    [inv {tag}] step {i+1}/{k} t={int(t)} "
                     f"mean={latent.mean().item():.4f} std={latent.std().item():.4f}")
        t_end = int(used[-1].item()) if len(used) else None
        if not torch.isfinite(latent).all():
            raise RuntimeError("NaN/Inf in inverted latent -- aborting.")
        return latent, t_end, k

    def reconstruct(latent_start, t_start, control, guidance_scale, n_steps_total, tag=""):
        ddim_scheduler.set_timesteps(n_steps_total, device=device)
        fwd_timesteps = ddim_scheduler.timesteps
        if t_start is None:
            used = fwd_timesteps
        else:
            idx = len(fwd_timesteps) - 1
            for i, tv in enumerate(fwd_timesteps):
                if tv.item() <= t_start:
                    idx = i
                    break
            used = fwd_timesteps[idx:]
        latent = latent_start
        for i, t in enumerate(used):
            noise_pred = predict_noise(latent, t, control, guidance_scale)
            latent = ddim_scheduler.step(noise_pred, t, latent, eta=0.0, return_dict=True).prev_sample
            if args.debug_inversion:
                print(f"    [rec {tag}] step {i+1}/{len(used)} t={int(t)} "
                     f"mean={latent.mean().item():.4f} std={latent.std().item():.4f}")
        if not torch.isfinite(latent).all():
            raise RuntimeError("NaN/Inf in reconstructed latent -- aborting.")
        return latent, len(used)

    @torch.no_grad()
    def run_translate(source_rgb, inv_ctrl_rgb, rec_ctrl_rgb, seed, tag):
        """Runs the inversion+reconstruction trajectory EXACTLY ONCE, then
        decodes the single resulting latent through BOTH decoders -- the
        ticket's mandatory exact-latent A/B, built into every call."""
        torch.manual_seed(seed)
        latent0 = encode_latent(source_rgb)
        inv_control = control_tensor_or_zero(inv_ctrl_rgb)
        rec_control = control_tensor_or_zero(rec_ctrl_rgb)
        latent_inv, t_end, k_actual = invert(
            latent0, inv_control, args.inversion_guidance,
            args.inversion_steps, args.inversion_fraction, tag=tag)
        latent_rec, n_rec = reconstruct(
            latent_inv, t_end, rec_control, args.guidance, args.translation_steps, tag=tag)
        out_stock = decode_with(vae, scaling, latent_rec)
        out_alt = decode_with(alt_vae, alt_scaling, latent_rec)
        if args.debug_inversion:
            print(f"    [meta {tag}] requested_fraction={args.inversion_fraction} "
                 f"actual_inv_steps={k_actual}/{args.inversion_steps} t_end={t_end} "
                 f"translation_steps_used={n_rec}/{args.translation_steps}")
        return out_stock, out_alt, t_end, k_actual, n_rec

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    man_path = out_dir / "eval_manifest.csv"
    man = open(man_path, "w", newline="")
    mw = csv.writer(man)
    mw.writerow(["seed", "source_mode", "crop_id", "slide", "frame", "x", "y",
                "output_path", "reference_path", "aperio_path",
                "method", "inversion_condition", "inversion_fraction",
                "actual_inversion_steps", "actual_start_timestep", "actual_translation_steps"])

    n_out = 0
    for i, c in enumerate(crops):
        tag = c["tag_id"]
        ref_path = out_dir / "reference" / f"{tag}.png"
        if not ref_path.exists():
            Image.fromarray(c["ref"]).save(ref_path)

        for s in args.seeds:
            seed_val = seed_for(tag, s)
            run_tag = f"{tag}_seed{s}"
            inv_ctrl_rgb = c["src"] if args.inversion_condition == "source" else None
            rec_ctrl_rgb = control_source_for(i)

            out_stock, out_alt, t_end, k_actual, n_rec = run_translate(
                c["src"], inv_ctrl_rgb, rec_ctrl_rgb, seed_val, run_tag)

            for method, out in [("stock_decoder", out_stock), ("alt_decoder", out_alt)]:
                p = out_dir / "outputs" / f"{run_tag}_{args.source_mode}_{method}.png"
                Image.fromarray(out).save(p)
                mw.writerow([s, f"{args.source_mode}_{method}", tag, c["slide"], c["frame"], c["x"], c["y"],
                            os.path.relpath(p, out_dir), os.path.relpath(ref_path, out_dir),
                            c["aperio_path"], method, args.inversion_condition,
                            args.inversion_fraction, k_actual, t_end, n_rec])
                n_out += 1

    man.close()

    def git_commit():
        try:
            repo_root = Path(__file__).resolve().parent.parent.parent
            return subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"

    metadata = {
        "script": "infer_colour_source_ddim_inversion_alt_decoder.py",
        "checkpoint_lora": args.lora, "checkpoint_controlnet": args.controlnet,
        "direction": args.direction, "source_mode": args.source_mode,
        "inversion_condition": args.inversion_condition,
        "prediction_type": ddim_scheduler.config.prediction_type,
        "stock_vae_info": stock_info, "alt_vae_info": alt_info,
        "alt_decoder_repo": args.alt_decoder_repo,
        "inversion_steps": args.inversion_steps, "translation_steps": args.translation_steps,
        "inversion_fraction": args.inversion_fraction, "inversion_guidance": args.inversion_guidance,
        "guidance": args.guidance, "controlnet_conditioning_scale": args.controlnet_scale,
        "seeds": args.seeds, "dtype": str(dtype), "model": args.model, "prompt": args.prompt,
        "git_commit": git_commit(),
    }
    with open(out_dir / "run_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nWrote {n_out} output rows (stock_decoder + alt_decoder per crop x seed) across "
         f"{len(crops)} crops x {len(args.seeds)} seed(s), source_mode={args.source_mode}.")
    print(f"Manifest: {man_path}")
    print(f"Metadata: {out_dir / 'run_metadata.json'}")
    print("Next: score with score_p1_10_ablation.py (same schema as P1-11's manifest).")


if __name__ == "__main__":
    main()
