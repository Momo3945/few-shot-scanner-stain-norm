#!/usr/bin/env python3
"""
infer_colour_source_ddim_inversion_ensemble.py -- P1-11 future-work idea #2
(tickets/PHASE1-TICKETS.md P1-11 "Future work", genuine self-ensembling,
StainDiff's technique): average multiple genuinely-different stochastic
reconstructions of the same crop to see whether ensembling improves SSIM/
colour over a single deterministic sample.

Deliberately a standalone script -- does NOT edit
infer_colour_source_ddim_inversion.py (P1-11's validated script). That
script's `reconstruct()` hardcodes `eta=0.0` on every `ddim_scheduler.
step()` call, which makes the whole inversion->reconstruction trajectory
fully deterministic once conditioning is fixed -- the reason its own
`--seeds` list currently produces numerically IDENTICAL outputs per crop
(documented in that script's own `--seeds` help text). Self-ensembling
needs genuinely different samples to average, so this script adds a
`--eta` flag (default 0.0, matching the original script's fixed behaviour
if left unset) that is threaded through to the RECONSTRUCTION pass only --
same precedent already used in this project for standalone variant scripts
(e.g. P1-15's alt-decoder script) rather than risking the validated
original.

Why only the reconstruction pass, never inversion: `DDIMInverseScheduler.
step()` has no `eta` parameter at all -- inversion is only mathematically
well-defined (invertible) at eta=0, by construction. Stochasticity is
standard practice on the DENOISING/reconstruction pass only (matching
DDPM-style ancestral sampling), never on inversion.

Translate mode only (not identity) -- this script tests whether ensembling
helps the actual colour/structure question against the real target, which
is what the self-ensembling idea is for; identity mode's vae_only/img2img
comparison points aren't needed here and are appropriately out of scope
(same scoping precedent as P1-15's own standalone script, which dropped
what its specific question didn't need).

Ensemble mechanism: `--seeds` IS the ensemble membership (e.g. `0 1 2` for
a 3-member ensemble; a single seed reduces to the same single-sample
behaviour as the original script when `--eta 0` too). For each crop, every
seed's full invert()+reconstruct() trajectory runs independently (with its
own `torch.Generator` for the eta>0 noise draw, so seeds are reproducible
and genuinely different from each other, not just relying on unseeded
global RNG state) and its own output is saved and manifested exactly like
the original script's per-seed rows (`method=ddim_inversion`,
`ensemble_role=member`) -- so single-sample comparisons stay available.
ADDITIONALLY, once every seed for a crop is done, this script saves ONE
extra pixel-wise mean of all per-seed outputs for that crop
(`method=ddim_inversion_ensemble`, `ensemble_role=ensemble`,
`ensemble_n=<len(seeds)>`), averaged in decoded RGB space (not latent
space -- simpler, and avoids re-deriving a valid "average latent" for a
non-linear VAE decoder).

Manifest's `source_mode` column carries `"<source_mode>_<ensemble_role>"`
(e.g. `correct_member` / `correct_ensemble`), same trick already used by
P1-15's own standalone script, so the EXISTING `score_p1_10_ablation.py`
groups members vs. the ensemble cleanly with zero scorer changes and
answers "does ensembling help SSIM" directly via its own per-mode
aggregation.

Usage
-----
    # Sanity check: eta=0, single seed -- should reduce to the same output
    # as infer_colour_source_ddim_inversion.py's own translate mode:
    python infer_colour_source_ddim_inversion_ensemble.py \
        --lora <ckpt> --controlnet <ckpt>/controlnet \
        --root <mitos_heldout> --heldout pairs/heldout_frames.csv \
        --out eval/p1_11_ensemble/eta0_sanity --eta 0.0 \
        --source-mode correct --limit 1 --seeds 0

    # Real ensemble test: eta>0, 3-member ensemble, smoke scale:
    python infer_colour_source_ddim_inversion_ensemble.py ... \
        --out eval/p1_11_ensemble/eta05_n3 --eta 0.5 \
        --source-mode correct --limit 20 --seeds 0 1 2

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
        description="P1-11 future-work #2: genuine self-ensembling via eta>0 stochastic reconstruction.")
    ap.add_argument("--lora", required=True,
                    help="Path to the trained P1-10 LoRA checkpoint dir (e.g. lora/a2h_cond_r8/best).")
    ap.add_argument("--controlnet", required=True,
                    help="Path to the trained P1-10 ControlNet dir (e.g. lora/a2h_cond_r8/best/controlnet).")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--root", default=None, help="Dataset root (heldout paths are relative to this). "
                    "Mutually exclusive with --pairs-dir.")
    ap.add_argument("--heldout", default=None, help="heldout_frames.csv. Mutually exclusive with --pairs-dir.")
    ap.add_argument("--pairs-dir", default=None,
                    help="Run against training pairs instead of held-out frames. Mutually exclusive "
                         "with --root/--heldout.")
    ap.add_argument("--overfit-n", type=int, default=0)
    ap.add_argument("--out", required=True, help="Output dir for crops + manifest + run_metadata.json.")
    ap.add_argument("--direction", choices=["A2H", "H2A"], default="A2H")
    ap.add_argument("--source-mode", choices=["correct", "zero", "shuffled"], default="correct")
    ap.add_argument("--inversion-condition", choices=["none", "source"], default="source")
    ap.add_argument("--inversion-fraction", type=float, default=1.0)
    ap.add_argument("--inversion-steps", type=int, default=50)
    ap.add_argument("--translation-steps", type=int, default=50)
    ap.add_argument("--inversion-guidance", type=float, default=1.0)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--controlnet-scale", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.0,
                    help="DDIM eta for the RECONSTRUCTION pass only (never inversion -- "
                         "DDIMInverseScheduler has no eta parameter, inversion is only "
                         "well-defined at eta=0). 0.0 reduces to the same deterministic "
                         "behaviour as infer_colour_source_ddim_inversion.py. >0 injects "
                         "genuine per-seed stochasticity via a seeded torch.Generator, "
                         "required for self-ensembling to do anything.")
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0, help="Max held-out frames (0 = all).")
    ap.add_argument("--max-crops-per-frame", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Ensemble membership -- each seed is one independent stochastic "
                         "sample (genuinely different when --eta > 0), averaged together "
                         "into one additional ensemble output per crop.")
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--debug-inversion", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not (0.0 < args.inversion_fraction <= 1.0):
        raise SystemExit(f"--inversion-fraction must satisfy 0 < f <= 1, got {args.inversion_fraction}.")
    if args.eta < 0.0:
        raise SystemExit(f"--eta must be >= 0, got {args.eta}.")

    using_pairs_dir = bool(args.pairs_dir)
    using_heldout = bool(args.root or args.heldout)
    if using_pairs_dir and using_heldout:
        raise SystemExit("--pairs-dir is mutually exclusive with --root/--heldout.")
    if not using_pairs_dir and not (args.root and args.heldout):
        raise SystemExit("Either --pairs-dir, or both --root and --heldout, must be given.")

    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
    from diffusers import (ControlNetModel, DDIMInverseScheduler,
                           DDIMScheduler, StableDiffusionControlNetImg2ImgPipeline)

    print(f"Loading P1-10 pipeline: LoRA={args.lora}  ControlNet={args.controlnet}  "
          f"source_mode={args.source_mode}  f={args.inversion_fraction}  eta={args.eta} ...")
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.model, controlnet=controlnet, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False)
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = ddim_scheduler
    if ddim_scheduler.config.prediction_type != "epsilon":
        raise SystemExit(
            f"P1-10 was trained with epsilon prediction, but the resolved scheduler config "
            f"says prediction_type={ddim_scheduler.config.prediction_type!r}. Refusing to "
            f"silently run inversion/reconstruction under a mismatched prediction type.")
    inverse_scheduler = DDIMInverseScheduler.from_config(ddim_scheduler.config)
    pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    vae, unet, text_encoder, tokenizer = pipe.vae, pipe.unet, pipe.text_encoder, pipe.tokenizer
    scaling = vae.config.scaling_factor

    with torch.no_grad():
        def encode_prompt(text):
            tok = tokenizer(text, padding="max_length", truncation=True,
                            max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
            return text_encoder(tok.input_ids)[0].to(dtype)
        cond_embeds = encode_prompt(args.prompt)
        uncond_embeds = encode_prompt("")

    # ------------------------------------------------------------------
    # Core diffusion machinery -- identical to P1-11's script except
    # reconstruct() now takes eta + a per-seed generator.
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
        arr = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        arr = arr.unsqueeze(0).to(device, dtype=dtype)
        return vae.encode(arr).latent_dist.mode() * scaling

    def decode_latent(latent):
        decoded = vae.decode(latent / scaling).sample
        if not torch.isfinite(decoded).all():
            raise RuntimeError("NaN/Inf in decoded output -- aborting.")
        return ((decoded[0].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    def invert(latent0, control, guidance_scale, n_steps_total, fraction, tag=""):
        # Never stochastic -- DDIMInverseScheduler.step() has no eta parameter,
        # inversion is only well-defined at eta=0.
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

    def reconstruct(latent_start, t_start, control, guidance_scale, n_steps_total, eta, generator, tag=""):
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
            latent = ddim_scheduler.step(noise_pred, t, latent, eta=eta, generator=generator,
                                        return_dict=True).prev_sample
            if args.debug_inversion:
                print(f"    [rec {tag}] step {i+1}/{len(used)} t={int(t)} eta={eta} "
                     f"mean={latent.mean().item():.4f} std={latent.std().item():.4f}")
        if not torch.isfinite(latent).all():
            raise RuntimeError("NaN/Inf in reconstructed latent -- aborting.")
        return latent, len(used)

    @torch.no_grad()
    def run_one_member(source_rgb, inv_ctrl_rgb, rec_ctrl_rgb, seed, tag):
        """One ensemble member: full invert()+reconstruct() trajectory for one
        seed. eta=0 reduces this to the same deterministic behaviour as
        infer_colour_source_ddim_inversion.py's own run_ddim_inversion()."""
        torch.manual_seed(seed)
        gen = torch.Generator(device=device).manual_seed(seed) if args.eta > 0 else None
        latent0 = encode_latent(source_rgb)
        inv_control = control_tensor_or_zero(inv_ctrl_rgb)
        rec_control = control_tensor_or_zero(rec_ctrl_rgb)
        latent_inv, t_end, k_actual = invert(
            latent0, inv_control, args.inversion_guidance,
            args.inversion_steps, args.inversion_fraction, tag=tag)
        latent_rec, n_rec = reconstruct(
            latent_inv, t_end, rec_control, args.guidance, args.translation_steps,
            args.eta, gen, tag=tag)
        out = decode_latent(latent_rec)
        if args.debug_inversion:
            print(f"    [meta {tag}] requested_fraction={args.inversion_fraction} eta={args.eta} "
                 f"actual_inv_steps={k_actual}/{args.inversion_steps} t_end={t_end} "
                 f"translation_steps_used={n_rec}/{args.translation_steps}")
        return out, t_end, k_actual, n_rec

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    man_path = out_dir / "eval_manifest.csv"
    man = open(man_path, "w", newline="")
    mw = csv.writer(man)
    mw.writerow(["seed", "source_mode", "crop_id", "slide", "frame", "x", "y",
                "output_path", "reference_path", "aperio_path",
                "method", "ensemble_role", "ensemble_n", "eta", "inversion_condition",
                "inversion_fraction", "actual_inversion_steps", "actual_start_timestep",
                "actual_translation_steps"])

    n_out = 0
    for i, c in enumerate(crops):
        tag = c["tag_id"]
        ref_path = out_dir / "reference" / f"{tag}.png"
        if not ref_path.exists():
            Image.fromarray(c["ref"]).save(ref_path)

        inv_ctrl_rgb = c["src"] if args.inversion_condition == "source" else None
        rec_ctrl_rgb = control_source_for(i)

        member_outputs = []
        for s in args.seeds:
            seed_val = seed_for(tag, s)
            run_tag = f"{tag}_seed{s}"
            out_member, t_end, k_actual, n_rec = run_one_member(
                c["src"], inv_ctrl_rgb, rec_ctrl_rgb, seed_val, run_tag)
            member_outputs.append(out_member)

            p = out_dir / "outputs" / f"{run_tag}_{args.source_mode}_member.png"
            Image.fromarray(out_member).save(p)
            mw.writerow([s, f"{args.source_mode}_member", tag, c["slide"], c["frame"], c["x"], c["y"],
                        os.path.relpath(p, out_dir), os.path.relpath(ref_path, out_dir),
                        c["aperio_path"], "ddim_inversion", "member", len(args.seeds), args.eta,
                        args.inversion_condition, args.inversion_fraction, k_actual, t_end, n_rec])
            man.flush()
            n_out += 1

        # Ensemble = pixel-wise mean of every member's decoded RGB output for
        # this crop, in decoded RGB space (not latent space -- the VAE decoder
        # is non-linear, so averaging post-decode is the well-defined choice).
        ensemble_out = np.mean(np.stack(member_outputs, axis=0), axis=0).round().clip(0, 255).astype(np.uint8)
        p = out_dir / "outputs" / f"{tag}_{args.source_mode}_ensemble.png"
        Image.fromarray(ensemble_out).save(p)
        mw.writerow(["ensemble", f"{args.source_mode}_ensemble", tag, c["slide"], c["frame"], c["x"], c["y"],
                    os.path.relpath(p, out_dir), os.path.relpath(ref_path, out_dir),
                    c["aperio_path"], "ddim_inversion_ensemble", "ensemble", len(args.seeds), args.eta,
                    args.inversion_condition, args.inversion_fraction, "", "", ""])
        man.flush()
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
        "script": "infer_colour_source_ddim_inversion_ensemble.py",
        "checkpoint_lora": args.lora, "checkpoint_controlnet": args.controlnet,
        "direction": args.direction, "source_mode": args.source_mode,
        "inversion_condition": args.inversion_condition,
        "prediction_type": ddim_scheduler.config.prediction_type,
        "inversion_steps": args.inversion_steps, "translation_steps": args.translation_steps,
        "inversion_fraction": args.inversion_fraction, "inversion_guidance": args.inversion_guidance,
        "guidance": args.guidance, "controlnet_conditioning_scale": args.controlnet_scale,
        "eta": args.eta, "ensemble_n": len(args.seeds),
        "seeds": args.seeds, "dtype": str(dtype), "model": args.model, "prompt": args.prompt,
        "git_commit": git_commit(),
    }
    with open(out_dir / "run_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nWrote {n_out} output rows ({len(args.seeds)} member(s) + 1 ensemble per crop) across "
         f"{len(crops)} crops, source_mode={args.source_mode}, eta={args.eta}.")
    print(f"Manifest: {man_path}")
    print(f"Metadata: {out_dir / 'run_metadata.json'}")
    print("Next: score with score_p1_10_ablation.py -- source_mode column separates "
         "'<mode>_member' from '<mode>_ensemble' so aggregation/paired-win-rate answers "
         "'does ensembling help' directly.")


if __name__ == "__main__":
    main()
