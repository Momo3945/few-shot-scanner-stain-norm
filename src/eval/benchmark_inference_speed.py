#!/usr/bin/env python3
"""
benchmark_inference_speed.py -- P2-16: per-crop inference time of every method on one GPU node.

The proposal's "fast inference" requirement (H4: 4-step LCM vs 50-step DDIM; SDXL compared on
"the same colour, structure, and speed metrics") was never measured head to head: the timings in
the tickets include model loading and ECC registration. This times the per-crop call only, on the
same node and the same held-out Aperio crops (no registration, no scoring, no outputs written):
each method is built once, warmed up on --warmup crops, then timed on --n crops with
torch.cuda.synchronize() around every call.

Methods (pipeline construction copied from each method's validated inference script):
  sd15_inv_f3     final pipeline: P1-10 checkpoint, DDIM inversion 50 + 50 steps (guidance 1.0/2.0),
                  stock-VAE A_V, F3 fusion (sigma 8, beta 0.5)  [normalize_lizard_inversion_fusion.py]
  sd15_lcm_s6     P1-10 + generic LCM-LoRA, 6 steps, strength 0.70, guidance 2.0 (P1-12 best)
  sd15_lcm_4step  same, 4 steps at strength 1.0 (4 real denoising steps)  [infer_colour_translation_lcm.py]
  sdxl_1024       P3-07c 144-pair SDXL @1024, img2img strength 0.50, 50-step DDIM, guidance 2.0,
                  fp32 VAE  [infer_colour_translation_sdxl.py]; timed on 1024 px crops
  stainnet / paramnet / staingan   P2-11b A2H checkpoints on GPU  [infer_learned_baseline.py]
  macenko / reinhard / histmatch   CPU, fitted once on one Hamamatsu crop  [baseline_methods.py]

Writes timings.csv (one row per method x crop) and summary.csv (mean/sd/median s per crop, real
denoising steps, crop size, peak GPU memory). Run from src/eval (sibling imports).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

ALL_METHODS = ["sd15_inv_f3", "sd15_lcm_s6", "sd15_lcm_4step", "sdxl_1024",
               "stainnet", "paramnet", "staingan", "macenko", "reinhard", "histmatch"]


def parse_args():
    ap = argparse.ArgumentParser(description="P2-16 per-crop inference timing.")
    ap.add_argument("--root", required=True, help="MITOS held-out root (heldout paths are relative to this).")
    ap.add_argument("--heldout", required=True, help="heldout_frames.csv.")
    ap.add_argument("--sd15-ckpt", required=True, help="lora/a2h_cond_r8/best (P1-10).")
    ap.add_argument("--sdxl-ckpt", required=True, help="lora/a2h_cond_r8_sdxl_1024_overlap144/best (P3-07c).")
    ap.add_argument("--baselines-dir", required=True, help="baselines/ (P2-11b a2h_<method>/<method>_a2h.pt).")
    ap.add_argument("--sd15-model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--sdxl-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16, help="Timed crops per method.")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    return ap.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for baselines.models
    from canny import extract_canny_control_image
    from fuse_source_detail import fuse_f3
    from registration import read_rgb

    if not torch.cuda.is_available():
        raise SystemExit("ABORT: no CUDA device -- timing must run on a GPU node.")
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- crops: top-left 512 (and 1024 for SDXL) of the first held-out frames, one per frame ----
    with open(args.heldout, newline="") as fh:
        rows = list(csv.DictReader(fh))
    need = args.n + args.warmup
    frames, ref_rgb = [], None
    for r in rows:
        a = read_rgb(Path(args.root) / r["aperio_path"].replace("\\", "/"))
        if a.shape[0] >= 1024 and a.shape[1] >= 1024:
            frames.append((f"{r['aperio_slide']}_{r['frame_id']}", a))
            if ref_rgb is None:
                ref_rgb = read_rgb(Path(args.root) / r["hamamatsu_path"].replace("\\", "/"))[:512, :512]
        if len(frames) == need:
            break
    if len(frames) < need:
        raise SystemExit(f"only {len(frames)} frames >= 1024 px, need {need}")
    crops512 = [(k, a[:512, :512].copy()) for k, a in frames]
    crops1024 = [(k, a[:1024, :1024].copy()) for k, a in frames]
    print(f"GPU {gpu_name}; {len(frames)} crops ({args.warmup} warm-up + {args.n} timed)", flush=True)

    def control6(rgb, dtype=None):
        canny = extract_canny_control_image(rgb)
        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        canny_t = torch.from_numpy(canny.astype(np.float32) / 255.0).permute(2, 0, 1)
        t = torch.cat([rgb_t, canny_t], dim=0).unsqueeze(0)
        return t if dtype is None else t.to(device, dtype=dtype)

    # ---- builders: each returns (fn(rgb) -> output, real denoising steps, crop list) ----
    def build_sd15_inv_f3():
        from diffusers import (ControlNetModel, DDIMInverseScheduler, DDIMScheduler,
                               StableDiffusionControlNetImg2ImgPipeline)
        dtype = torch.float16
        cn = ControlNetModel.from_pretrained(f"{args.sd15_ckpt}/controlnet", torch_dtype=dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            args.sd15_model, controlnet=cn, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
        ddim = DDIMScheduler.from_config(pipe.scheduler.config)
        inv = DDIMInverseScheduler.from_config(ddim.config)
        pipe.load_lora_weights(args.sd15_ckpt, weight_name="pytorch_lora_weights.safetensors")
        pipe.to(device)
        vae, unet, te, tok = pipe.vae, pipe.unet, pipe.text_encoder, pipe.tokenizer
        sc = vae.config.scaling_factor
        with torch.no_grad():
            def emb(t):
                ids = tok(t, padding="max_length", truncation=True, max_length=tok.model_max_length,
                          return_tensors="pt").to(device)
                return te(ids.input_ids)[0].to(dtype)
            cond, uncond = emb(args.prompt), emb("")

        def eps(lat, t, ctl, g):
            if g <= 1.0:
                d, m = cn(lat, t, encoder_hidden_states=cond, controlnet_cond=ctl, conditioning_scale=1.0,
                          return_dict=False)
                return unet(lat, t, encoder_hidden_states=cond, down_block_additional_residuals=d,
                            mid_block_additional_residual=m).sample
            li, ei, ci = torch.cat([lat, lat]), torch.cat([uncond, cond]), torch.cat([ctl, ctl])
            d, m = cn(li, t, encoder_hidden_states=ei, controlnet_cond=ci, conditioning_scale=1.0, return_dict=False)
            nu, nc = unet(li, t, encoder_hidden_states=ei, down_block_additional_residuals=d,
                          mid_block_additional_residual=m).sample.chunk(2)
            return nu + g * (nc - nu)

        def dec(lat):
            x = vae.decode(lat / sc).sample
            return ((x[0].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        @torch.no_grad()
        def run(rgb):
            ctl = control6(rgb, dtype)
            x = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
            lat0 = vae.encode(x.to(device, dtype=dtype)).latent_dist.mode() * sc
            inv.set_timesteps(50, device=device)
            lat = lat0
            for t in inv.timesteps:
                lat = inv.step(eps(lat, t, ctl, 1.0), t, lat).prev_sample
            ddim.set_timesteps(50, device=device)
            for t in ddim.timesteps:
                lat = ddim.step(eps(lat, t, ctl, 2.0), t, lat, eta=0.0).prev_sample
            return fuse_f3(rgb, dec(lat), dec(lat0), sigma=8.0, beta=0.5)
        return run, "50 inv + 50", crops512

    def build_sd15_lcm(steps, strength):
        def builder():
            from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetImg2ImgPipeline
            cn = ControlNetModel.from_pretrained(f"{args.sd15_ckpt}/controlnet", torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                args.sd15_model, controlnet=cn, torch_dtype=torch.float16,
                safety_checker=None, requires_safety_checker=False)
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights(args.sd15_ckpt, weight_name="pytorch_lora_weights.safetensors",
                                   adapter_name="colour")
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5",
                                   weight_name="pytorch_lora_weights.safetensors", adapter_name="lcm")
            pipe.set_adapters(["colour", "lcm"], adapter_weights=[1.0, 1.0])
            pipe.to(device)
            pipe.set_progress_bar_config(disable=True)

            def run(rgb):
                g = torch.Generator(device=device).manual_seed(0)
                return pipe(prompt=args.prompt, image=Image.fromarray(rgb), strength=strength,
                            num_inference_steps=steps, guidance_scale=2.0, control_image=control6(rgb),
                            controlnet_conditioning_scale=1.0, generator=g).images[0]
            return run, str(int(steps * strength)), crops512
        return builder

    def build_sdxl():
        from diffusers import (AutoencoderKL, ControlNetModel, DDIMScheduler,
                               StableDiffusionXLControlNetImg2ImgPipeline)
        vae = AutoencoderKL.from_pretrained(args.sdxl_model, subfolder="vae", torch_dtype=torch.float32)
        cn = ControlNetModel.from_pretrained(f"{args.sdxl_ckpt}/controlnet", torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            args.sdxl_model, controlnet=cn, vae=vae, torch_dtype=torch.float16)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(args.sdxl_ckpt, weight_name="pytorch_lora_weights.safetensors", adapter_name="colour")
        pipe.set_adapters(["colour"], adapter_weights=[1.0])
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        def run(rgb):
            g = torch.Generator(device=device).manual_seed(0)
            return pipe(prompt=args.prompt, image=Image.fromarray(rgb), strength=0.50, num_inference_steps=50,
                        guidance_scale=2.0, control_image=control6(rgb), controlnet_conditioning_scale=1.0,
                        generator=g).images[0]
        return run, "25", crops1024

    def build_learned(method):
        def builder():
            from infer_learned_baseline import build_model, make_transform
            ck = torch.load(Path(args.baselines_dir) / f"a2h_{method}" / f"{method}_a2h.pt", map_location="cpu")
            if ck["method"] != method or ck["direction"] != "A2H":
                raise SystemExit(f"checkpoint is {ck['method']}/{ck['direction']}, expected {method}/A2H")
            return make_transform(build_model(method, ck).to(device), torch.device(device)), "-", crops512
        return builder

    def build_classical(method):
        def builder():
            import baseline_methods as bm
            if method == "histmatch":
                return (lambda rgb: bm.histogram_match_normalize(rgb, ref_rgb)), "-", crops512
            norm = bm.MacenkoNormalizer() if method == "macenko" else bm.ReinhardNormalizer()
            norm.fit(ref_rgb)  # one-off fit on the reference, not timed
            return norm.transform, "-", crops512
        return builder

    builders = {"sd15_inv_f3": build_sd15_inv_f3, "sd15_lcm_s6": build_sd15_lcm(6, 0.70),
                "sd15_lcm_4step": build_sd15_lcm(4, 1.0), "sdxl_1024": build_sdxl,
                "stainnet": build_learned("stainnet"), "paramnet": build_learned("paramnet"),
                "staingan": build_learned("staingan"), "macenko": build_classical("macenko"),
                "reinhard": build_classical("reinhard"), "histmatch": build_classical("histmatch")}

    trows, srows = [], []
    for m in args.methods:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fn, real_steps, crops = builders[m]()
        times = []
        for i, (key, rgb) in enumerate(crops):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(rgb)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if np.asarray(out).shape[:2] != rgb.shape[:2]:
                raise RuntimeError(f"{m}: output shape {np.asarray(out).shape} != input {rgb.shape}")
            if i >= args.warmup:
                times.append(dt)
                trows.append({"method": m, "crop": key, "seconds": round(dt, 4)})
        peak = torch.cuda.max_memory_allocated() / 2**30
        s = {"method": m, "crop_px": crops[0][1].shape[0], "real_denoising_steps": real_steps, "n": len(times),
             "mean_s": round(statistics.mean(times), 4), "sd_s": round(statistics.stdev(times), 4),
             "median_s": round(statistics.median(times), 4), "peak_gpu_gib": round(peak, 2)}
        srows.append(s)
        print(json.dumps(s), flush=True)
        del fn
        import gc
        gc.collect()

    for name, rws in (("timings.csv", trows), ("summary.csv", srows)):
        with open(out_dir / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rws[0]))
            w.writeheader()
            w.writerows(rws)
    json.dump({**vars(args), "gpu": gpu_name, "node": os.environ.get("SLURMD_NODENAME", "")},
              open(out_dir / "run_metadata.json", "w"), indent=2)
    print(f"\nDone -> {out_dir}/summary.csv")


if __name__ == "__main__":
    main()
