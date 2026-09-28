#!/usr/bin/env python3
"""
train_lora_pixcell.py -- PXC-01b: LoRA training-cost measurement (wall-clock
+ peak VRAM only, mirrors PROBE-SD35-TICKETS.md PR-01's exact discipline).
See tickets/PROBE-PIXCELL-TICKETS.md.

Not part of the graded proposal scope -- see that ticket file's own
caveat. This is a COST measurement, not a quality benchmark -- PXC-02
already found (job 61123) that PixCell's embedding-only conditioning
cannot hold structure well enough for this project's registered-pair
metrics, so this script's output checkpoint is not evaluated for quality
here; only training wall-clock and peak VRAM are the deliverable.

Reuses PXC-01a's fully-confirmed loading/wiring (src/eval/
probe_pixcell_pipeline.py, job 61121) verbatim: UNI2-h via timm with its
own documented timm_kwargs, the full PixCell DiffusionPipeline (gives
correctly-wired pipe.transformer/pipe.vae/pipe.scheduler in one call
rather than re-guessing component loading). The one genuinely new piece
here -- the training-time transformer call signature and the "learned
sigma" output-channel-halving -- is copied directly from the cached
pipeline.py's own denoising loop (read directly off the cluster, not
guessed):
    noise_pred = transformer(latent_model_input, encoder_hidden_states=...,
                              timestep=..., added_cond_kwargs={},
                              return_dict=False)[0]
    if transformer.config.out_channels // 2 == latent_channels:
        noise_pred = noise_pred.chunk(2, dim=1)[0]  # discard learned variance half

LoRA target modules restricted to attn2 (cross-attention) projections
only, matching PixCell's own virtual_staining/train_lora.py precedent for
this exact task shape (H&E->IHC domain shift; self-attention/attn1 left
untouched) -- rank 8 (this project's own established colour-LoRA
convention, not their rank-4 default; P2-05 already found rank barely
matters for this kind of task).

Direction: A->H (source=Aperio crop's own UNI2-h embedding, target=
Hamamatsu crop, encoded to latents as the denoising target) -- matches
this project's A2H convention elsewhere.

Usage
-----
    # smoke test:
    python train_lora_pixcell.py --pairs-dir <pairs>/train_256 \
        --output-dir <out>/pxc_01b_smoke --smoke

    # real run (cost measurement only):
    python train_lora_pixcell.py --pairs-dir <pairs>/train_256 \
        --output-dir <out>/pxc_01b_lora --train-steps 1000

Dependencies: torch, diffusers, transformers, timm, peft, Pillow, numpy.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

HELD_OUT_SLIDES = {"A06", "A08", "A09", "A13", "A16"}


def build_pairs(pairs_dir: str) -> list[dict]:
    """Same leak-checked convention as every other training script in this
    project (train_p3_07c_lora_sdxl.py etc.) -- A03/H03 only, both sides
    must exist."""
    aperio_paths = sorted(glob.glob(os.path.join(pairs_dir, "*_aperio.png")))
    pairs = []
    for ap in aperio_paths:
        stem = ap[: -len("_aperio.png")]
        hp = stem + "_hamamatsu.png"
        if not os.path.exists(hp):
            raise SystemExit(f"Missing Hamamatsu side for pair {stem} (expected {hp}).")
        pair_id = os.path.basename(stem)
        slide = pair_id.split("_")[0]
        if slide in HELD_OUT_SLIDES:
            raise SystemExit(f"LEAK CHECK FAILED: pair {pair_id} belongs to held-out slide {slide}.")
        pairs.append({"pair_id": pair_id, "slide": slide, "aperio_path": ap, "hamamatsu_path": hp})
    if not pairs:
        raise SystemExit(f"No *_aperio.png/*_hamamatsu.png pairs found in {pairs_dir}.")
    bad = {p["slide"] for p in pairs} - {"A03"}
    if bad:
        raise SystemExit(f"LEAK CHECK FAILED: unexpected slide(s) {bad} in {pairs_dir} (A03/H03 only).")
    return pairs


def parse_args():
    ap = argparse.ArgumentParser(description="PXC-01b: PixCell LoRA training-cost measurement.")
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--pixcell-repo", default="StonyBrook-CVLab/PixCell-256")
    ap.add_argument("--pixcell-pipeline-repo", default="StonyBrook-CVLab/PixCell-pipeline")
    ap.add_argument("--sd35-vae-repo", default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--uni2h-repo", default="hf-hub:MahmoodLab/UNI2-h")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--rank", type=int, default=8, help="This project's own colour-LoRA convention.")
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 -> defaults to rank.")
    ap.add_argument("--train-steps", type=int, default=1000,
                    help="Matches this project's SD1.5/SD3.5 cost-comparison step count.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--online", action="store_true",
                    help="trust_remote_code needs Hub access on every load -- confirmed empirically "
                         "(PXC-02 Part B, job 61122/61123).")
    ap.add_argument("--smoke", action="store_true",
                    help="5-step dry run: env/loop check only, matching every other LoRA script's "
                         "own --smoke convention in this project.")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.train_steps = 5
        args.save_every = 5
        args.log_every = 1
    if args.lora_alpha == 0:
        args.lora_alpha = args.rank
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import timm
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from diffusers import AutoencoderKL, DiffusionPipeline
    from peft import LoraConfig, get_peft_model
    from peft.utils import get_peft_model_state_dict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("No CUDA device -- this script is meant for a GPU node.")
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(args.pairs_dir)
    print(f"PXC-01b -- {len(pairs)} A03/H03 pairs from {args.pairs_dir}.")

    # ------------------------------------------------------------------
    # Load UNI2-h (frozen), SD3.5 VAE (frozen, bf16), PixCell pipeline --
    # exact confirmed pattern from probe_pixcell_pipeline.py (PXC-01a,
    # job 61121).
    # ------------------------------------------------------------------
    print(f"Loading UNI2-h ({args.uni2h_repo}) via timm ...")
    uni2h_kwargs = dict(
        img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
        embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
        reg_tokens=8, dynamic_img_size=True,
    )
    uni_model = timm.create_model(args.uni2h_repo, pretrained=True, **uni2h_kwargs)
    uni_model.eval().to(device)
    for p in uni_model.parameters():
        p.requires_grad_(False)
    uni_config = timm.data.resolve_data_config(uni_model.pretrained_cfg)
    uni_transform = timm.data.create_transform(**uni_config)

    print(f"Loading SD3.5 VAE ({args.sd35_vae_repo}, bf16) ...")
    sd35_vae = AutoencoderKL.from_pretrained(
        args.sd35_vae_repo, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    sd35_vae.requires_grad_(False)
    sd35_vae.eval()

    print(f"Loading PixCell pipeline ({args.pixcell_repo}) ...")
    pipe = DiffusionPipeline.from_pretrained(
        args.pixcell_repo, vae=sd35_vae,
        custom_pipeline=args.pixcell_pipeline_repo,
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    transformer.requires_grad_(False)

    latent_channels = transformer.config.in_channels
    learned_sigma = (transformer.config.out_channels // 2 == latent_channels)
    print(f"Transformer: in_channels={latent_channels} out_channels="
          f"{transformer.config.out_channels} learned_sigma={learned_sigma}")

    # ------------------------------------------------------------------
    # LoRA -- attn2 (cross-attention) projections only, matching PixCell's
    # own virtual_staining/train_lora.py precedent for this task shape.
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha, init_lora_weights="gaussian",
        target_modules=["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"],
    )
    # PixCellTransformer2DModel is a custom community class (not a core
    # diffusers ModelMixin subclass with PeftAdapterMixin) -- confirmed
    # empirically (job 61134): it has no .add_adapter() at all. Use peft's
    # own backend-agnostic get_peft_model() instead, which works on any
    # nn.Module.
    transformer = get_peft_model(transformer, lora_config)
    n_lora = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    if n_lora == 0:
        raise SystemExit(
            "get_peft_model produced 0 trainable params -- target_modules "
            "['attn2.to_q','attn2.to_k','attn2.to_v','attn2.to_out.0'] doesn't match this "
            "transformer's actual module names. Inspect transformer.named_modules() directly.")
    print(f"LoRA (rank {args.rank}, attn2 only): {n_lora:,} trainable params.")

    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad], lr=args.lr)

    vae_scale = sd35_vae.config.scaling_factor
    vae_shift = getattr(sd35_vae.config, "shift_factor", 0.0) or 0.0

    def load_rgb(path, resize=None):
        img = Image.open(path).convert("RGB")
        if resize and img.size != (resize, resize):
            img = img.resize((resize, resize), Image.LANCZOS)
        return np.asarray(img)

    def embed_source(aperio_rgb):
        img = Image.fromarray(aperio_rgb)
        tensor = uni_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = uni_model(tensor)
        return raw.unsqueeze(1).to(torch.bfloat16)  # (1, 1, 1536), caption_num_tokens=1 confirmed PXC-01a

    def encode_target(hamamatsu_rgb):
        arr = torch.from_numpy(hamamatsu_rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        arr = arr.unsqueeze(0).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            latents = (sd35_vae.encode(arr).latent_dist.sample() - vae_shift) * vae_scale
        return latents

    # ------------------------------------------------------------------
    # Training loop -- step-based, matching this project's other LoRA
    # scripts' own convention.
    # ------------------------------------------------------------------
    log_path = out_dir / "loss_log.csv"
    import csv
    log_fh = open(log_path, "w", newline="")
    log_w = csv.writer(log_fh); log_w.writerow(["step", "loss", "sec"])

    print(f"Training for {args.train_steps} steps (batch 1, lr {args.lr}) ...")
    transformer.train()
    step = 0
    t0 = time.time()
    running = 0.0
    rng = np.random.default_rng(args.seed)
    while step < args.train_steps:
        p = pairs[rng.integers(0, len(pairs))]
        aperio_rgb = load_rgb(p["aperio_path"], resize=args.resolution)
        hamamatsu_rgb = load_rgb(p["hamamatsu_path"], resize=args.resolution)

        uni_embeds = embed_source(aperio_rgb)
        latents = encode_target(hamamatsu_rgb)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
        noisy = scheduler.add_noise(latents, noise, timesteps)

        model_pred = transformer(
            noisy, encoder_hidden_states=uni_embeds, timestep=timesteps,
            added_cond_kwargs={}, return_dict=False,
        )[0]
        if learned_sigma:
            model_pred = model_pred.chunk(2, dim=1)[0]

        loss = F.mse_loss(model_pred.float(), noise.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1
        running += loss.item()
        if step % args.log_every == 0:
            avg = running / args.log_every
            running = 0.0
            sec = time.time() - t0
            print(f"  step {step:5d}/{args.train_steps}  loss {avg:.4f}  ({sec:.1f}s)")
            log_w.writerow([step, f"{avg:.6f}", f"{sec:.1f}"]); log_fh.flush()

        if step % args.save_every == 0 or step >= args.train_steps:
            ckpt_path = out_dir / (f"checkpoint-{step}.pt" if step < args.train_steps else "final.pt")
            torch.save(get_peft_model_state_dict(transformer), ckpt_path)
            print(f"  saved LoRA state dict -> {ckpt_path}")

    log_fh.close()
    total_sec = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\nTotal training wall-clock: {total_sec:.1f}s ({total_sec/60:.1f} min) "
          f"for {args.train_steps} steps ({total_sec/args.train_steps:.3f}s/step).")
    print(f"Peak VRAM: {peak_vram_gb:.2f} GB")

    with open(out_dir / "training_config.json", "w") as fh:
        json.dump(vars(args) | {
            "n_pairs": len(pairs), "n_lora_params": n_lora,
            "total_train_sec": round(total_sec, 1),
            "sec_per_step": round(total_sec / args.train_steps, 4),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "learned_sigma": learned_sigma, "latent_channels": latent_channels,
        }, fh, indent=2)
    print(f"Config + cost measurement written -> {out_dir / 'training_config.json'}")


if __name__ == "__main__":
    main()
