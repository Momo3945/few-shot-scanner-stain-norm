#!/usr/bin/env python3
"""
train_colour_lora.py -- scanner-colour LoRA training on a frozen SD 1.5 base (A2).

Objective (see project notes): a target-domain denoising LoRA. The adapter is
trained with the standard diffusion loss on the TARGET scanner's crops so it
learns that scanner's colour statistics; the directional mapping is applied at
inference time via img2img (start from the source-scanner latent). This respects
the coordinate-corresponding (not pixel-exact) nature of the training pairs -- no
pixel-reconstruction loss that would penalise sub-crop spatial offset as colour error.

  --direction A2H : train on *_hamamatsu.png  (applied to Aperio inputs at inference)
  --direction H2A : train on *_aperio.png     (applied to Hamamatsu inputs; cycle consistency)

Only LoRA params in the UNet attention layers are trained; VAE, text encoder, and
UNet base weights stay frozen. Rank is an experiment variable (proposal: r in {4,8}).

Environment: run on a GPU node with the model cached locally. Set HF_HOME to the
cache and the script runs offline (no network needed on the compute node).

Usage
-----
    # 5-step smoke test FIRST (validates env + loop cheaply):
    python train_colour_lora.py --pairs-dir <pairs>/train --direction A2H \
        --output-dir <out>/smoke --smoke

    # real run:
    python train_colour_lora.py --pairs-dir <pairs>/train --direction A2H \
        --rank 8 --train-steps 1000 --output-dir <out>/a2h_r8

Dependencies: torch, diffusers, transformers, peft, safetensors, Pillow, numpy.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import time
from pathlib import Path


# ----------------------------------------------------------------------
# Pure helper (import-light so it can be unit-tested without torch)
# ----------------------------------------------------------------------
def list_target_images(train_dir: str, direction: str) -> list[str]:
    """Return the sorted target-scanner crop paths for the given direction.

    A2H trains on the Hamamatsu crops; H2A trains on the Aperio crops.
    """
    suffix = "_hamamatsu.png" if direction == "A2H" else "_aperio.png"
    paths = sorted(glob.glob(os.path.join(train_dir, f"*{suffix}")))
    return paths


def parse_args():
    ap = argparse.ArgumentParser(description="Train a scanner-colour LoRA on SD 1.5.")
    ap.add_argument("--pairs-dir", required=True,
                    help="Directory with *_aperio.png / *_hamamatsu.png crop pairs (the train/ folder).")
    ap.add_argument("--direction", choices=["A2H", "H2A"], required=True)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                    help="HF repo id (resolved from HF_HOME cache; runs offline).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank (proposal: 4 or 8).")
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 -> defaults to rank.")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--train-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--mixed-precision", choices=["bf16", "fp16", "no"], default="bf16")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--online", action="store_true",
                    help="Allow HF network access (default: offline, use local cache).")
    ap.add_argument("--smoke", action="store_true",
                    help="5-step dry run: overrides train-steps/save-every for a quick env+loop check.")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.train_steps = 5
        args.save_every = 5
        args.log_every = 1
    if args.lora_alpha == 0:
        args.lora_alpha = args.rank

    # Offline by default -- the weights are cached; the compute node may lack network.
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Heavy imports after arg parse so --help and import errors are fast/clear.
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    # -- reproducibility --
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device found -- training on CPU will be unusably slow. "
              "This script is meant for a GPU node.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    target_paths = list_target_images(args.pairs_dir, args.direction)
    if not target_paths:
        raise SystemExit(f"No target crops for direction {args.direction} in {args.pairs_dir}. "
                         f"Expected *_{'hamamatsu' if args.direction=='A2H' else 'aperio'}.png files.")
    print(f"Direction {args.direction}: {len(target_paths)} target crops "
          f"({'Hamamatsu' if args.direction=='A2H' else 'Aperio'} scanner).")

    class CropDataset(Dataset):
        def __init__(self, paths, resolution):
            self.paths = paths
            self.res = resolution

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            if img.size != (self.res, self.res):
                img = img.resize((self.res, self.res), Image.LANCZOS)
            arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # [-1, 1]
            return torch.from_numpy(arr).permute(2, 0, 1)           # CHW

    loader = DataLoader(CropDataset(target_paths, args.resolution),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)

    # ------------------------------------------------------------------
    # Model components (all frozen except the LoRA we inject)
    # ------------------------------------------------------------------
    print(f"Loading SD 1.5 components from cache ({args.model}) ...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    trainable = [p for p in unet.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"LoRA rank {args.rank} (alpha {args.lora_alpha}) -> {n_train:,} trainable params "
          f"({n_train / sum(p.numel() for p in unet.parameters()) * 100:.3f}% of UNet).")

    vae.to(device); text_encoder.to(device); unet.to(device)
    vae.eval(); text_encoder.eval(); unet.train()

    # Precompute the (fixed) prompt embedding once.
    with torch.no_grad():
        tok = tokenizer(args.prompt, padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
        prompt_embeds = text_encoder(tok.input_ids)[0]   # [1, 77, 768]

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": None}[args.mixed_precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    scaling = vae.config.scaling_factor

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log_path = out_dir / "loss_log.csv"
    log_fh = open(log_path, "w", newline="")
    log_w = csv.writer(log_fh); log_w.writerow(["step", "loss", "sec"])

    print(f"Training for {args.train_steps} steps (batch {args.batch_size}, lr {args.lr}, "
          f"{args.mixed_precision}) ...")
    step = 0
    t0 = time.time()
    running = 0.0
    done = False
    while not done:
        for pixel_values in loader:
            pixel_values = pixel_values.to(device, dtype=torch.float32)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * scaling
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (bsz,), device=device).long()
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)

            if noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                target = noise

            enc = prompt_embeds.expand(bsz, -1, -1)

            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=(amp_dtype is not None and device.type == "cuda")):
                model_pred = unet(noisy, timesteps, encoder_hidden_states=enc).sample
                loss = F.mse_loss(model_pred.float(), target.float())

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()

            step += 1
            running += loss.item()
            if step % args.log_every == 0:
                avg = running / args.log_every
                running = 0.0
                sec = time.time() - t0
                print(f"  step {step:5d}/{args.train_steps}  loss {avg:.4f}  ({sec:.1f}s)")
                log_w.writerow([step, f"{avg:.6f}", f"{sec:.1f}"]); log_fh.flush()

            if step % args.save_every == 0 or step >= args.train_steps:
                ckpt = out_dir / (f"checkpoint-{step}" if step < args.train_steps else "final")
                ckpt.mkdir(parents=True, exist_ok=True)
                lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
                StableDiffusionPipeline.save_lora_weights(
                    save_directory=str(ckpt), unet_lora_layers=lora_state,
                    safe_serialization=True)
                print(f"  saved LoRA -> {ckpt}")

            if step >= args.train_steps:
                done = True
                break

    log_fh.close()
    with open(out_dir / "training_config.json", "w") as fh:
        json.dump(vars(args) | {"n_target_crops": len(target_paths),
                                "trainable_params": n_train,
                                "prediction_type": noise_scheduler.config.prediction_type},
                  fh, indent=2)
    print(f"\nDone. Final LoRA in {out_dir/'final'}. Config + loss log written.")


if __name__ == "__main__":
    main()
