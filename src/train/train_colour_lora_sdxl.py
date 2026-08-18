#!/usr/bin/env python3
"""
train_colour_lora_sdxl.py -- scanner-colour LoRA training on a frozen SDXL base
(P3-03: transfer of A2's training objective to SDXL).

Same objective as train_colour_lora.py: a target-domain denoising LoRA, trained
with the standard diffusion loss on the TARGET scanner's crops so it learns that
scanner's colour statistics; the directional mapping is applied at inference time
via img2img. This script only trains --direction A2H (P3-03's scope: transfer
ONLY the A4 config used by the SD1.5-vs-SDXL comparison, A2H direction -- H2A is
out of scope here, see tickets/PHASE3-TICKETS.md P3-03).

Differences from train_colour_lora.py (SD1.5), all required by SDXL's architecture:
  - Two tokenizers/text encoders (tokenizer/tokenizer_2, text_encoder/text_encoder_2)
    instead of one. Their hidden states are concatenated for encoder_hidden_states;
    text_encoder_2's pooled output feeds SDXL's required `added_cond_kwargs`.
  - `added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}`
    is a hard SDXL UNet input requirement, not optional. add_time_ids is built from
    (original_size, crop_coords_top_left, target_size) = (512,512)/(0,0)/(512,512) --
    the HONEST values for this data (pairs/train crops are exactly 512x512 already,
    see extract_pairs.py; do NOT spoof 1024, per the P3-03 ticket's resolution decision).
  - VAE forced to fp32 and kept OUT of the autocast region: SDXL's official VAE is
    documented to NaN under fp16. Do NOT "fix" this back to fp16 -- that would
    silently reintroduce NaN latents. Only the VAE encode is affected; UNet training
    still runs under bf16/fp16 autocast as before.
  - Gradient checkpointing enabled on the UNet: SDXL's UNet (~2.6B params vs SD1.5's
    ~860M) is far heavier; needed to fit LoRA fine-tuning (batch 1, bf16 autocast,
    fp32 VAE) inside a 24GB RTX 3090. Not present in the SD1.5 script -- new
    requirement, not a copy-paste omission.
  - Checkpoint save via StableDiffusionXLPipeline.save_lora_weights (SDXL analogue
    of StableDiffusionPipeline.save_lora_weights) -- same
    pytorch_lora_weights.safetensors output filename convention.

Everything else (CropDataset, LoRA config/target_modules, AdamW over LoRA params
only, save/log cadence, CLI shape, --smoke) is an unchanged port.

Usage
-----
    # 5-step smoke test FIRST (validates env + loop cheaply):
    python train_colour_lora_sdxl.py --pairs-dir <pairs>/train --direction A2H \
        --output-dir <out>/smoke --smoke

    # real run:
    python train_colour_lora_sdxl.py --pairs-dir <pairs>/train --direction A2H \
        --rank 8 --train-steps 1000 --output-dir <out>/a2h_r8_sdxl

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
    ap = argparse.ArgumentParser(description="Train a scanner-colour LoRA on SDXL.")
    ap.add_argument("--pairs-dir", required=True,
                    help="Directory with *_aperio.png / *_hamamatsu.png crop pairs (the train/ folder).")
    ap.add_argument("--direction", choices=["A2H", "H2A"], required=True)
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0",
                    help="HF repo id (resolved from HF_HOME cache; runs offline).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank (proposal: 4 or 8).")
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 -> defaults to rank.")
    ap.add_argument("--resolution", type=int, default=512,
                    help="pairs/train crops are already exactly this size -- honest "
                         "SDXL micro-conditioning value, not spoofed to 1024 (P3-03).")
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

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
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
    # Data (unchanged from train_colour_lora.py)
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
    print(f"Loading SDXL components from cache ({args.model}) ...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.model, subfolder="text_encoder_2")
    # VAE forced to fp32 -- SDXL's official VAE NaNs under fp16. Kept out of the
    # autocast region entirely (encode below runs outside torch.autocast). Do not
    # change this dtype back to match the UNet/text-encoder precision.
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    # SDXL's UNet is ~3x SD1.5's -- gradient checkpointing keeps LoRA fine-tuning
    # (batch 1, bf16 autocast, fp32 VAE) inside a 24GB RTX 3090. Not needed for
    # SD1.5's much smaller UNet; new for this script, not an omission there.
    unet.enable_gradient_checkpointing()

    trainable = [p for p in unet.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"LoRA rank {args.rank} (alpha {args.lora_alpha}) -> {n_train:,} trainable params "
          f"({n_train / sum(p.numel() for p in unet.parameters()) * 100:.3f}% of UNet).")

    vae.to(device)
    text_encoder.to(device); text_encoder_2.to(device); unet.to(device)
    vae.eval(); text_encoder.eval(); text_encoder_2.eval(); unet.train()

    # Precompute the (fixed) prompt embedding once, via BOTH text encoders --
    # SDXL requires encoder_hidden_states from the concatenated penultimate
    # hidden states of text_encoder + text_encoder_2, plus a separate pooled
    # embedding from text_encoder_2 for added_cond_kwargs["text_embeds"].
    with torch.no_grad():
        tok = tokenizer(args.prompt, padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
        tok_2 = tokenizer_2(args.prompt, padding="max_length", truncation=True,
                            max_length=tokenizer_2.model_max_length, return_tensors="pt").to(device)
        enc_out_1 = text_encoder(tok.input_ids, output_hidden_states=True)
        hidden_1 = enc_out_1.hidden_states[-2]                       # [1, 77, 768]
        enc_out_2 = text_encoder_2(tok_2.input_ids, output_hidden_states=True)
        hidden_2 = enc_out_2.hidden_states[-2]                       # [1, 77, 1280]
        pooled_prompt_embeds = enc_out_2[0]                          # [1, 1280]
        prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1)      # [1, 77, 2048]

    # SDXL micro-conditioning: honest values for this data -- pairs/train crops
    # are already exactly (resolution, resolution) (extract_pairs.py resizes the
    # Hamamatsu side to crop x crop at extraction time), so original_size ==
    # target_size == (resolution, resolution) and there is no sub-crop offset to
    # report (crop_coords_top_left = (0, 0)). Do not spoof this to 1024 (P3-03
    # ticket's resolution decision) -- see P3-03b for the native-1024 fallback.
    add_time_ids = torch.tensor(
        [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
        device=device, dtype=torch.float32,
    )

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
                # VAE encode stays in fp32, outside autocast -- see the dtype note above.
                latents = vae.encode(pixel_values).latent_dist.sample() * scaling
                latents = latents.to(amp_dtype if amp_dtype is not None else torch.float32)
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
            pooled = pooled_prompt_embeds.expand(bsz, -1)
            time_ids = add_time_ids.expand(bsz, -1)

            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=(amp_dtype is not None and device.type == "cuda")):
                model_pred = unet(
                    noisy, timesteps, encoder_hidden_states=enc,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
                ).sample
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
                StableDiffusionXLPipeline.save_lora_weights(
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
