#!/usr/bin/env python3
"""
train_hist_lora_sdxl.py -- histopathology warm-start LoRA training on a frozen
SDXL base (P3-05: transfer of A5's histopathology-prior LoRA to SDXL).

Same objective as train_hist_lora.py: an unpaired H&E texture/colour prior,
NOT a directional scanner mapping. Trained on the same 3,000-patch pool
(1,000 PanNuke + 1,500 TCGA-BRCA + 500 MITOS x20 crops) with the standard SD
diffusion loss. Rank 32, batch size 1, 3,000 steps -- unchanged from the
SD1.5 version (proposal sec:hist_lora specifies these hyperparameters, no
reason to deviate for the backbone transfer).

Differences from train_hist_lora.py (SD1.5), all required by SDXL's
architecture -- identical to train_colour_lora_sdxl.py's differences from
train_colour_lora.py, since this is the same SD1.5->SDXL port pattern applied
to a different (unpaired) dataset:
  - Two tokenizers/text encoders (tokenizer/tokenizer_2, text_encoder/
    text_encoder_2) instead of one. Their hidden states are concatenated for
    encoder_hidden_states; text_encoder_2's pooled output feeds SDXL's
    required `added_cond_kwargs`.
  - `added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids":
    add_time_ids}` is a hard SDXL UNet input requirement. add_time_ids is
    built from the honest (resolution,resolution)/(0,0)/(resolution,resolution)
    values -- hist_lora_pool patches are already exactly `--resolution`
    square (see hist_lora_pool/composition.json), same honesty rule as the
    colour LoRA's SDXL port.
  - VAE forced to fp32, kept OUT of the autocast region: SDXL's official VAE
    is documented to NaN under fp16.
  - Gradient checkpointing enabled on the UNet: same 24GB-RTX-3090 fit
    requirement as train_colour_lora_sdxl.py. This is a *plain* UNet LoRA
    fine-tune on a flat patch pool (no ControlNet, no paired data) -- exactly
    the training shape already proven to fit.
  - Checkpoint save via StableDiffusionXLPipeline.save_lora_weights.

Per sec:training_order, this adapter must be trained and FROZEN before any
SDXL colour-LoRA-based composition uses it -- this script only produces the
frozen checkpoint; stacking it at inference (P3-05's actual "A5-on-SDXL") is
a separate step, extending infer_colour_lora_sdxl.py with --hist-lora.

Usage
-----
    # 5-step smoke test FIRST (validates env + loop cheaply):
    python train_hist_lora_sdxl.py --manifest <pool>/manifest.csv --patches-dir <pool>/patches \
        --output-dir <out>/hist_sdxl_smoke --smoke

    # real run (rank 32, 3000 steps, per proposal):
    python train_hist_lora_sdxl.py --manifest <pool>/manifest.csv --patches-dir <pool>/patches \
        --output-dir <out>/hist_r32_sdxl

Dependencies: torch, diffusers, transformers, peft, safetensors, Pillow, numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

# Five official MITOS-ATYPIA-14 held-out test slides -- must never appear in
# this pool. Checked against the manifest's mitos-row slide_id column (exact
# match, not substring), same independent check as train_hist_lora.py.
HELD_OUT_SLIDES = {"A06", "A08", "A09", "A13", "A16"}


# ----------------------------------------------------------------------
# Pure helper (import-light so it can be unit-tested without torch) --
# identical to train_hist_lora.py's version, unchanged.
# ----------------------------------------------------------------------
def load_pool_manifest(manifest_path: str, patches_dir: str) -> list[str]:
    """Read the pool manifest and return absolute patch paths, after an independent
    leak check against the five held-out MITOS slides. Aborts loudly on any leak
    rather than silently dropping rows -- a leak here means the upstream pool build
    is broken and should be fixed there, not patched around here.
    """
    patches_root = Path(patches_dir)
    paths = []
    leaks = []
    with open(manifest_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row.get("source") == "mitos" and row.get("slide_id_or_tissue_type") in HELD_OUT_SLIDES:
                leaks.append(row["filename"])
                continue
            paths.append(str(patches_root / row["filename"]))
    if leaks:
        raise SystemExit(
            f"REFUSING TO TRAIN: {len(leaks)} held-out MITOS patch(es) found in "
            f"{manifest_path} (e.g. {leaks[:5]}). Held-out slides {sorted(HELD_OUT_SLIDES)} "
            f"must never be used in training -- fix the pool, do not bypass this check."
        )
    return paths


def parse_args():
    ap = argparse.ArgumentParser(description="Train the A5 histopathology warm-start LoRA on SDXL.")
    ap.add_argument("--manifest", required=True,
                    help="hist_lora_pool manifest.csv (filename,source,original_path,"
                         "slide_id_or_tissue_type,width,height).")
    ap.add_argument("--patches-dir", required=True,
                    help="Directory containing the manifest's patch images (the patches/ folder).")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0",
                    help="HF repo id (resolved from HF_HOME cache; runs offline).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank", type=int, default=32, help="LoRA rank (proposal sec:hist_lora: 32).")
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 -> defaults to rank.")
    ap.add_argument("--resolution", type=int, default=512,
                    help="hist_lora_pool patches are already exactly this size -- honest "
                         "SDXL micro-conditioning value (same rule as the colour LoRA's port).")
    ap.add_argument("--train-steps", type=int, default=3000, help="proposal sec:hist_lora: 3000.")
    ap.add_argument("--batch-size", type=int, default=1, help="proposal sec:hist_lora: 1.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--save-every", type=int, default=750,
                    help="Default gives 4 checkpoints over 3000 steps (matches the SD1.5 "
                         "hist LoRA's cadence).")
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
    # Data (unchanged from train_hist_lora.py)
    # ------------------------------------------------------------------
    patch_paths = load_pool_manifest(args.manifest, args.patches_dir)
    if not patch_paths:
        raise SystemExit(f"No patches loaded from {args.manifest}.")
    print(f"Loaded {len(patch_paths)} H&E patches from {args.manifest} "
          f"(unpaired warm-start pool, held-out leak check passed).")

    class PatchDataset(Dataset):
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

    loader = DataLoader(PatchDataset(patch_paths, args.resolution),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)

    # ------------------------------------------------------------------
    # Model components (all frozen except the LoRA we inject) --
    # SDXL machinery ported from train_colour_lora_sdxl.py.
    # ------------------------------------------------------------------
    print(f"Loading SDXL components from cache ({args.model}) ...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.model, subfolder="text_encoder_2")
    # VAE forced to fp32 -- SDXL's official VAE NaNs under fp16. Kept out of the
    # autocast region entirely. Do not change this dtype back to match the
    # UNet/text-encoder precision.
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
    # (batch 1, bf16 autocast, fp32 VAE) inside a 24GB RTX 3090. Same requirement
    # as train_colour_lora_sdxl.py; this is a plain UNet LoRA fine-tune (no
    # ControlNet), so no new memory question beyond what that script already proved.
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

    # SDXL micro-conditioning: honest values for this data -- hist_lora_pool
    # patches are already exactly (resolution, resolution) square, so
    # original_size == target_size == (resolution, resolution) and there is
    # no sub-crop offset to report (crop_coords_top_left = (0, 0)).
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
        json.dump(vars(args) | {"n_patches": len(patch_paths),
                                "trainable_params": n_train,
                                "prediction_type": noise_scheduler.config.prediction_type},
                  fh, indent=2)
    print(f"\nDone. Final LoRA in {out_dir/'final'}. Config + loss log written. "
          f"Freeze this checkpoint before stacking it at inference for P3-05.")


if __name__ == "__main__":
    main()
