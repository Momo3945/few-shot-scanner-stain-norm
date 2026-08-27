#!/usr/bin/env python3
"""
train_colour_lora_sd35.py -- SD 3.5 A2H colour-LoRA training (PR-01, SD3.5 probe).

Same objective as train_colour_lora.py: a target-domain denoising LoRA, trained
on the TARGET scanner's crops so it learns that scanner's colour statistics;
the directional mapping is applied at inference time via img2img. Only PR-01's
scope: measure wall-clock training time and peak VRAM on the same <=50 A03/H03
crop pairs and the same hyperparameters (rank 8, 1000 steps, batch 1, lr 1e-4)
as the SD1.5 A2 baseline, for an apples-to-apples comparison. This is a
feasibility probe, not a new ablation arm -- see tickets/PROBE-SD35-TICKETS.md.

Differences from train_colour_lora.py / train_colour_lora_sdxl.py, all required
by SD 3.5's MMDiT architecture (fundamentally different from the UNet family):
  - Three text encoders (CLIP-L, CLIP-G, T5-XXL) instead of one/two. Since this
    project always trains on ONE fixed prompt (no per-sample captions, no
    text-encoder training), the combined embedding is computed ONCE and the
    text encoders are then deleted and CUDA-cache-cleared before the training
    loop starts -- T5-XXL alone is ~4.7B params (~9.4GB fp16), worth freeing
    given the 8.06B transformer must also fit on the same GPU.
  - FlowMatchEulerDiscreteScheduler, not DDPMScheduler: SD3.5 is trained with a
    flow-matching objective, not epsilon-prediction. Sigma sampling and loss
    weighting reuse diffusers.training_utils.compute_density_for_timestep_sampling
    / compute_loss_weighting_for_sd3 (verified present in the installed
    diffusers 0.39.0) -- the same helpers HuggingFace's own
    examples/dreambooth/train_dreambooth_lora_sd3.py uses, not hand-rolled.
  - VAE has a shift_factor (0.0609) SD1.5/SDXL's VAE configs don't:
    latents = (encode(x).sample() - shift_factor) * scaling_factor. The
    SDXL-style formula (no shift) would be silently wrong here. VAE forced to
    fp32 (its config's force_upcast=true corroborates this is the safe choice).
  - LoRA target_modules stay ["to_k","to_q","to_v","to_out.0"] -- image-stream
    attention only, matching SD1.5/SDXL's existing convention AND confirmed as
    HuggingFace's own official SD3 LoRA choice; no expansion to the
    text-stream add_q_proj/add_k_proj/add_v_proj/to_add_out projections.
  - transformer.enable_gradient_checkpointing() -- required at 8B params.
  - Checkpoint save via StableDiffusion3Pipeline.save_lora_weights(...,
    transformer_lora_layers=...) -- kwarg name differs from the UNet scripts'
    unet_lora_layers.
  - New: explicit peak-VRAM tracking (torch.cuda.max_memory_allocated()),
    logged into training_config.json alongside wall-clock time -- PR-01
    explicitly requires both numbers; no existing script tracks VRAM.

Everything else (CropDataset, list_target_images, CLI shape, --smoke, save/log
cadence, offline-by-default HF env vars) is an unchanged port.

Usage
-----
    # 5-step smoke test FIRST (validates env + loop cheaply):
    python train_colour_lora_sd35.py --pairs-dir <pairs>/train --direction A2H \
        --output-dir <out>/smoke --smoke

    # real run (rank 8, 1000 steps -- parity with the SD1.5 A2 baseline):
    python train_colour_lora_sd35.py --pairs-dir <pairs>/train --direction A2H \
        --rank 8 --train-steps 1000 --output-dir <out>/a2h_r8_sd35

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
    ap = argparse.ArgumentParser(description="Train a scanner-colour LoRA on SD 3.5 (PR-01 probe).")
    ap.add_argument("--pairs-dir", required=True,
                    help="Directory with *_aperio.png / *_hamamatsu.png crop pairs (the train/ folder).")
    ap.add_argument("--direction", choices=["A2H", "H2A"], required=True)
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-large",
                    help="HF repo id (resolved from HF_HOME cache; runs offline).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank (parity with SD1.5 A2 baseline).")
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 -> defaults to rank.")
    ap.add_argument("--resolution", type=int, default=512,
                    help="pairs/train crops are already exactly this size.")
    ap.add_argument("--train-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--max-t5-length", type=int, default=256,
                    help="T5 tokenizer max sequence length (SD3 pipeline default).")
    ap.add_argument("--logit-mean", type=float, default=0.0)
    ap.add_argument("--logit-std", type=float, default=1.0)
    ap.add_argument("--weighting-scheme", default="logit_normal",
                    choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"])
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

    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        SD3Transformer2DModel,
        StableDiffusion3Pipeline,
    )
    from diffusers.training_utils import (
        compute_density_for_timestep_sampling,
        compute_loss_weighting_for_sd3,
    )
    from diffusers.utils import convert_state_dict_to_diffusers
    from transformers import (
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        T5EncoderModel,
        T5TokenizerFast,
    )
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    # -- reproducibility --
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device found -- training on CPU will be unusably slow. "
              "This script is meant for a GPU node.")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

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

    # amp_dtype also used as the STORAGE dtype for the frozen transformer
    # (below) -- unlike SD1.5/SDXL's UNet, SD3.5's 8.06B-param transformer
    # cannot be kept resident in fp32 (~32GB) on any single GPU in this
    # cluster; HuggingFace's own official SD3 LoRA training script stores the
    # frozen transformer directly in the mixed-precision dtype for this exact
    # reason at this model size. VAE stays fp32 regardless (see below).
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": None}[args.mixed_precision]
    transformer_dtype = amp_dtype if amp_dtype is not None else torch.float32

    # ------------------------------------------------------------------
    # Text encoders + tokenizers -- loaded only long enough to encode the
    # ONE fixed prompt this project always trains on, then freed (see
    # docstring: T5-XXL alone is ~4.7B params, worth not keeping resident).
    # ------------------------------------------------------------------
    # local_files_only=True is required, not just HF_HUB_OFFLINE=1: diffusers'
    # sharded-checkpoint loading path (_get_checkpoint_shard_files, hit by the
    # transformer below) calls the HF API for shard metadata regardless of the
    # offline env vars and raises OfflineModeIsEnabled instead of silently
    # using the cache unless local_files_only is passed explicitly. Applied to
    # every from_pretrained call here for consistency, even where not strictly
    # required (single-file components didn't hit this bug in testing).
    local_only = not args.online

    print(f"Loading SD 3.5 text encoders from cache ({args.model}) ...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer", local_files_only=local_only)
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer_2", local_files_only=local_only)
    tokenizer_3 = T5TokenizerFast.from_pretrained(args.model, subfolder="tokenizer_3", local_files_only=local_only)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        args.model, subfolder="text_encoder", torch_dtype=torch.float16, local_files_only=local_only)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.model, subfolder="text_encoder_2", torch_dtype=torch.float16, local_files_only=local_only)
    text_encoder_3 = T5EncoderModel.from_pretrained(
        args.model, subfolder="text_encoder_3", torch_dtype=torch.float16, local_files_only=local_only)
    text_encoder.to(device); text_encoder_2.to(device); text_encoder_3.to(device)
    text_encoder.eval(); text_encoder_2.eval(); text_encoder_3.eval()

    with torch.no_grad():
        # CLIP-L / CLIP-G: penultimate hidden states concatenated for the
        # sequence embedding, pooled projections concatenated separately --
        # mirrors HuggingFace's own encode_prompt() in train_dreambooth_lora_sd3.py.
        tok = tokenizer(args.prompt, padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
        tok_2 = tokenizer_2(args.prompt, padding="max_length", truncation=True,
                            max_length=tokenizer_2.model_max_length, return_tensors="pt").to(device)
        out_1 = text_encoder(tok.input_ids, output_hidden_states=True)
        out_2 = text_encoder_2(tok_2.input_ids, output_hidden_states=True)
        clip_hidden_1 = out_1.hidden_states[-2]          # [1, 77, 768]
        clip_hidden_2 = out_2.hidden_states[-2]          # [1, 77, 1280]
        pooled_1 = out_1[0]                              # [1, 768]
        pooled_2 = out_2[0]                              # [1, 1280]
        clip_prompt_embeds = torch.cat([clip_hidden_1, clip_hidden_2], dim=-1)   # [1, 77, 2048]
        pooled_prompt_embeds = torch.cat([pooled_1, pooled_2], dim=-1)           # [1, 2048]

        # T5-XXL: hidden states only (no pooled output), fixed max length.
        tok_3 = tokenizer_3(args.prompt, padding="max_length", truncation=True,
                            max_length=args.max_t5_length, return_tensors="pt").to(device)
        t5_prompt_embed = text_encoder_3(tok_3.input_ids)[0]   # [1, max_t5_length, 4096]

        # Zero-pad CLIP's concatenated width up to T5's, then concat along
        # the sequence dim -- same recipe as HuggingFace's encode_prompt().
        clip_prompt_embeds = F.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]))
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)  # [1, 77+256, 4096]
        prompt_embeds = prompt_embeds.detach().clone()
        pooled_prompt_embeds = pooled_prompt_embeds.detach().clone()

    del text_encoder, text_encoder_2, text_encoder_3, tokenizer, tokenizer_2, tokenizer_3
    del tok, tok_2, tok_3, out_1, out_2, clip_hidden_1, clip_hidden_2, pooled_1, pooled_2, t5_prompt_embed, clip_prompt_embeds
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Fixed prompt encoded; text encoders freed from GPU memory.")

    # ------------------------------------------------------------------
    # VAE + transformer + scheduler (the only things resident for training)
    # ------------------------------------------------------------------
    print(f"Loading SD 3.5 transformer/VAE from cache ({args.model}) ...")
    # VAE forced to fp32 -- its config's force_upcast=true corroborates this
    # is the numerically-safe choice (same precedent as the SDXL script).
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32, local_files_only=local_only)
    transformer = SD3Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=transformer_dtype, local_files_only=local_only)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler", local_files_only=local_only)

    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(lora_config)
    # 8.06B params -- gradient checkpointing required to fit LoRA fine-tuning
    # (batch 1, bf16 autocast, fp32 VAE) on a single GPU.
    transformer.enable_gradient_checkpointing()

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"LoRA rank {args.rank} (alpha {args.lora_alpha}) -> {n_train:,} trainable params "
          f"({n_train / sum(p.numel() for p in transformer.parameters()) * 100:.4f}% of transformer).")

    vae.to(device); transformer.to(device)
    vae.eval(); transformer.train()

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    # amp_dtype/transformer_dtype computed earlier (used as the transformer's storage dtype too).
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    scaling = vae.config.scaling_factor
    shift = vae.config.shift_factor   # SD3-specific -- SD1.5/SDXL's VAE configs have no shift term.

    def get_sigmas(timesteps, n_dim, dtype):
        sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ------------------------------------------------------------------
    # Training loop (flow matching -- NOT DDPM epsilon-prediction)
    # ------------------------------------------------------------------
    log_path = out_dir / "loss_log.csv"
    log_fh = open(log_path, "w", newline="")
    log_w = csv.writer(log_fh); log_w.writerow(["step", "loss", "sec", "sigma_mean"])

    print(f"Training for {args.train_steps} steps (batch {args.batch_size}, lr {args.lr}, "
          f"{args.mixed_precision}, weighting_scheme={args.weighting_scheme}) ...")
    step = 0
    t0 = time.time()
    running = 0.0
    running_sigma = 0.0
    done = False
    while not done:
        for pixel_values in loader:
            pixel_values = pixel_values.to(device, dtype=torch.float32)

            with torch.no_grad():
                latents = (vae.encode(pixel_values).latent_dist.sample() - shift) * scaling
                latents = latents.to(amp_dtype if amp_dtype is not None else torch.float32)
            bsz = latents.shape[0]
            noise = torch.randn_like(latents)

            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme, batch_size=bsz,
                logit_mean=args.logit_mean, logit_std=args.logit_std, mode_scale=None,
                device="cpu",
            )
            indices = (u * noise_scheduler.config.num_train_timesteps).long()
            timesteps = noise_scheduler.timesteps[indices].to(device=device)
            sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)

            noisy = (1.0 - sigmas) * latents + sigmas * noise
            target = noise - latents   # flow-matching velocity target, not epsilon.

            enc = prompt_embeds.to(amp_dtype if amp_dtype is not None else torch.float32).expand(bsz, -1, -1)
            pooled = pooled_prompt_embeds.to(amp_dtype if amp_dtype is not None else torch.float32).expand(bsz, -1)

            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=(amp_dtype is not None and device.type == "cuda")):
                model_pred = transformer(
                    hidden_states=noisy, timestep=timesteps,
                    encoder_hidden_states=enc, pooled_projections=pooled,
                    return_dict=False,
                )[0]
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(bsz, -1),
                    1,
                ).mean()

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()

            step += 1
            running += loss.item()
            running_sigma += sigmas.mean().item()
            if step % args.log_every == 0:
                avg = running / args.log_every
                avg_sigma = running_sigma / args.log_every
                running = 0.0
                running_sigma = 0.0
                sec = time.time() - t0
                print(f"  step {step:5d}/{args.train_steps}  loss {avg:.4f}  sigma {avg_sigma:.4f}  ({sec:.1f}s)")
                log_w.writerow([step, f"{avg:.6f}", f"{sec:.1f}", f"{avg_sigma:.4f}"]); log_fh.flush()

            if step % args.save_every == 0 or step >= args.train_steps:
                ckpt = out_dir / (f"checkpoint-{step}" if step < args.train_steps else "final")
                ckpt.mkdir(parents=True, exist_ok=True)
                lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(transformer))
                StableDiffusion3Pipeline.save_lora_weights(
                    save_directory=str(ckpt), transformer_lora_layers=lora_state,
                    safe_serialization=True)
                print(f"  saved LoRA -> {ckpt}")

            if step >= args.train_steps:
                done = True
                break

    log_fh.close()
    total_sec = time.time() - t0
    peak_vram_gb = (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else None
    if peak_vram_gb is not None:
        print(f"Peak VRAM allocated: {peak_vram_gb:.2f} GB")
    with open(out_dir / "training_config.json", "w") as fh:
        json.dump(vars(args) | {"n_target_crops": len(target_paths),
                                "trainable_params": n_train,
                                "total_train_seconds": total_sec,
                                "peak_vram_gb": peak_vram_gb},
                  fh, indent=2)
    print(f"\nDone. Final LoRA in {out_dir/'final'}. Config + loss log written. "
          f"Total time: {total_sec:.1f}s.")


if __name__ == "__main__":
    main()
