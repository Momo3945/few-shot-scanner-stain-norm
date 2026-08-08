#!/usr/bin/env python3
"""
train_hist_lora.py -- histopathology warm-start LoRA training on a frozen SD 1.5 base
(ablation A5, proposal sec:hist_lora / sec:training_order).

Objective: an unpaired H&E texture/colour prior, NOT a directional scanner mapping.
Trained on a pool of 3,000 H&E patches (1,000 PanNuke + 1,500 TCGA-BRCA + 500 MITOS
x20 crops from the 11 training pairs) with the standard SD diffusion loss -- same
formulation as train_colour_lora.py, just over a broader unpaired corpus instead of
target-scanner-only crops. Rank 32, batch size 1, 3,000 steps per the proposal.

Per sec:training_order, this adapter must be trained and FROZEN before colour-LoRA
training begins for the A5 leg -- do NOT train jointly with the colour LoRA. This
script only produces the frozen hist LoRA checkpoint; loading it alongside a colour
LoRA at inference (the actual "stacking") is a separate step (P1-07), not done here.

Data-leakage boundary (see CLAUDE.md): the pool may include the same A03/H03 crops
used for colour-LoRA training (allowed -- this adapter learns an unpaired prior, not
the paired A->H mapping), but must NEVER include the five held-out MITOS test pairs
(A06, A08, A09, A13, A16). The pool this script reads from was already built with
that exclusion (see hist_lora_pool/composition.json's "heldout_excluded"), but this
script re-checks it independently at load time rather than trusting that blindly.

Environment: run on a GPU node with the model cached locally. Set HF_HOME to the
cache and the script runs offline (no network needed on the compute node).

Usage
-----
    # 5-step smoke test FIRST (validates env + loop cheaply):
    python train_hist_lora.py --manifest <pool>/manifest.csv --patches-dir <pool>/patches \
        --output-dir <out>/hist_smoke --smoke

    # real run (rank 32, 3000 steps, per proposal):
    python train_hist_lora.py --manifest <pool>/manifest.csv --patches-dir <pool>/patches \
        --output-dir <out>/hist_r32

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

# Five official MITOS-ATYPIA-14 held-out test slides -- must never appear in this
# pool. Checked against the manifest's mitos-row slide_id column (exact match, not
# substring -- some TCGA barcodes coincidentally contain "A06" etc. as a substring).
HELD_OUT_SLIDES = {"A06", "A08", "A09", "A13", "A16"}


# ----------------------------------------------------------------------
# Pure helper (import-light so it can be unit-tested without torch)
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
    ap = argparse.ArgumentParser(description="Train the A5 histopathology warm-start LoRA on SD 1.5.")
    ap.add_argument("--manifest", required=True,
                    help="hist_lora_pool manifest.csv (filename,source,original_path,"
                         "slide_id_or_tissue_type,width,height).")
    ap.add_argument("--patches-dir", required=True,
                    help="Directory containing the manifest's patch images (the patches/ folder).")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                    help="HF repo id (resolved from HF_HOME cache; runs offline).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank", type=int, default=32, help="LoRA rank (proposal sec:hist_lora: 32).")
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 -> defaults to rank.")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--train-steps", type=int, default=3000, help="proposal sec:hist_lora: 3000.")
    ap.add_argument("--batch-size", type=int, default=1, help="proposal sec:hist_lora: 1.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--save-every", type=int, default=750,
                    help="Default gives 4 checkpoints over 3000 steps (same 25%%-of-run "
                         "cadence as the colour LoRA's save-every=250 over 1000 steps).")
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
        json.dump(vars(args) | {"n_patches": len(patch_paths),
                                "trainable_params": n_train,
                                "prediction_type": noise_scheduler.config.prediction_type},
                  fh, indent=2)
    print(f"\nDone. Final LoRA in {out_dir/'final'}. Config + loss log written. "
          f"Freeze this checkpoint before starting colour-LoRA training for A5 "
          f"(sec:training_order) -- do not train jointly.")


if __name__ == "__main__":
    main()
