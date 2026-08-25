#!/usr/bin/env python3
"""
train_colour_translation_lora_sdxl.py -- P3-06: transfer P1-10's
source-conditioned colour LoRA + fresh 6-channel ControlNet to SDXL.

This is NOT a transfer of A4/A5 (already done by P3-03/P3-05) -- P1-10 has
since become the best-performing SD1.5 result in this project (see
tickets/PHASE1-TICKETS.md P1-10/P1-11), so this ticket (P3-06, see
tickets/PHASE3-TICKETS.md) transfers THAT architecture instead. Scope is
P1-10 only (plain 50-step DDIM, A2H direction) -- P1-11's DDIM-inversion-on-
SDXL is a separate, later ticket (P3-07, not started here).

Merges two already-validated scripts in this project, with one genuinely new
wiring point (not copy-paste from either parent):

  train_colour_lora_sdxl.py (P3-03) supplies the SDXL adaptation: dual
  tokenizers/text encoders concatenated for encoder_hidden_states,
  text_encoder_2's pooled output + add_time_ids for added_cond_kwargs, VAE
  forced fp32 and kept outside the autocast region (SDXL's official VAE NaNs
  under fp16 -- never "fix" this back), unet.enable_gradient_checkpointing()
  (SDXL's UNet is ~3x SD1.5's), StableDiffusionXLPipeline.save_lora_weights.

  train_colour_translation_lora.py (P1-10, SD1.5) supplies the
  source-conditioning recipe: ControlNetModel.from_unet(unet,
  conditioning_channels=6) (fresh, zero-initialised conditioning encoder +
  zero-conv output layers -- only these + the LoRA are trainable, the cloned
  backbone stays frozen), the 6-channel source-RGB+Canny conditioning tensor,
  conditioning-only spatial jitter, leak-checked build_pairs() (A03/H03
  only), leave-one-frame-out val split with best-val-loss checkpoint
  selection.

  NEW wiring (neither parent script needs this): SDXL's ControlNetModel and
  UNet2DConditionModel BOTH require added_cond_kwargs={"text_embeds":
  pooled, "time_ids": time_ids} in their forward() calls when conditioning
  is present -- P1-10's SD1.5 script passes neither (SD1.5 has no micro-
  conditioning), and P3-03's SDXL script only ever calls unet() (no
  ControlNet in that script), so this is the one place this script cannot
  just copy from either parent. Confirm via the mandatory --smoke run before
  trusting anything past import/argument-parse time.

Never touches train_colour_translation_lora.py, train_colour_lora_sdxl.py,
lora/a2h_cond_r8/, or lora/a2h_r8_sdxl/ -- new script, new checkpoint tag
(lora/a2h_cond_r8_sdxl/), per this project's isolation convention.

Direction: --direction is still a flag for symmetry with the SD1.5 script,
but only A2H is in scope for this ticket (matches P3-03/P3-05's own descope
of H2A on SDXL) -- H2A is accepted but not validated/tested here.

Leak check, spatial-jitter tolerance, and validation-split rationale are
IDENTICAL to train_colour_translation_lora.py -- see that script's docstring
for the full explanation; not repeated here.

Usage
-----
    # mandatory smoke test FIRST (env/loop check only):
    python train_colour_translation_lora_sdxl.py --pairs-dir <pairs>/train \
        --direction A2H --output-dir <out>/smoke_sdxl --smoke

    # mandatory overfit-8 control (must reach near-zero loss, no NaNs):
    python train_colour_translation_lora_sdxl.py --pairs-dir <pairs>/train \
        --direction A2H --output-dir <out>/overfit_test_sdxl --overfit-n 8 --train-steps 300

    # real run:
    python train_colour_translation_lora_sdxl.py --pairs-dir <pairs>/train \
        --direction A2H --rank 8 --train-steps 1000 --output-dir <out>/a2h_cond_r8_sdxl

Dependencies: torch, diffusers, transformers, peft, safetensors, Pillow,
numpy, opencv-python-headless. Requires canny.py on the path (src/eval/,
reached via a relative sys.path insert since this script lives in src/train/).
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

# canny.py lives in src/eval, a sibling of this file's parent (src/train)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

HELD_OUT_SLIDES = {"A06", "A08", "A09", "A13", "A16"}


# ----------------------------------------------------------------------
# Pure helpers -- IDENTICAL to train_colour_translation_lora.py (SD1.5),
# copied verbatim since they have nothing to do with the backbone.
# ----------------------------------------------------------------------
def build_pairs(pairs_dir: str) -> list[dict]:
    """Return every (aperio_path, hamamatsu_path, pair_id, slide, frame_id) pair
    found in pairs_dir, asserting both sides exist and only A03/H03 appear.

    Filenames follow the extract_pairs.py convention:
        <slide>_<frame_id>_c<idx>_aperio.png / ..._hamamatsu.png
    """
    aperio_paths = sorted(glob.glob(os.path.join(pairs_dir, "*_aperio.png")))
    pairs = []
    for ap in aperio_paths:
        stem = ap[: -len("_aperio.png")]
        hp = stem + "_hamamatsu.png"
        if not os.path.exists(hp):
            raise SystemExit(f"Missing Hamamatsu side for pair {stem} (expected {hp}).")
        pair_id = os.path.basename(stem)
        slide = pair_id.split("_")[0]
        frame_id = pair_id.split("_")[1]
        if slide in HELD_OUT_SLIDES:
            raise SystemExit(
                f"LEAK CHECK FAILED: pair {pair_id} belongs to held-out slide {slide} "
                f"({HELD_OUT_SLIDES}). --pairs-dir must contain ONLY A03/H03 training pairs.")
        pairs.append({"pair_id": pair_id, "slide": slide, "frame_id": frame_id,
                      "aperio_path": ap, "hamamatsu_path": hp})
    if not pairs:
        raise SystemExit(f"No *_aperio.png/*_hamamatsu.png pairs found in {pairs_dir}.")
    bad_slides = {p["slide"] for p in pairs} - {"A03"}
    if bad_slides:
        raise SystemExit(f"LEAK CHECK FAILED: unexpected slide(s) {bad_slides} in {pairs_dir} "
                         f"(P1-10/P3-06 scope is A03/H03 only).")
    return pairs


def pairs_hash(pairs: list[dict]) -> str:
    """Hash pair IDs AND each file's (size, mtime) -- see train_colour_
    translation_lora.py's identical helper for the full rationale.
    """
    parts = []
    for p in sorted(pairs, key=lambda p: p["pair_id"]):
        a_stat = os.stat(p["aperio_path"])
        h_stat = os.stat(p["hamamatsu_path"])
        parts.append(f"{p['pair_id']}:{a_stat.st_size}:{a_stat.st_mtime_ns}:"
                     f"{h_stat.st_size}:{h_stat.st_mtime_ns}")
    return hashlib.sha256(",".join(parts).encode()).hexdigest()[:16]


def parse_args():
    ap = argparse.ArgumentParser(
        description="P3-06: train P1-10's colour LoRA + source-conditioning ControlNet on SDXL.")
    ap.add_argument("--pairs-dir", required=True,
                    help="Directory with *_aperio.png / *_hamamatsu.png pairs (A03/H03 only).")
    ap.add_argument("--direction", choices=["A2H", "H2A"], required=True,
                    help="A2H: source=Aperio, target=Hamamatsu. In scope for P3-06. "
                         "H2A accepted but not validated here (matches P3-03/P3-05's descope).")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0",
                    help="HF repo id (resolved from HF_HOME cache; runs offline).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank", type=int, default=8, help="Colour LoRA rank.")
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 -> defaults to rank.")
    ap.add_argument("--resolution", type=int, default=512,
                    help="pairs/train crops are already exactly this size -- honest SDXL "
                         "micro-conditioning value, not spoofed to 1024 (matches P3-03).")
    ap.add_argument("--jitter", type=int, default=8,
                    help="Max px random crop-offset jitter applied to the conditioning "
                         "(source) image only -- see train_colour_translation_lora.py's "
                         "docstring for the full rationale. 0 disables it.")
    ap.add_argument("--train-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--mixed-precision", choices=["bf16", "fp16", "no"], default="bf16")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="Fraction of FRAME IDs (not crops) held out for validation, from "
                         "within the training-domain pairs only (never A06/08/09/13/16).")
    ap.add_argument("--online", action="store_true",
                    help="Allow HF network access (default: offline, use local cache).")
    ap.add_argument("--smoke", action="store_true",
                    help="5-step dry run: overrides train-steps/save-every for a quick env+loop "
                         "check. NOT the mandatory overfit control -- use --overfit-n for that.")
    ap.add_argument("--overfit-n", type=int, default=0,
                    help="Mandatory first diagnostic control (same as P1-10): restrict to the "
                         "first N pairs (e.g. 8) and train longer. Loss must approach near-zero "
                         "with no NaNs before anything else here is trusted. 0 = use all pairs.")
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
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    from diffusers import (AutoencoderKL, ControlNetModel, DDPMScheduler,
                           StableDiffusionXLPipeline, UNet2DConditionModel)
    from diffusers.utils import convert_state_dict_to_diffusers
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    from canny import extract_canny_control_image

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("No CUDA device -- this script is meant for a GPU node.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Paired dataset + leak check + train/val split by FRAME ID
    # (identical to train_colour_translation_lora.py -- backbone-agnostic)
    # ------------------------------------------------------------------
    all_pairs = build_pairs(args.pairs_dir)
    if args.overfit_n:
        all_pairs = all_pairs[: args.overfit_n]
        print(f"OVERFIT-TEST MODE: restricted to {len(all_pairs)} pairs.")

    frame_ids = sorted({p["frame_id"] for p in all_pairs})
    rng_split = random.Random(args.seed)
    rng_split.shuffle(frame_ids)
    n_val_frames = max(1, round(len(frame_ids) * args.val_frac)) if len(frame_ids) > 1 else 0
    val_frames = set(frame_ids[:n_val_frames]) if not args.overfit_n else set()
    train_pairs = [p for p in all_pairs if p["frame_id"] not in val_frames]
    val_pairs = [p for p in all_pairs if p["frame_id"] in val_frames]

    print(f"Direction {args.direction}: {len(all_pairs)} pairs, "
          f"{len(frame_ids)} frames -> {len(val_frames)} held out for val "
          f"({len(train_pairs)} train pairs / {len(val_pairs)} val pairs).")

    h = pairs_hash(all_pairs)
    with open(out_dir / "pair_manifest.json", "w") as fh:
        json.dump({"pairs_hash": h, "n_pairs": len(all_pairs),
                   "pairs": [p["pair_id"] for p in all_pairs],
                   "val_frames": sorted(val_frames)}, fh, indent=2)
    print(f"Pair manifest saved (hash {h}) -- {len(all_pairs)} pairs, "
          f"slides={sorted({p['slide'] for p in all_pairs})}.")

    src_key, tgt_key = ("aperio_path", "hamamatsu_path") if args.direction == "A2H" \
        else ("hamamatsu_path", "aperio_path")

    class PairDataset(Dataset):
        def __init__(self, pairs, resolution, jitter):
            self.pairs = pairs
            self.res = resolution
            self.jitter = jitter

        def __len__(self):
            return len(self.pairs)

        def _load(self, path):
            img = Image.open(path).convert("RGB")
            if img.size != (self.res, self.res):
                img = img.resize((self.res, self.res), Image.LANCZOS)
            return np.asarray(img)

        def __getitem__(self, i):
            p = self.pairs[i]
            src_rgb = self._load(p[src_key])
            tgt_rgb = self._load(p[tgt_key])

            if self.jitter > 0:
                dx = random.randint(-self.jitter, self.jitter)
                dy = random.randint(-self.jitter, self.jitter)
                padded = np.pad(src_rgb, ((self.jitter, self.jitter), (self.jitter, self.jitter), (0, 0)),
                                mode="reflect")
                y0, x0 = self.jitter + dy, self.jitter + dx
                src_rgb = padded[y0:y0 + self.res, x0:x0 + self.res]

            canny = extract_canny_control_image(src_rgb)  # HxWx3 uint8, 0-255
            rgb_t = torch.from_numpy(src_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            canny_t = torch.from_numpy(canny.astype(np.float32) / 255.0).permute(2, 0, 1)
            control = torch.cat([rgb_t, canny_t], dim=0)  # 6xHxW

            tgt_arr = np.asarray(tgt_rgb, dtype=np.float32) / 127.5 - 1.0
            target = torch.from_numpy(tgt_arr).permute(2, 0, 1)
            return target, control

    train_loader = DataLoader(PairDataset(train_pairs, args.resolution, args.jitter),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=(len(train_pairs) >= args.batch_size), pin_memory=True)
    val_loader = (DataLoader(PairDataset(val_pairs, args.resolution, jitter=0),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
                 if val_pairs else None)

    # ------------------------------------------------------------------
    # Model: SDXL's dual text encoders (P3-03) + frozen UNet -> fresh
    # 6-channel ControlNet cloned from it (P1-10) -> THEN the LoRA.
    # ------------------------------------------------------------------
    print(f"Loading SDXL components from cache ({args.model}) ...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.model, subfolder="text_encoder_2")
    # VAE forced fp32 -- SDXL's official VAE NaNs under fp16. Never "fix" this
    # back to fp16 (see train_colour_lora_sdxl.py's identical note).
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    print("Building fresh source-conditioning ControlNet via ControlNetModel.from_unet() "
          "(conditioning_channels=6: source RGB + Canny, zero-initialised conditioning "
          "layers, weights cloned from the pristine frozen SDXL UNet)...")
    controlnet = ControlNetModel.from_unet(unet, conditioning_channels=6)
    controlnet.train()
    # Freeze the backbone from_unet() CLONED from the UNet -- only the genuinely
    # NEW, zero-initialised parts get trained (identical rationale to P1-10's
    # SD1.5 script; see that script's docstring for the full explanation).
    controlnet.requires_grad_(False)
    TRAINABLE_CONTROLNET_PREFIXES = (
        "controlnet_cond_embedding",
        "controlnet_down_blocks",
        "controlnet_mid_block",
    )
    n_frozen = n_trained = 0
    for name, param in controlnet.named_parameters():
        if any(name.startswith(p) for p in TRAINABLE_CONTROLNET_PREFIXES):
            param.requires_grad_(True)
            n_trained += param.numel()
        else:
            n_frozen += param.numel()
    if n_trained == 0:
        raise SystemExit(
            "ControlNet freeze produced 0 trainable params -- TRAINABLE_CONTROLNET_PREFIXES "
            "doesn't match this diffusers version's ControlNetModel attribute names. "
            "Check controlnet.named_parameters() directly before proceeding.")
    n_controlnet = n_trained
    print(f"ControlNet: {n_trained:,} trainable (conditioning encoder + zero-conv "
          f"output layers), {n_frozen:,} frozen (cloned SDXL UNet backbone).")

    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    n_lora = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Trainable params -- LoRA (rank {args.rank}): {n_lora:,}  "
          f"ControlNet (fresh, source-conditioning branch): {n_controlnet:,}  "
          f"Total: {n_lora + n_controlnet:,}")

    # SDXL's UNet is ~3x SD1.5's, and this script also runs a full ControlNet
    # forward/backward every step (P3-03's plain colour-LoRA script has no
    # ControlNet at all) -- gradient checkpointing on the UNet keeps this
    # inside a 24GB RTX 3090.
    unet.enable_gradient_checkpointing()

    vae.to(device)
    text_encoder.to(device); text_encoder_2.to(device)
    unet.to(device); controlnet.to(device)
    vae.eval(); text_encoder.eval(); text_encoder_2.eval()
    unet.train(); controlnet.train()

    # Precompute the (fixed) prompt embedding once, via BOTH text encoders --
    # identical convention to train_colour_lora_sdxl.py.
    with torch.no_grad():
        tok = tokenizer(args.prompt, padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
        tok_2 = tokenizer_2(args.prompt, padding="max_length", truncation=True,
                            max_length=tokenizer_2.model_max_length, return_tensors="pt").to(device)
        enc_out_1 = text_encoder(tok.input_ids, output_hidden_states=True)
        hidden_1 = enc_out_1.hidden_states[-2]
        enc_out_2 = text_encoder_2(tok_2.input_ids, output_hidden_states=True)
        hidden_2 = enc_out_2.hidden_states[-2]
        pooled_prompt_embeds = enc_out_2[0]
        prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1)

    # SDXL micro-conditioning: honest values for this data (matches
    # train_colour_lora_sdxl.py's identical resolution decision -- P3-03).
    add_time_ids = torch.tensor(
        [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
        device=device, dtype=torch.float32,
    )

    trainable = [p for p in unet.parameters() if p.requires_grad] + \
        [p for p in controlnet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": None}[args.mixed_precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    scaling = vae.config.scaling_factor

    def run_step(target_pixels, control_images, generator=None):
        target_pixels = target_pixels.to(device, dtype=torch.float32)
        control_images = control_images.to(device, dtype=torch.float32)

        with torch.no_grad():
            # VAE encode stays fp32, outside autocast (see the dtype note above).
            dist = vae.encode(target_pixels).latent_dist
            latents = (dist.mode() if generator is not None else dist.sample()) * scaling
            latents = latents.to(amp_dtype if amp_dtype is not None else torch.float32)
        bsz = latents.shape[0]
        if generator is not None:
            noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (bsz,), generator=generator, device=device).long()
        else:
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (bsz,), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        target = (noise_scheduler.get_velocity(latents, noise, timesteps)
                 if noise_scheduler.config.prediction_type == "v_prediction" else noise)
        enc = prompt_embeds.expand(bsz, -1, -1)
        pooled = pooled_prompt_embeds.expand(bsz, -1)
        time_ids = add_time_ids.expand(bsz, -1)
        added_cond_kwargs = {"text_embeds": pooled, "time_ids": time_ids}

        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=(amp_dtype is not None and device.type == "cuda")):
            # NEW wiring vs both parent scripts: SDXL's ControlNetModel needs
            # added_cond_kwargs too, not just the UNet (see module docstring).
            down_res, mid_res = controlnet(
                noisy, timesteps, encoder_hidden_states=enc,
                controlnet_cond=control_images, added_cond_kwargs=added_cond_kwargs,
                return_dict=False)
            model_pred = unet(
                noisy, timesteps, encoder_hidden_states=enc,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res).sample
            loss = F.mse_loss(model_pred.float(), target.float())
        return loss

    @torch.no_grad()
    def evaluate():
        if val_loader is None:
            return None
        unet.eval(); controlnet.eval()
        gen = torch.Generator(device=device).manual_seed(args.seed)
        losses = []
        for target_pixels, control_images in val_loader:
            loss = run_step(target_pixels, control_images, generator=gen)
            losses.append(loss.item())
        unet.train(); controlnet.train()
        return sum(losses) / len(losses) if losses else None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log_path = out_dir / "loss_log.csv"
    log_fh = open(log_path, "w", newline="")
    log_w = csv.writer(log_fh); log_w.writerow(["step", "loss", "val_loss", "sec"])

    def save_checkpoint(ckpt: Path):
        ckpt.mkdir(parents=True, exist_ok=True)
        lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=str(ckpt), unet_lora_layers=lora_state, safe_serialization=True)
        controlnet.save_pretrained(str(ckpt / "controlnet"))

    print(f"Training for {args.train_steps} steps (batch {args.batch_size}, lr {args.lr}, "
          f"{args.mixed_precision}) ...")
    step = 0
    t0 = time.time()
    running = 0.0
    best_val_loss = float("inf")
    done = False
    while not done:
        for target_pixels, control_images in train_loader:
            loss = run_step(target_pixels, control_images)

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
                log_w.writerow([step, f"{avg:.6f}", "", f"{sec:.1f}"]); log_fh.flush()

            if step % args.save_every == 0 or step >= args.train_steps:
                val_loss = evaluate()
                if val_loss is not None:
                    print(f"    val_loss {val_loss:.4f}")
                    log_w.writerow([step, "", f"{val_loss:.6f}", f"{time.time()-t0:.1f}"]); log_fh.flush()
                ckpt = out_dir / (f"checkpoint-{step}" if step < args.train_steps else "final")
                save_checkpoint(ckpt)
                print(f"  saved LoRA + ControlNet -> {ckpt}")
                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(out_dir / "best")
                    print(f"    new best (val_loss {val_loss:.4f}) -> {out_dir / 'best'}")

            if step >= args.train_steps:
                done = True
                break

    log_fh.close()
    with open(out_dir / "training_config.json", "w") as fh:
        json.dump(vars(args) | {"n_pairs": len(all_pairs), "pairs_hash": h,
                                "n_train_pairs": len(train_pairs), "n_val_pairs": len(val_pairs),
                                "val_frames": sorted(val_frames),
                                "trainable_params_lora": n_lora,
                                "trainable_params_controlnet": n_controlnet,
                                "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
                                "prediction_type": noise_scheduler.config.prediction_type},
                  fh, indent=2)
    print(f"\nDone. Final LoRA + ControlNet in {out_dir/'final'}. Config + loss log written.")
    print("Next (mandatory before trusting this run): source-conditioning ablation via "
          "infer_colour_translation_sdxl.py --source-mode {correct,zero,shuffled}.")


if __name__ == "__main__":
    main()
