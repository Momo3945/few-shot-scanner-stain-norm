#!/usr/bin/env python3
"""
train_colour_translation_lora.py -- P1-10 corrective experiment: genuine
source-conditioned scanner translation. Trains a colour LoRA jointly with a
fresh, trainable ControlNet-style source-conditioning branch, so the source
image actually participates in the training-time noise prediction -- not just
the img2img starting latent at inference (see tickets/PHASE1-TICKETS.md P1-10
for the full diagnosis and required controls).

Root cause this fixes: train_colour_lora.py only ever loads TARGET-domain
crops (list_target_images()) -- the source crop never enters the dataset
item, UNet conditioning, loss, or gradient path. That checkpoint learns an
unconditional target-domain prior P(H), not the proposal's directional
mapping P(H|A); direction is only imposed at inference via img2img's starting
latent, a much weaker mechanism. This script trains a genuine conditional
model instead.

Design (resolves the ticket's two descriptions -- step 3's "encode the source
crop... feed through a conditioning adapter" vs step 4's "start with the
existing Canny pathway" -- as COMPLEMENTARY signals, not alternatives; an
earlier draft of this script implemented only the geometry half (Canny) and
lost all source colour/appearance information, which a review caught before
any GPU time was spent): use diffusers' ControlNetModel, freshly initialised
via ControlNetModel.from_unet(unet, conditioning_channels=6) -- NOT the
pretrained lllyasviel/sd-controlnet-canny used elsewhere in this project for
inference (that one is frozen and was never trained on this task at all).
from_unet() zero-initialises the new conditioning-specific layers
automatically, matching the ControlNet paper's convention and this ticket's
"zero-initialised" requirement with no custom code. Its conditioning input is
a 6-channel tensor: the SOURCE crop's raw RGB (channels 0-2, normalised to
[0,1] -- this is what carries source colour/appearance, satisfying step 3)
concatenated with its Canny edge map (channels 3-5, also [0,1] -- explicit
structure, satisfying step 4, same canny.py pathway used everywhere else in
this project). ControlNetModel's own conditioning encoder maps that into a
feature space aligned with the UNet's latent resolution, and its residuals are
injected into the frozen UNet's down/mid blocks during the training forward
pass -- this reuses StableDiffusionControlNetImg2ImgPipeline's exact
underlying mechanism (already proven correct in this project at inference,
just with conditioning_channels widened from 3 to 6) rather than a bespoke
module.

Train/inference distribution note: training denoises a noisy TARGET latent
conditioned on the source (RGB+Canny); inference applies the trained model via
img2img starting from a noised SOURCE latent, same conditioning. This is the
same convention train_colour_lora.py/infer_colour_lora.py already use
successfully everywhere else in this project (train via target-domain
denoising, apply via partial-noise img2img at inference) -- not a new problem
introduced by this ticket, flagged here only so a future reader doesn't
re-raise it as a surprise.

Never touches train_colour_lora.py, lora/a2h_r8/, or eval/a{0-5}*/ -- new
script, new checkpoint tag (lora/a2h_cond_r8/), new eval tag
(eval/a2h_cond_r8/), per the ticket's isolation requirement.

Leak check: every pair is asserted to come from A03/H03 only; hard-fails if
any held-out slide (A06/A08/A09/A13/A16) appears in --pairs-dir. The exact
pair list and a hash of it are saved to training_config.json.

Spatial-jitter tolerance: pairs are coordinate-corresponding, not pixel-exact
(no training-time registration, per this project's standing methodology
guardrail -- see CLAUDE.md). A small independent random crop-offset jitter is
applied to the conditioning (source) image only, decoupled from the target
crop's own position, so the ControlNet is not implicitly trained to expect
pixel-perfect alignment that doesn't exist in this data. This is the
documented alternative the ticket asks for in place of a hidden registration
step.

Validation split: only one slide pair (A03/H03) is in scope for this
experiment, so "leave-slides-out" (used by train_atypia_classifier.py) does
not apply -- instead this holds out a fraction of FRAME IDs (not individual
crops, to avoid leaking crops from the same frame across the split), per the
ticket's "strictly training-domain leave-one-frame-out split" wording.
Checkpoints are saved at --save-every steps (default lines up with 250/500/
750/1000 for a 1000-step run, per the ticket's mandatory multi-checkpoint
evaluation requirement) with a per-checkpoint val loss so the final step is
never silently assumed best.

Usage
-----
    # mandatory first control -- overfit an 8-pair subset, must reach near-zero
    # loss with no NaNs before anything else here is trusted:
    python train_colour_translation_lora.py --pairs-dir <pairs>/train \
        --direction A2H --output-dir <out>/overfit_test --overfit-n 8 --train-steps 300

    # 5-step smoke test (env/loop check only, NOT the overfit control above):
    python train_colour_translation_lora.py --pairs-dir <pairs>/train \
        --direction A2H --output-dir <out>/smoke --smoke

    # real run:
    python train_colour_translation_lora.py --pairs-dir <pairs>/train \
        --direction A2H --rank 8 --train-steps 1000 --output-dir <out>/a2h_cond_r8

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
# Pure helpers (import-light so they can be unit-tested without torch)
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
                         f"(P1-10 scope is A03/H03 only).")
    return pairs


def pairs_hash(pairs: list[dict]) -> str:
    """Hash pair IDs AND each file's (size, mtime) -- hashing IDs alone would
    give an identical hash if a crop file were silently replaced in place
    (same pair_id, different pixel content). Full content hashing would be
    more airtight but is unnecessary I/O for a <=50-pair dataset; size+mtime
    catches the realistic failure mode (a re-extraction overwriting a file)
    at near-zero cost.
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
        description="P1-10: train a colour LoRA jointly with a source-conditioning ControlNet.")
    ap.add_argument("--pairs-dir", required=True,
                    help="Directory with *_aperio.png / *_hamamatsu.png pairs (A03/H03 only).")
    ap.add_argument("--direction", choices=["A2H", "H2A"], required=True,
                    help="A2H: source=Aperio, target=Hamamatsu (noise-prediction target). "
                         "H2A: source=Hamamatsu, target=Aperio.")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                    help="HF repo id (resolved from HF_HOME cache; runs offline).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank", type=int, default=8, help="Colour LoRA rank.")
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 -> defaults to rank.")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--jitter", type=int, default=8,
                    help="Max px random crop-offset jitter applied to the conditioning "
                         "(source) image only -- alignment-tolerant substitute for "
                         "training-time registration (P1-10 requirement). 0 disables it.")
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
                    help="P1-10's mandatory first diagnostic control: restrict to the first N "
                         "pairs (e.g. 8) and train longer. Loss must approach near-zero with no "
                         "NaNs before anything else in this ticket is trusted. 0 = use all pairs.")
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
                           StableDiffusionPipeline, UNet2DConditionModel)
    from diffusers.utils import convert_state_dict_to_diffusers
    from transformers import CLIPTextModel, CLIPTokenizer
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

            # Spatial-jitter tolerance: independently perturb the conditioning
            # (source) image's effective crop window via padding+re-crop, NOT
            # a registration step -- pairs are coordinate-corresponding, not
            # pixel-exact, so the ControlNet must not learn to expect perfect
            # alignment between source and target.
            if self.jitter > 0:
                dx = random.randint(-self.jitter, self.jitter)
                dy = random.randint(-self.jitter, self.jitter)
                padded = np.pad(src_rgb, ((self.jitter, self.jitter), (self.jitter, self.jitter), (0, 0)),
                                mode="reflect")
                y0, x0 = self.jitter + dy, self.jitter + dx
                src_rgb = padded[y0:y0 + self.res, x0:x0 + self.res]

            # 6-channel conditioning: source RGB (appearance, ticket step 3) +
            # source Canny (explicit structure, ticket step 4) -- complementary
            # signals, not alternatives (see module docstring).
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
    # Model: frozen base UNet -> fresh ControlNet cloned from it -> THEN
    # add the LoRA (order matters: ControlNet must clone the pristine base,
    # not a base with an untrained-but-already-present LoRA adapter).
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

    print("Building fresh source-conditioning ControlNet via ControlNetModel.from_unet() "
          "(conditioning_channels=6: source RGB + Canny, zero-initialised conditioning "
          "layers, weights cloned from the pristine frozen UNet)...")
    controlnet = ControlNetModel.from_unet(unet, conditioning_channels=6)
    controlnet.train()
    # Freeze the backbone from_unet() CLONED from the UNet (down_blocks, mid_block,
    # conv_in, time_proj/time_embedding) -- it duplicates feature extraction the
    # UNet already learned, and training the whole clone end-to-end on <=50 pairs
    # is a real overfitting/memory risk the ticket's "lightweight adapter" wording
    # explicitly warns against. Only the genuinely NEW, zero-initialised parts get
    # trained: the 6-channel conditioning-image encoder and the zero-conv residual
    # output layers -- these are the actual "adapter" the ticket describes.
    controlnet.requires_grad_(False)
    TRAINABLE_CONTROLNET_PREFIXES = (
        "controlnet_cond_embedding",  # new: encodes the 6-channel source RGB+Canny input
        "controlnet_down_blocks",     # new: zero-initialised residual output convs
        "controlnet_mid_block",       # new: zero-initialised residual output conv
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
          f"output layers), {n_frozen:,} frozen (cloned UNet backbone).")

    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    n_lora = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Trainable params -- LoRA (rank {args.rank}): {n_lora:,}  "
          f"ControlNet (fresh, source-conditioning branch): {n_controlnet:,}  "
          f"Total: {n_lora + n_controlnet:,}")

    vae.to(device); text_encoder.to(device); unet.to(device); controlnet.to(device)
    vae.eval(); text_encoder.eval(); unet.train(); controlnet.train()

    with torch.no_grad():
        tok = tokenizer(args.prompt, padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
        prompt_embeds = text_encoder(tok.input_ids)[0]

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
            dist = vae.encode(target_pixels).latent_dist
            # generator is not None only during evaluate() -- use the deterministic
            # mean (.mode(), zero variance) there so val_loss's comparability across
            # checkpoints isn't undermined by the VAE's own encoding stochasticity
            # (the noise/timestep generator above only fixes THAT source of
            # randomness, not this one). Training keeps .sample() (proper stochastic
            # training, matches train_colour_lora.py's convention).
            latents = (dist.mode() if generator is not None else dist.sample()) * scaling
        bsz = latents.shape[0]
        if generator is not None:
            # Deterministic noise/timestep draw -- ONLY used by evaluate() below, so
            # val_loss is directly comparable across checkpoints (same draws every
            # call). Training keeps using the global RNG (generator=None) as before.
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

        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=(amp_dtype is not None and device.type == "cuda")):
            down_res, mid_res = controlnet(
                noisy, timesteps, encoder_hidden_states=enc,
                controlnet_cond=control_images, return_dict=False)
            model_pred = unet(
                noisy, timesteps, encoder_hidden_states=enc,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res).sample
            loss = F.mse_loss(model_pred.float(), target.float())
        return loss

    @torch.no_grad()
    def evaluate():
        if val_loader is None:
            return None
        unet.eval(); controlnet.eval()
        # Fixed seed every call (not the global RNG) -- otherwise each evaluate()
        # call draws different noise/timesteps and val_loss across checkpoints is
        # not directly comparable, which is exactly what "final step is not
        # automatically best" checkpoint selection needs to be reliable.
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
        StableDiffusionPipeline.save_lora_weights(
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
                # Ticket requirement: "final step is not automatically the best
                # checkpoint" -- track and save the best-val-loss checkpoint
                # separately (val_loss is now deterministic across calls, see
                # evaluate(), so this comparison is meaningful, not noise).
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
          "infer_colour_translation.py --source-mode {correct,zero,shuffled}.")


if __name__ == "__main__":
    main()
