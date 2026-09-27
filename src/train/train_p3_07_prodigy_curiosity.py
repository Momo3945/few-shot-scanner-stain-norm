#!/usr/bin/env python3
"""
train_p3_07_prodigy_curiosity.py -- informal curiosity test: does swapping
AdamW for the Prodigy optimizer change anything about SDXL source-conditioned
colour-LoRA + ControlNet training?

**Not a ticketed experiment.** This is NOT "P3-07d" and is not tracked
against docs/proposal.tex's H1/RQ1 the way P3-07/P3-07b/P3-07c are -- it
exists purely because the optimizer choice (Prodigy, learning-rate-free)
came up while looking at an unrelated personal LoRA-training config
(kohya_ss GUI presets for an anime-character LoRA, nothing to do with this
project) and seemed worth trying here out of curiosity. Do not cite this
script's results in the thesis as a validated ablation unless it's later
promoted to a proper ticket with its own overfit-control run.

**IDENTICAL in every other respect to train_p3_07c_lora_sdxl.py** (itself
identical to train_p3_07b_lora_sdxl.py / train_colour_translation_lora_sdxl.py):
same SDXL base, same frozen-UNet + LoRA (rank 8, target_modules to_k/to_q/
to_v/to_out.0), same fresh 6-channel source-conditioning ControlNet
(ControlNetModel.from_unet, conditioning_channels=6, only the conditioning
encoder + zero-conv output layers trainable), same leak-checked
build_pairs() (A03/H03 only), same leave-one-frame-out val split, same
SDXL added_cond_kwargs wiring. The ONLY variable that changes is the
optimizer: torch.optim.AdamW(lr=1e-4) -> prodigyopt.Prodigy(lr=1.0, ...).

Why Prodigy needs a different --lr default: it is a learning-rate-FREE
method -- lr acts as a scale multiplier on its own internally-estimated
step size (the "D-adaptation" family), not a literal step size. lr=1.0 is
Prodigy's own documented convention (also what every kohya_ss preset in
this codebase's unrelated test/ folder used) -- do NOT reuse AdamW's
1e-4 here, it would silently cripple the optimizer's self-tuning.

Prodigy hyperparameters below (decouple/weight_decay/d_coef/
use_bias_correction/safeguard_warmup/betas) are carried over verbatim from
those same kohya_ss presets, which were tuned for a completely different
task (anime-character LoRA on a handful of captioned images, illustriousXL
base) -- they are a reasonable, real-world-tested starting point, NOT
validated for this project's domain-transfer + ControlNet setup. Treat
every one of these as a free variable for your own curiosity poking, not
as an authoritative choice.

**Memory note (found empirically, job 60753, 2026-09-27):** Prodigy's
D-adaptation bookkeeping needs extra full-sized per-parameter buffers
(p0/s) beyond AdamW's usual exp_avg/exp_avg_sq -- at slice_p=1 (full
precision) this OOM'd at step 4 on a 24GB RTX 3090 running the identical
architecture that fits fine under AdamW (GPU already at 23.49/23.55 GiB
even in the validated P3-07c run). --prodigy-slice-p (default 11, upstream's
own suggested value for large models) computes those buffers on only every
Nth flattened parameter entry instead of the full tensor -- an approximation
to standard Prodigy, but the only way this fits in memory at all here.

Usage
-----
    # smoke test (env/loop check only):
    python train_p3_07_prodigy_curiosity.py --pairs-dir <pairs>/train_1024_full96 \
        --direction A2H --output-dir <out>/p3_07_prodigy_smoke --smoke

    # real run (same pairs-dir shape as P3-07b's a2h_cond_r8_sdxl_1024_full96,
    # or P3-07c's overlap144 -- point --pairs-dir at whichever you want to
    # compare against; this script makes no claim about which is "correct"):
    python train_p3_07_prodigy_curiosity.py --pairs-dir <pairs>/train_1024_full96 \
        --direction A2H --rank 8 --train-steps 4000 --resolution 1024 \
        --output-dir <out>/prodigy_curiosity_full96

Dependencies: torch, diffusers, transformers, peft, safetensors, Pillow,
numpy, opencv-python-headless, prodigyopt (pip install prodigyopt -- NOT
yet in the stainnorm conda env as of 2026-09-27, the slurm launcher installs
it on demand). Requires canny.py on the path (src/eval/, reached via a
relative sys.path insert since this script lives in src/train/).
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
# Pure helpers -- IDENTICAL to train_p3_07c_lora_sdxl.py, copied verbatim
# (this script changes the optimizer only, nothing about the leak check,
# hashing, or CLI surface for data handling).
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
                         f"(this script's scope is A03/H03 only, matching P3-06/P3-07/P3-07b/P3-07c).")
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
        description="Informal curiosity test: Prodigy optimizer swap on the SDXL "
                    "source-conditioned colour-LoRA + ControlNet architecture "
                    "(P3-06/P3-07/P3-07b/P3-07c's architecture, unmodified). "
                    "NOT a ticketed experiment -- see module docstring.")
    ap.add_argument("--pairs-dir", required=True,
                    help="Directory with *_aperio.png / *_hamamatsu.png pairs (A03/H03 only).")
    ap.add_argument("--direction", choices=["A2H", "H2A"], required=True,
                    help="A2H: source=Aperio, target=Hamamatsu. H2A: reverse.")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0",
                    help="HF repo id (resolved from HF_HOME cache; runs offline).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank", type=int, default=8, help="Colour LoRA rank -- fixed at P3-07's value.")
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 -> defaults to rank.")
    ap.add_argument("--resolution", type=int, default=1024,
                    help="Matches P3-07/P3-07b/P3-07c's native-1024 setup.")
    ap.add_argument("--jitter", type=int, default=8,
                    help="Max px random crop-offset jitter applied to the conditioning "
                         "(source) image only. 0 disables it.")
    ap.add_argument("--train-steps", type=int, default=4000,
                    help="Fixed at P3-07/P3-07b/P3-07c's step count for rough comparability.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1.0,
                    help="Prodigy's own convention: this is a SCALE on its self-estimated "
                         "step size, not a literal LR. 1.0 is Prodigy's documented default -- "
                         "do NOT reuse AdamW's 1e-4 here (see module docstring).")
    # Prodigy-specific hyperparameters, carried over verbatim from the kohya_ss
    # presets that prompted this curiosity test (test/*.json in this repo --
    # an unrelated anime-LoRA project, not tuned for this domain). All exposed
    # as flags so they're trivial to poke without editing this file.
    ap.add_argument("--prodigy-weight-decay", type=float, default=0.01)
    ap.add_argument("--prodigy-d-coef", type=float, default=2.0)
    ap.add_argument("--prodigy-beta1", type=float, default=0.9)
    ap.add_argument("--prodigy-beta2", type=float, default=0.999)
    ap.add_argument("--prodigy-decouple", action="store_true", default=True)
    ap.add_argument("--no-prodigy-decouple", dest="prodigy_decouple", action="store_false")
    ap.add_argument("--prodigy-use-bias-correction", action="store_true", default=True)
    ap.add_argument("--no-prodigy-use-bias-correction", dest="prodigy_use_bias_correction",
                    action="store_false")
    ap.add_argument("--prodigy-safeguard-warmup", action="store_true", default=False)
    ap.add_argument("--prodigy-slice-p", type=int, default=11,
                    help="Prodigy's own documented memory-saving knob: compute its D-adaptation "
                         "statistics (p0/s buffers) on only every Nth flattened parameter entry "
                         "instead of the full tensor. 1 = full precision, no savings -- empirically "
                         "OOMs on this project's 24GB RTX 3090 at this SDXL+ControlNet size (job "
                         "60753, OOM at step 4, GPU already at 23.49/23.55 GiB even under the "
                         "validated AdamW run). ~11 is upstream's own suggested default for large "
                         "models -- an approximation to standard Prodigy, not exact, but the only "
                         "way this fits in memory at all. Set to 1 only if you have more VRAM to spare.")
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
                         "check. NOT a mandatory overfit control -- use --overfit-n for that if "
                         "this ever gets promoted to a real ablation.")
    ap.add_argument("--overfit-n", type=int, default=0,
                    help="Restrict to the first N pairs and train longer -- useful here "
                         "specifically to sanity-check that Prodigy's self-tuned step size "
                         "actually drives the loss down at all before spending a real run on it.")
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

    try:
        from prodigyopt import Prodigy
    except ImportError:
        raise SystemExit(
            "prodigyopt not installed -- `pip install prodigyopt` in the stainnorm env "
            "(the slurm launcher for this script does this automatically).")

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
    # (identical to train_p3_07c_lora_sdxl.py)
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

    print(f"Prodigy curiosity test -- Direction {args.direction}: {len(all_pairs)} pairs, "
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
    # Model: SDXL's dual text encoders + frozen UNet -> fresh 6-channel
    # ControlNet cloned from it -> THEN the LoRA. Identical to P3-07c.
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

    unet.enable_gradient_checkpointing()

    vae.to(device)
    text_encoder.to(device); text_encoder_2.to(device)
    unet.to(device); controlnet.to(device)
    vae.eval(); text_encoder.eval(); text_encoder_2.eval()
    unet.train(); controlnet.train()

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

    add_time_ids = torch.tensor(
        [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
        device=device, dtype=torch.float32,
    )

    trainable = [p for p in unet.parameters() if p.requires_grad] + \
        [p for p in controlnet.parameters() if p.requires_grad]

    # THE ONE CHANGE vs train_p3_07c_lora_sdxl.py: Prodigy instead of AdamW.
    optimizer = Prodigy(
        trainable, lr=args.lr, betas=(args.prodigy_beta1, args.prodigy_beta2),
        weight_decay=args.prodigy_weight_decay, d_coef=args.prodigy_d_coef,
        decouple=args.prodigy_decouple, use_bias_correction=args.prodigy_use_bias_correction,
        safeguard_warmup=args.prodigy_safeguard_warmup, slice_p=args.prodigy_slice_p,
    )
    print(f"Optimizer: Prodigy (lr scale={args.lr}, d_coef={args.prodigy_d_coef}, "
          f"weight_decay={args.prodigy_weight_decay}, decouple={args.prodigy_decouple}, "
          f"use_bias_correction={args.prodigy_use_bias_correction}, "
          f"safeguard_warmup={args.prodigy_safeguard_warmup}, slice_p={args.prodigy_slice_p}) -- "
          f"everything else identical to train_p3_07c_lora_sdxl.py.")
    if args.prodigy_slice_p > 1:
        print(f"  NOTE: slice_p={args.prodigy_slice_p} > 1 -- this is upstream's documented "
              f"approximation to standard Prodigy (D-adaptation stats computed on every "
              f"{args.prodigy_slice_p}th flattened param entry, not the full tensor), needed to "
              f"fit in 24GB VRAM at this SDXL+ControlNet size. Not exact Prodigy.")

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": None}[args.mixed_precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    scaling = vae.config.scaling_factor

    def run_step(target_pixels, control_images, generator=None):
        target_pixels = target_pixels.to(device, dtype=torch.float32)
        control_images = control_images.to(device, dtype=torch.float32)

        with torch.no_grad():
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
    # Training loop -- identical to train_p3_07c_lora_sdxl.py
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

    print(f"Training for {args.train_steps} steps (batch {args.batch_size}, "
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
                # Prodigy exposes its own self-estimated step size ("d") per
                # param group -- log it so you can actually see whether it's
                # doing anything sane, since there's no manually-set lr to eyeball.
                d_val = optimizer.param_groups[0].get("d", None)
                d_str = f"  prodigy_d={d_val:.3e}" if d_val is not None else ""
                print(f"  step {step:5d}/{args.train_steps}  loss {avg:.4f}{d_str}  ({sec:.1f}s)")
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
                                "prediction_type": noise_scheduler.config.prediction_type,
                                "optimizer": "prodigy",
                                "not_a_ticketed_experiment": True},
                  fh, indent=2)
    print(f"\nDone. Final LoRA + ControlNet in {out_dir/'final'}. Config + loss log written.")
    print("Informal curiosity test (Prodigy optimizer) -- NOT a ticketed ablation. "
          "Compare loss_log.csv/training_config.json's val_loss against the AdamW run "
          "you're curious about (P3-07/P3-07b/P3-07c) yourself before drawing conclusions "
          "-- no automated comparison is done here.")
    print("Next (mandatory before trusting this run for anything beyond curiosity): "
          "source-conditioning ablation via infer_colour_translation_sdxl.py "
          "--source-mode {correct,zero,shuffled}.")


if __name__ == "__main__":
    main()
