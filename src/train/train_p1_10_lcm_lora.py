#!/usr/bin/env python3
"""
train_p1_10_lcm_lora.py -- P1-13: distil a task-specific LCM-LoRA directly
from the frozen P1-10 source-conditioned teacher (lora/a2h_cond_r8/best),
instead of relying on the generic pretrained latent-consistency/lcm-lora-
sdv1-5 checkpoint P1-12 found attenuates P1-10's scanner-colour mapping at
few inference steps (see tickets/PHASE1-TICKETS.md P1-12/P1-13 and
tickets/P1-13_task_specific_lcm_distillation.md for the full spec and the
prerequisite-gate evidence that justified this ticket).

Algorithm: single-GPU port of diffusers v0.39.0's official LoRA consistency-
distillation recipe (examples/consistency_distillation/
train_lcm_distill_lora_sd_wds.py -- confirmed to be the version actually
installed on the cluster before writing this), with its accelerate/
webdataset scaffolding dropped in favour of this project's own data
pipeline. DDIMSolver, scalings_for_boundary_conditions,
get_predicted_original_sample, get_predicted_noise, append_dims and
extract_into_tensor below are ported near-verbatim from that reference --
small, dependency-free helpers implementing the LCM paper's boundary
conditions and multistep ODE-solver target, not reinvented here.

Model composition (verified against the actual installed diffusers/peft
versions, not assumed):
  - Frozen base UNet + VAE + text encoder + DDPMScheduler, loaded exactly as
    train_colour_translation_lora.py does.
  - Frozen ControlNet: the P1-10 source-conditioning branch, loaded from
    <teacher>/controlnet.
  - Frozen P1-10 colour LoRA, loaded as a NAMED PEFT adapter via
    unet.load_lora_adapter(<teacher>, prefix="unet", adapter_name="colour")
    -- UNet2DConditionModel.load_lora_adapter is confirmed present in
    diffusers 0.39.0, the checkpoint's safetensors keys are confirmed
    prefixed "unet." (e.g. "unet.down_blocks.0...to_k.lora.down.weight"), so
    prefix="unet" is verified correct, and the loader auto-infers the
    LoraConfig (rank/target_modules) from the state dict itself -- no need
    to hand-reconstruct P1-10's training-time LoraConfig here.
  - New trainable LoRA, adapter_name="lcm": attention-only
    (to_k/to_q/to_v/to_out.0, same target modules as the colour LoRA),
    rank 64 -- narrower than the official lcm-lora-sdv1-5's full-UNet
    coverage (confirmed via its own safetensors keys: it also touches
    proj_in/proj_out, resnet convs, up/down-samplers and time_emb_proj) but
    matching its rank, since only 39 training pairs are available here and
    full-UNet coverage was judged too high an overfitting risk for a first
    pass (see P1-13 ticket entry, PHASE1-TICKETS.md).
  unet.set_adapters(["colour", "lcm"]) keeps both active on every forward
  pass -- the same multi-adapter pattern already validated at inference in
  infer_colour_lora.py's --lora/--hist-lora/--lcm composition.

Teacher vs student, per the ticket's Conditional Distillation Requirement:
both receive the SAME correct-source ControlNet residuals (down_res/
mid_res) computed once per training example from the real per-crop 6-
channel (source RGB + Canny) conditioning tensor -- CFG (used only for the
teacher's PF-ODE step) varies solely the TEXT embedding (cond vs uncond),
never the source condition. The teacher is the frozen UNet+colour-LoRA+
ControlNet stack (adapters ["colour"] only); the student is the same stack
with adapters ["colour", "lcm"]. Per the reference script's own documented
convention, LoRA distillation needs no separate EMA target network: the
"target" prediction at x_prev/timesteps reuses the identical student stack,
just under torch.no_grad().

Never touches lora/a2h_cond_r8/best or train_colour_translation_lora.py.
New checkpoint namespace: lora/a2h_cond_r8_lcm_distilled/. Only the "lcm"
adapter's weights are ever saved.

Checkpoint selection: the scalar distillation val_loss (out_dir/best) is
tracked but is NOT trusted as the final selection criterion -- confirmed on
a real 39-pair run where the lowest-val_loss checkpoint (ssim=0.288,
lab_total=32.70) was one of the WORST checkpoints in the entire sweep on
actual image-level Validation B, while loss itself decreased almost
monotonically throughout training. At the end of training, every saved
checkpoint (checkpoint-N/, final/, and best/ if it exists) is instead swept
through the real Validation B teacher-matching comparison (teacher generated
once per pair, reused across checkpoints). Selection rule (precommitted,
P1-13): among checkpoints with ssim >= 0.95 x the best ssim in the sweep (a
structural-fidelity floor), pick the lowest lab_total -- NOT simply the
lowest lab_total on the Pareto frontier, since a frontier point can still
have collapsed SSIM (traded away too much structure for colour recovery).
The Pareto frontier is still computed and reported for reference. The
selected checkpoint is copied to out_dir/best_by_validation_b -- THAT is the
checkpoint downstream scripts should use, not out_dir/best or out_dir/final.
Never uses held-out slides to choose between checkpoints -- only the
internal training-domain validation split.

Trainable-parameter safety check (ticket requirement): at startup, logs
total/trainable parameter counts and every trainable parameter's name, and
aborts via SystemExit if any parameter outside the "lcm" LoRA adapter shows
requires_grad=True (base UNet, colour LoRA, ControlNet, VAE, text encoder
must all be frozen).

Data: reuses build_pairs() from train_colour_translation_lora.py unchanged
(same leak check restricting --pairs-dir to A03/H03). PairDataset is this
script's own standalone copy of that file's identically-named nested class
(same 6-channel conditioning construction, same jitter convention) --
train_colour_translation_lora.py defines it inside main(), not importable,
and this script never edits that file. --seed 0 with the same frame-level
split logic deterministically reproduces the exact 39-train/11-val split
already recorded in the P1-10 teacher's own training_config.json; this
script loads that JSON and asserts its own recreated val_frames matches, so
the reuse is verified, not assumed.

Usage
-----
    # 5-step smoke test (env/loop check only):
    python train_p1_10_lcm_lora.py --pairs-dir <pairs>/train \
        --teacher-dir <out>/a2h_cond_r8/best --output-dir <out>/a2h_cond_r8_lcm_distilled/smoke --smoke

    # real run:
    python train_p1_10_lcm_lora.py --pairs-dir <pairs>/train \
        --teacher-dir <out>/a2h_cond_r8/best \
        --output-dir <out>/a2h_cond_r8_lcm_distilled --train-steps 4000

Dependencies: torch, diffusers, transformers, peft, safetensors, Pillow,
numpy, opencv-python-headless. Requires canny.py and metrics.py on the path
(src/eval/) and train_colour_translation_lora.py (src/train/, this file's
own directory).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

# canny.py/metrics.py live in src/eval, a sibling of this file's parent (src/train)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))


# ----------------------------------------------------------------------
# Ported near-verbatim from diffusers v0.39.0's
# examples/consistency_distillation/train_lcm_distill_lora_sd_wds.py
# ----------------------------------------------------------------------
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dims."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    """From LCMScheduler.get_scalings_for_boundary_condition_discrete."""
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    """Compare LCMScheduler.step, Step 4. Never hardcode prediction_type -- read it
    from noise_scheduler.config.prediction_type, same convention as
    train_colour_translation_lora.py."""
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} is not supported; currently, "
                         f"`epsilon`, `sample`, and `v_prediction` are supported.")
    return pred_x_0


def get_predicted_noise(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    """Based on step 4 in DDIMScheduler.step."""
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(f"Prediction type {prediction_type} is not supported; currently, "
                         f"`epsilon`, `sample`, and `v_prediction` are supported.")
    return pred_epsilon


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class PairDataset:
    """A2->H2 pair dataset: source RGB + Canny (6-channel control) plus the
    target pixel tensor. Deliberately a standalone copy of
    train_colour_translation_lora.py's identically-named nested class (that
    one is defined INSIDE its main(), not importable, and this script never
    touches train_colour_translation_lora.py) -- same 6-channel conditioning
    construction, same jitter convention, hardcoded to A2H (aperio source /
    hamamatsu target) since the P1-10 teacher this ticket distills from
    (lora/a2h_cond_r8) is itself A2H-only."""

    def __init__(self, pairs, resolution, jitter):
        self.pairs = pairs
        self.res = resolution
        self.jitter = jitter

    def __len__(self):
        return len(self.pairs)

    def _load(self, path):
        import numpy as np
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if img.size != (self.res, self.res):
            img = img.resize((self.res, self.res), Image.LANCZOS)
        return np.asarray(img)

    def __getitem__(self, i):
        import random as _random
        import numpy as np
        import torch
        from canny import extract_canny_control_image

        p = self.pairs[i]
        src_rgb = self._load(p["aperio_path"])
        tgt_rgb = self._load(p["hamamatsu_path"])

        if self.jitter > 0:
            dx = _random.randint(-self.jitter, self.jitter)
            dy = _random.randint(-self.jitter, self.jitter)
            padded = np.pad(src_rgb, ((self.jitter, self.jitter), (self.jitter, self.jitter), (0, 0)),
                            mode="reflect")
            y0, x0 = self.jitter + dy, self.jitter + dx
            src_rgb = padded[y0:y0 + self.res, x0:x0 + self.res]

        canny = extract_canny_control_image(src_rgb)
        rgb_t = torch.from_numpy(src_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        canny_t = torch.from_numpy(canny.astype(np.float32) / 255.0).permute(2, 0, 1)
        control = torch.cat([rgb_t, canny_t], dim=0)

        tgt_arr = np.asarray(tgt_rgb, dtype=np.float32) / 127.5 - 1.0
        target = torch.from_numpy(tgt_arr).permute(2, 0, 1)
        return target, control


class DDIMSolver:
    """Multistep DDIM ODE solver used to jump from t_{n+k} to t_n along the
    augmented probability-flow ODE trajectory (LCM paper Algorithm 1)."""

    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        import numpy as np
        import torch
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist())
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        import torch
        # float32 cast is required, not cosmetic: ddim_alpha_cumprods_prev is
        # built via a numpy .tolist() round-trip (see __init__) which silently
        # upcasts float32 -> Python float (64-bit) -- left uncorrected, that
        # float64 buffer propagates into x_prev and collides with the bf16
        # ControlNet/UNet weights ("mat1 and mat2 must have the same dtype").
        # Same latent asymmetry exists in the official reference script this
        # class was ported from (ddim_alpha_cumprods, built without a
        # .tolist() round-trip, doesn't have it).
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device=device, dtype=torch.float32)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device=device, dtype=torch.float32)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


def parse_args():
    ap = argparse.ArgumentParser(
        description="P1-13: distil a task-specific LCM-LoRA from the frozen P1-10 teacher.")
    ap.add_argument("--pairs-dir", required=True,
                    help="Same A03/H03 training pairs P1-10 was trained on (e.g. pairs/train).")
    ap.add_argument("--teacher-dir", required=True,
                    help="Frozen P1-10 checkpoint dir, e.g. lora/a2h_cond_r8/best -- must contain "
                         "pytorch_lora_weights.safetensors, controlnet/, and (in its parent) "
                         "training_config.json for the val-split cross-check.")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lcm-rank", type=int, default=64, help="Rank of the new trainable LCM-LoRA adapter.")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--jitter", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=1, help="Microbatch size per forward/backward pass.")
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="Accumulate gradients over this many microbatches before each optimizer "
                         "step -- effective batch size = --batch-size x --grad-accum-steps. Reduces "
                         "per-step target variance (random timestep/index/guidance draw every "
                         "microbatch) without changing LR or any other hyperparameter.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--mixed-precision", choices=["bf16", "fp16", "no"], default="bf16")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="Must match train_colour_translation_lora.py's --val-frac (default 0.2) "
                         "for the val-split cross-check against the teacher's training_config.json "
                         "to pass.")
    ap.add_argument("--num-ddim-timesteps", type=int, default=50,
                    help="ODE-solver discretisation of the teacher trajectory. Reference-script "
                         "default; matches P1-10's own 50-step DDIM teacher reference point.")
    ap.add_argument("--w-min", type=float, default=1.0,
                    help="Per-example CFG guidance weight sampled from U[w_min, w_max] during "
                         "distillation, using the LCM PAPER's CFG form pred_x0 = cond + "
                         "w*(cond-uncond) for the teacher target (see pred_x0/pred_noise "
                         "combination in run_distill_step). This is NOT the same scale as "
                         "diffusers' ordinary guidance_scale (G), which uses "
                         "uncond + G*(cond-uncond) -- equating the two forms gives G = w + 1. "
                         "So w~U[1.0,2.0] (this default) corresponds to teacher trajectories at "
                         "diffusers-style guidance G~U[2.0,3.0], NOT an inference guidance_scale "
                         "range of 1.0-2.0 -- do not conflate the two when choosing an inference "
                         "guidance_scale for the resulting adapter (see P1-13 ticket note on the "
                         "train/inference guidance-semantics distinction this caused).")
    ap.add_argument("--w-max", type=float, default=2.0)
    ap.add_argument("--loss-type", choices=["l2", "huber"], default="huber")
    ap.add_argument("--huber-c", type=float, default=0.001)
    ap.add_argument("--timestep-scaling-factor", type=float, default=10.0)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="5-step dry run: env/loop check only, not a training or overfit control.")
    ap.add_argument("--overfit-n", type=int, default=0,
                    help="Restrict to the first N pairs (0 = all) -- mandatory wiring control "
                         "before trusting a full run, mirrors train_colour_translation_lora.py's "
                         "own --overfit-n.")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.train_steps = 5
        args.save_every = 5
        args.log_every = 1

    import os
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    import torch.nn.functional as F

    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig

    from metrics import score_aligned_pair  # src/eval/metrics.py

    # train_colour_translation_lora.py lives in this same directory (src/train).
    # Only build_pairs is imported from it (module-level, safe) -- PairDataset
    # is this file's own standalone copy (see class above), since the
    # original is nested inside that script's main() and this script never
    # touches that file.
    from train_colour_translation_lora import build_pairs

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("No CUDA device -- this script is meant for a GPU node.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir = Path(args.teacher_dir)

    # ------------------------------------------------------------------
    # Reproduce P1-10's exact frame-level train/val split, and verify it
    # against the teacher's own recorded split rather than assuming.
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

    if not args.overfit_n:
        teacher_config_path = teacher_dir.parent / "training_config.json"
        if teacher_config_path.exists():
            with open(teacher_config_path) as fh:
                teacher_config = json.load(fh)
            teacher_val_frames = sorted(teacher_config.get("val_frames", []))
            if teacher_val_frames and teacher_val_frames != sorted(val_frames):
                raise SystemExit(
                    f"Val-split mismatch against teacher's training_config.json: "
                    f"teacher val_frames={teacher_val_frames}, recreated val_frames={sorted(val_frames)}. "
                    f"Check --seed/--val-frac/--pairs-dir/--overfit-n match the teacher run exactly.")
            print(f"Val-split cross-check OK: recreated val_frames match teacher's "
                  f"training_config.json ({teacher_val_frames}).")
        else:
            print(f"WARNING: {teacher_config_path} not found -- skipping val-split cross-check.")

    print(f"{len(all_pairs)} pairs, {len(frame_ids)} frames -> {len(val_frames)} held out for val "
          f"({len(train_pairs)} train pairs / {len(val_pairs)} val pairs).")

    from torch.utils.data import DataLoader
    train_loader = DataLoader(PairDataset(train_pairs, args.resolution, args.jitter),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=(len(train_pairs) >= args.batch_size), pin_memory=True)
    val_loader = (DataLoader(PairDataset(val_pairs, args.resolution, jitter=0),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
                 if val_pairs else None)

    # ------------------------------------------------------------------
    # Model composition: frozen base + frozen ControlNet + frozen "colour"
    # LoRA (teacher) + trainable "lcm" LoRA (student), both adapters active
    # together via set_adapters.
    # ------------------------------------------------------------------
    print(f"Loading SD 1.5 components from cache ({args.model}) ...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    prediction_type = noise_scheduler.config.prediction_type
    print(f"noise_scheduler.config.prediction_type = {prediction_type!r}")

    print(f"Loading frozen P1-10 ControlNet from {teacher_dir / 'controlnet'} ...")
    controlnet = ControlNetModel.from_pretrained(str(teacher_dir / "controlnet"))
    controlnet.requires_grad_(False)

    print(f"Loading frozen P1-10 colour LoRA from {teacher_dir} as adapter 'colour' ...")
    unet.load_lora_adapter(str(teacher_dir), prefix="unet", adapter_name="colour",
                           weight_name="pytorch_lora_weights.safetensors")

    print(f"Adding trainable 'lcm' LoRA adapter (rank {args.lcm_rank}, attention-only) ...")
    lcm_lora_config = LoraConfig(
        r=args.lcm_rank, lora_alpha=args.lcm_rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lcm_lora_config, adapter_name="lcm")
    unet.set_adapters(["colour", "lcm"])

    # PEFT gotcha: add_adapter()/inject_adapter_in_model() marks EVERY LoRA
    # parameter across ALL adapters as requires_grad=True on injection --  it
    # does not respect an adapter frozen before a later adapter is added. So
    # the freeze/unfreeze pass has to happen HERE, after both "colour" and
    # "lcm" exist, not before -- a per-name pass is the only reliable way to
    # end up with exactly "lcm" trainable and everything else (base UNet +
    # "colour") frozen.
    for name, p in unet.named_parameters():
        p.requires_grad_("lcm" in name)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # ------------------------------------------------------------------
    # Trainable-parameter safety check (ticket requirement): log everything,
    # abort if anything outside the "lcm" adapter is trainable.
    # ------------------------------------------------------------------
    total_params = sum(p.numel() for p in unet.parameters()) + sum(p.numel() for p in controlnet.parameters())
    trainable_names = [n for n, p in unet.named_parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for n, p in unet.named_parameters() if p.requires_grad)
    bad = [n for n in trainable_names if "lcm" not in n]
    controlnet_trainable = [n for n, p in controlnet.named_parameters() if p.requires_grad]
    print(f"Total params (unet+controlnet): {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Trainable parameter names ({len(trainable_names)}):")
    for n in trainable_names:
        print(f"  {n}")
    if bad:
        raise SystemExit(
            f"ABORT: {len(bad)} trainable parameter(s) found outside the 'lcm' adapter -- "
            f"colour LoRA / base UNet must stay frozen. Offending names: {bad[:10]}")
    if controlnet_trainable:
        raise SystemExit(
            f"ABORT: {len(controlnet_trainable)} trainable parameter(s) found in the frozen "
            f"source-conditioning ControlNet. Offending names: {controlnet_trainable[:10]}")
    if trainable_params == 0:
        raise SystemExit("ABORT: 0 trainable parameters -- 'lcm' adapter did not attach correctly.")

    vae.to(device); text_encoder.to(device); unet.to(device); controlnet.to(device)
    vae.eval(); text_encoder.eval(); controlnet.eval()
    unet.train()

    with torch.no_grad():
        tok = tokenizer(args.prompt, padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
        prompt_embeds = text_encoder(tok.input_ids)[0]
        uncond_tok = tokenizer("", padding="max_length", truncation=True,
                               max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
        uncond_prompt_embeds = text_encoder(uncond_tok.input_ids)[0]

    optimizer = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=args.lr)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": None}[args.mixed_precision]
    scaling = vae.config.scaling_factor

    # Alpha/sigma schedules + DDIM solver over the noise_scheduler's own
    # alphas_cumprod, matching the reference script exactly.
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod).to(device)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod).to(device)
    solver = DDIMSolver(noise_scheduler.alphas_cumprod.numpy(),
                        timesteps=noise_scheduler.config.num_train_timesteps,
                        ddim_timesteps=args.num_ddim_timesteps).to(device)

    def controlnet_residuals(noisy, timesteps, enc, control_images):
        return controlnet(noisy, timesteps, encoder_hidden_states=enc,
                          controlnet_cond=control_images, return_dict=False)

    def run_distill_step(target_pixels, control_images):
        target_pixels = target_pixels.to(device, dtype=torch.float32)
        control_images = control_images.to(device, dtype=torch.float32)
        bsz = target_pixels.shape[0]

        with torch.no_grad():
            latents = vae.encode(target_pixels).latent_dist.sample() * scaling

        # 1. Sample t_{n+k} from the ODE-solver timestep grid, and its
        # (skip-k) neighbour t_n -- same convention as the reference script.
        topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
        index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=device).long()
        start_timesteps = solver.ddim_timesteps[index]
        timesteps = torch.clamp(start_timesteps - topk, min=0)

        c_skip_start, c_out_start = scalings_for_boundary_conditions(
            start_timesteps, timestep_scaling=args.timestep_scaling_factor)
        c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
        c_skip, c_out = scalings_for_boundary_conditions(
            timesteps, timestep_scaling=args.timestep_scaling_factor)
        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

        noise = torch.randn_like(latents)
        noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)

        w = (args.w_max - args.w_min) * torch.rand((bsz,), device=device) + args.w_min
        w = w.reshape(bsz, 1, 1, 1).to(dtype=latents.dtype)

        enc = prompt_embeds.expand(bsz, -1, -1)
        uncond_enc = uncond_prompt_embeds.expand(bsz, -1, -1)

        # 2. Online student prediction at t_{n+k} (adapters colour+lcm active,
        # correct source conditioning), WITH grad.
        unet.set_adapters(["colour", "lcm"])
        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=(amp_dtype is not None)):
            down_res, mid_res = controlnet_residuals(noisy_model_input, start_timesteps, enc, control_images)
            noise_pred = unet(noisy_model_input, start_timesteps, encoder_hidden_states=enc,
                              down_block_additional_residuals=down_res,
                              mid_block_additional_residual=mid_res).sample
        pred_x_0 = get_predicted_original_sample(noise_pred, start_timesteps, noisy_model_input,
                                                 prediction_type, alpha_schedule, sigma_schedule)
        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

        # 3. Teacher CFG (cond/uncond TEXT only, same correct source
        # conditioning both times) -> PF-ODE step -> x_prev. Teacher =
        # adapters ["colour"] only (lcm inactive).
        with torch.no_grad():
            unet.set_adapters(["colour"])
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=(amp_dtype is not None)):
                cond_down, cond_mid = controlnet_residuals(noisy_model_input, start_timesteps, enc, control_images)
                cond_teacher_output = unet(
                    noisy_model_input, start_timesteps, encoder_hidden_states=enc,
                    down_block_additional_residuals=cond_down,
                    mid_block_additional_residual=cond_mid).sample
                uncond_down, uncond_mid = controlnet_residuals(
                    noisy_model_input, start_timesteps, uncond_enc, control_images)
                uncond_teacher_output = unet(
                    noisy_model_input, start_timesteps, encoder_hidden_states=uncond_enc,
                    down_block_additional_residuals=uncond_down,
                    mid_block_additional_residual=uncond_mid).sample

            cond_pred_x0 = get_predicted_original_sample(cond_teacher_output, start_timesteps,
                                                         noisy_model_input, prediction_type,
                                                         alpha_schedule, sigma_schedule)
            cond_pred_noise = get_predicted_noise(cond_teacher_output, start_timesteps,
                                                  noisy_model_input, prediction_type,
                                                  alpha_schedule, sigma_schedule)
            uncond_pred_x0 = get_predicted_original_sample(uncond_teacher_output, start_timesteps,
                                                            noisy_model_input, prediction_type,
                                                            alpha_schedule, sigma_schedule)
            uncond_pred_noise = get_predicted_noise(uncond_teacher_output, start_timesteps,
                                                     noisy_model_input, prediction_type,
                                                     alpha_schedule, sigma_schedule)

            pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
            pred_noise = cond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)
            x_prev = solver.ddim_step(pred_x0, pred_noise, index)

            # 4. Target student prediction at x_prev/t_n, adapters
            # colour+lcm active again, NO grad (self-consistency target --
            # reference script's own note: no separate EMA target needed for
            # LoRA distillation).
            unet.set_adapters(["colour", "lcm"])
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=(amp_dtype is not None)):
                target_down, target_mid = controlnet_residuals(x_prev, timesteps, enc, control_images)
                target_noise_pred = unet(
                    x_prev, timesteps, encoder_hidden_states=enc,
                    down_block_additional_residuals=target_down,
                    mid_block_additional_residual=target_mid).sample
            target_pred_x_0 = get_predicted_original_sample(target_noise_pred, timesteps, x_prev,
                                                            prediction_type, alpha_schedule, sigma_schedule)
            target = c_skip * x_prev + c_out * target_pred_x_0

        if args.loss_type == "l2":
            loss = F.mse_loss(model_pred.float(), target.float())
        else:
            loss = torch.mean(torch.sqrt((model_pred.float() - target.float()) ** 2 + args.huber_c**2)
                              - args.huber_c)
        return loss

    @torch.no_grad()
    def evaluate():
        if val_loader is None:
            return None
        unet.eval()
        torch.manual_seed(args.seed)
        losses = []
        for target_pixels, control_images in val_loader:
            losses.append(run_distill_step(target_pixels, control_images).item())
        unet.train()
        return sum(losses) / len(losses) if losses else None

    def save_checkpoint(ckpt: Path):
        ckpt.mkdir(parents=True, exist_ok=True)
        unet.save_lora_adapter(str(ckpt), adapter_name="lcm")

    print(f"Training for {args.train_steps} optimizer steps (microbatch {args.batch_size} x "
          f"grad-accum {args.grad_accum_steps} = effective batch {args.batch_size * args.grad_accum_steps}, "
          f"lr {args.lr}, {args.mixed_precision}, w~U[{args.w_min},{args.w_max}], "
          f"num_ddim_timesteps={args.num_ddim_timesteps}, loss={args.loss_type}) ...")
    log_path = out_dir / "loss_log.csv"
    log_fh = open(log_path, "w", newline="")
    log_w = csv.writer(log_fh); log_w.writerow(["step", "loss", "val_loss", "sec"])

    step = 0  # optimizer steps (what --train-steps/--save-every/--log-every count)
    micro_step = 0  # forward/backward passes -- grad_accum_steps of these make one optimizer step
    t0 = time.time()
    running = 0.0
    best_val_loss = float("inf")
    done = False
    optimizer.zero_grad(set_to_none=True)
    while not done:
        for target_pixels, control_images in train_loader:
            unet.set_adapters(["colour", "lcm"])
            loss = run_distill_step(target_pixels, control_images)
            # Gradient accumulation: average the loss over grad_accum_steps
            # microbatches before an optimizer step, so a higher effective
            # batch size reduces the per-step target variance (random
            # timestep/index/guidance draw every microbatch) without
            # changing any other hyperparameter -- motivated by a confirmed
            # case (P1-13) where LAB kept oscillating between adjacent
            # checkpoints even after lowering LR alone stabilised SSIM,
            # pointing at gradient noise rather than LR as the remaining
            # driver. See tickets/PHASE1-TICKETS.md P1-13.
            (loss / args.grad_accum_steps).backward()
            micro_step += 1
            running += loss.item()
            if micro_step % args.grad_accum_steps != 0:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                avg = running / (args.log_every * args.grad_accum_steps)
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
                print(f"  saved 'lcm' adapter -> {ckpt}")
                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(out_dir / "best")
                    print(f"    new best (val_loss {val_loss:.4f}) -> {out_dir / 'best'}")

            if step >= args.train_steps:
                done = True
                break

    log_fh.close()
    with open(out_dir / "training_config.json", "w") as fh:
        json.dump(vars(args) | {"prediction_type": prediction_type,
                                "trainable_params_lcm": trainable_params,
                                "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
                                "val_frames": sorted(val_frames)},
                  fh, indent=2)
    print(f"\nDone. Final 'lcm' adapter in {out_dir/'final'}. Config + loss log written.")

    # ------------------------------------------------------------------
    # Validation B: teacher-matching diagnostic, using the BEST checkpoint
    # (falls back to 'final' if none was saved as best). Not the biological
    # held-out evaluation -- a teacher-fidelity check. Normally run on the
    # held-out val split; under --overfit-n there IS no val split (by
    # design), so this instead runs on the overfit training pairs themselves
    # -- the only way to tell whether the noisy/flat per-step loss the
    # ticket's own guardrail warns not to trust actually corresponds to a
    # learned adapter, since the loss curve alone is explicitly not a valid
    # success signal in this project (see CLAUDE.md).
    # ------------------------------------------------------------------
    diag_pairs = val_pairs if val_pairs else (train_pairs if args.overfit_n else [])
    if diag_pairs:
        diag_label = "held-out val split" if val_pairs else "overfit training pairs (no val split under --overfit-n)"

        # Sweep EVERY saved checkpoint, not just one. Motivated by a confirmed
        # case (P1-13, 8-pair overfit ablation) where the scalar distillation
        # val_loss doesn't track actual teacher-matching quality: checkpoints
        # oscillated non-monotonically, with an early checkpoint scoring far
        # better on Validation B than later ones despite a similar/higher
        # loss. Selecting "best" by val_loss alone (or by the final step) is
        # therefore not trustworthy here -- image-level Validation B is,
        # consistent with this project's standing guardrail that diffusion
        # training loss is not a valid success signal on its own (CLAUDE.md).
        import re
        ckpt_candidates = []
        for d in sorted(out_dir.glob("checkpoint-*")):
            m = re.match(r"checkpoint-(\d+)$", d.name)
            if m and (d / "pytorch_lora_weights.safetensors").exists():
                ckpt_candidates.append((d.name, d))
        ckpt_candidates.sort(key=lambda t: int(t[0].split("-")[1]))
        if (out_dir / "best").exists() and (out_dir / "best" / "pytorch_lora_weights.safetensors").exists():
            ckpt_candidates.append(("best", out_dir / "best"))
        if (out_dir / "final" / "pytorch_lora_weights.safetensors").exists():
            ckpt_candidates.append(("final", out_dir / "final"))
        print(f"\nValidation B: sweeping {len(ckpt_candidates)} saved checkpoint(s) "
              f"({diag_label}) -- {[t for t, _ in ckpt_candidates]} ...")

        # Reuse the already-validated StableDiffusionControlNetImg2ImgPipeline
        # mechanics (img2img strength handling, ControlNet residual injection,
        # CFG) instead of hand-rolling an equivalent sampling loop -- built
        # directly from the in-memory, adapter-loaded components so the
        # trained "colour" adapter carries over (a fresh from_pretrained()
        # load, as infer_colour_translation.py uses, would NOT have it
        # attached).
        from diffusers import DDIMScheduler, LCMScheduler, StableDiffusionControlNetImg2ImgPipeline
        from canny import extract_canny_control_image
        from PIL import Image
        import shutil

        eval_pipe = StableDiffusionControlNetImg2ImgPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet, scheduler=DDIMScheduler.from_config(noise_scheduler.config),
            safety_checker=None, feature_extractor=None, requires_safety_checker=False)
        eval_pipe.set_progress_bar_config(disable=True)

        def build_control_tensor(rgb):
            canny = extract_canny_control_image(rgb)
            rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            canny_t = torch.from_numpy(canny.astype(np.float32) / 255.0).permute(2, 0, 1)
            return torch.cat([rgb_t, canny_t], dim=0).unsqueeze(0)

        # Teacher: P1-10's own quality defaults (infer_colour_translation.py's
        # --steps 50/--strength 0.50/--guidance 2.0), generated ONCE per pair
        # and reused across every checkpoint (the teacher never changes).
        # Student op point: guidance=1.0 (no external CFG) -- the student is
        # trained with a single conditional forward pass only (see
        # run_distill_step: no w-embedding fed to the UNet), so any
        # guidance_scale > 1.0 here would trigger diffusers' internal
        # cond/uncond doubling on a model never trained to expect it. See
        # tickets/PHASE1-TICKETS.md P1-13 "train/inference guidance-semantics
        # mismatch" note. NOT job 45650's op point (that predates this fix).
        teacher_outputs, control_tensors = [], []
        for p in diag_pairs:
            src_rgb = np.asarray(Image.open(p["aperio_path"]).convert("RGB"))
            control_tensor = build_control_tensor(src_rgb)
            control_tensors.append(control_tensor)
            unet.set_adapters(["colour"])
            eval_pipe.scheduler = DDIMScheduler.from_config(noise_scheduler.config)
            gen = torch.Generator(device=device).manual_seed(0)
            teacher_outputs.append(np.asarray(eval_pipe(
                prompt=args.prompt, image=Image.fromarray(src_rgb), strength=0.50,
                num_inference_steps=50, guidance_scale=2.0,
                control_image=control_tensor, generator=gen).images[0]))

        sweep_results = {}
        for tag, ckpt_dir in ckpt_candidates:
            adapter_name = f"lcm_{tag.replace('-', '_')}"
            unet.load_lora_adapter(str(ckpt_dir), prefix=None, adapter_name=adapter_name,
                                   weight_name="pytorch_lora_weights.safetensors")
            diag_scores = []
            for p, control_tensor, teacher_out in zip(diag_pairs, control_tensors, teacher_outputs):
                src_rgb = np.asarray(Image.open(p["aperio_path"]).convert("RGB"))
                unet.set_adapters(["colour", adapter_name])
                eval_pipe.scheduler = LCMScheduler.from_config(noise_scheduler.config)
                gen = torch.Generator(device=device).manual_seed(0)
                student_out = np.asarray(eval_pipe(
                    prompt=args.prompt, image=Image.fromarray(src_rgb), strength=0.70,
                    num_inference_steps=8, guidance_scale=1.0,
                    control_image=control_tensor, generator=gen).images[0])
                diag_scores.append(score_aligned_pair(student_out, teacher_out))
            keys = diag_scores[0].keys()
            means = {k: sum(d[k] for d in diag_scores) / len(diag_scores) for k in keys}
            sweep_results[tag] = {"per_pair": diag_scores, "mean": means}
            print(f"  {tag:>12s}  ssim={means['ssim']:.4f}  lab_total={means['lab_total']:.2f}")

        # Pareto frontier over (ssim higher-is-better, lab_total lower-is-
        # better) -- documented/reported for reference, but NOT itself the
        # selection rule (see below): a checkpoint is on the frontier if no
        # other swept checkpoint beats it on BOTH axes at once.
        def dominates(a, b):
            a_ssim, a_lab = sweep_results[a]["mean"]["ssim"], sweep_results[a]["mean"]["lab_total"]
            b_ssim, b_lab = sweep_results[b]["mean"]["ssim"], sweep_results[b]["mean"]["lab_total"]
            not_worse = a_ssim >= b_ssim and a_lab <= b_lab
            strictly_better = a_ssim > b_ssim or a_lab < b_lab
            return not_worse and strictly_better

        tags = list(sweep_results.keys())
        frontier = [t for t in tags if not any(dominates(o, t) for o in tags if o != t)]
        frontier.sort(key=lambda t: sweep_results[t]["mean"]["lab_total"])
        print(f"\nPareto frontier ({len(frontier)}/{len(tags)} checkpoints -- not dominated on "
              f"BOTH ssim and lab_total by any other swept checkpoint; reported for reference):")
        for t in frontier:
            print(f"  {t:>12s}  ssim={sweep_results[t]['mean']['ssim']:.4f}  "
                  f"lab_total={sweep_results[t]['mean']['lab_total']:.2f}")

        # Selection rule (P1-13, precommitted): among checkpoints within 5%
        # of the best SSIM in the sweep (a structural-fidelity floor -- never
        # trade away meaningful structure for colour recovery), pick the one
        # with the lowest lab_total. This is NOT the same as "lowest lab_total
        # on the Pareto frontier": a frontier point can still have collapsed
        # SSIM (e.g. this run's checkpoint-750, ssim=0.291, the best lab_total
        # in the whole sweep at 15.21, but excluded here because it falls
        # below the 5% floor) -- confirmed on a real run where the OLD rule
        # (lowest scalar distillation val_loss) selected one of the worst
        # image-quality checkpoints in the entire sweep, motivating this
        # image-level, floor-constrained rule in the first place. Never
        # select using held-out slides -- this uses only the internal
        # training-domain validation split.
        max_ssim = max(sweep_results[t]["mean"]["ssim"] for t in tags)
        ssim_floor = 0.95 * max_ssim
        eligible = [t for t in tags if sweep_results[t]["mean"]["ssim"] >= ssim_floor]
        best_tag = min(eligible, key=lambda t: sweep_results[t]["mean"]["lab_total"])
        best_src = dict(ckpt_candidates)[best_tag]
        best_dst = out_dir / "best_by_validation_b"
        if best_dst.exists():
            shutil.rmtree(best_dst)
        shutil.copytree(best_src, best_dst)
        print(f"\nSelection rule: lowest lab_total among checkpoints with ssim >= "
              f"0.95 x best_ssim ({ssim_floor:.4f}) -- eligible: {eligible}")
        print(f"Selected: '{best_tag}' (ssim={sweep_results[best_tag]['mean']['ssim']:.4f}, "
              f"lab_total={sweep_results[best_tag]['mean']['lab_total']:.2f}) -> copied to {best_dst}")

        with open(out_dir / "validation_b_teacher_match.json", "w") as fh:
            json.dump({"checkpoints": sweep_results, "pareto_frontier": frontier,
                       "ssim_floor": ssim_floor, "eligible_by_ssim_floor": eligible,
                       "selected": best_tag}, fh, indent=2)


if __name__ == "__main__":
    main()
