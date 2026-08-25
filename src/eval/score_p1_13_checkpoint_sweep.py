#!/usr/bin/env python3
"""
score_p1_13_checkpoint_sweep.py -- P1-13 inference-only diagnostic: for
every saved checkpoint of a training run x a small student guidance_scale
sweep, score against BOTH the frozen P1-10 teacher (teacher-fidelity, what
Validation B already measured) AND the REAL paired Hamamatsu ground truth
(real-H, actual scanner-normalisation quality) -- on the run's own internal
training-domain validation split only, never held-out slides. No new
training. See tickets/PHASE1-TICKETS.md P1-13.

Motivation (2026-08-25): Control 1 found the task-specific adapter losing
to the generic pretrained LCM-LoRA at guidance_scale=2.0. But
train_p1_10_lcm_lora.py's `w` uses the LCM PAPER's CFG form for the teacher
target (pred = cond + w*(cond-uncond)), not diffusers' ordinary
guidance_scale (G) form (uncond + G*(cond-uncond)) -- equating the two
gives G = w + 1, so this run's w~U[1.0,2.0] training range corresponds to
teacher trajectories at G~U[2.0,3.0], not an inference guidance_scale range
of 1.0-2.0. The student's own training-time forward pass is also a single
CONDITIONAL pass with no external CFG at all, so evaluating it at
guidance_scale=2.0 (which diffusers' do_classifier_free_guidance =
guidance_scale > 1 turns into a doubled cond/uncond pass) may be testing an
operating point the adapter was never calibrated for. This script sweeps
guidance so that hypothesis can be ruled in or out with real numbers before
Control 1's result is accepted as final.

Teacher-fidelity and real-H metrics are reported SEPARATELY and are
expected to diverge (P1-13's own controls already showed matching the
teacher well does not imply matching the real target well) -- real-H is
the metric that matters for the final selection here, teacher-fidelity is
reported as a diagnostic only.

Loads the teacher stack (frozen base UNet + frozen ControlNet + frozen
"colour" LoRA) ONCE and generates each pair's 50-step teacher reference
ONCE per pair (reused across every checkpoint x guidance combo, since the
teacher is fixed). Each checkpoint's "lcm" adapter is loaded under its own
unique adapter name so all checkpoints share the same loaded base model --
no repeated from_pretrained() calls; only guidance_scale/scheduler state
change between student generations.

Val-pair reconstruction: EITHER --overfit-n (legacy 8-pair diagnostic mode,
build_pairs()[:N]) OR --val-frames-json (path to the training run's own
training_config.json, whose "val_frames" list is read directly -- the
same pairs train_p1_10_lcm_lora.py validated on, reconstructed by reading
its own recorded ground truth rather than re-deriving the seed/shuffle
logic a second time).

Usage
-----
    python score_p1_13_checkpoint_sweep.py \
        --pairs-dir <data>/pairs/train \
        --val-frames-json <data>/lora/a2h_cond_r8_lcm_distilled_lr0.00003_ga4/training_config.json \
        --teacher-dir <data>/lora/a2h_cond_r8/best \
        --run-dir <data>/lora/a2h_cond_r8_lcm_distilled_lr0.00003_ga4 \
        --student-guidances 1.0 1.5 2.0 \
        --out <data>/lora/a2h_cond_r8_lcm_distilled_lr0.00003_ga4/guidance_sweep.json

Dependencies: torch, diffusers, transformers, peft, numpy, Pillow,
opencv-python-headless, scikit-image. Requires canny.py/metrics.py
(src/eval/, this file's own directory) and train_colour_translation_lora.py
(src/train/, sibling directory) for build_pairs().
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))


def parse_args():
    ap = argparse.ArgumentParser(
        description="P1-13: guidance sweep, teacher-fidelity vs real-H, every saved checkpoint.")
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--overfit-n", type=int, default=0,
                    help="Legacy 8-pair diagnostic mode: build_pairs()[:N]. Mutually exclusive "
                         "with --val-frames-json.")
    ap.add_argument("--val-frames-json", default=None,
                    help="Path to the training run's own training_config.json -- its recorded "
                         "'val_frames' list is used to select pairs (frame_id in that list), "
                         "reproducing the exact internal validation split that run used. "
                         "Mutually exclusive with --overfit-n.")
    ap.add_argument("--teacher-dir", required=True, help="e.g. lora/a2h_cond_r8/best")
    ap.add_argument("--run-dir", required=True,
                    help="Training output dir containing checkpoint-N/ and final/ (and best/ if "
                         "present), e.g. lora/a2h_cond_r8_lcm_distilled_lr0.00003_ga4")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher-steps", type=int, default=50)
    ap.add_argument("--teacher-strength", type=float, default=0.50)
    ap.add_argument("--teacher-guidance", type=float, default=2.0)
    ap.add_argument("--student-steps", type=int, default=8)
    ap.add_argument("--student-strength", type=float, default=0.70)
    ap.add_argument("--student-guidances", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--ssim-floor-frac", type=float, default=0.95,
                    help="Selection rule: eligible = real-H ssim >= this fraction of the best "
                         "real-H ssim observed; selected = lowest real-H lab_total among eligible.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--online", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if bool(args.overfit_n) == bool(args.val_frames_json):
        raise SystemExit("Exactly one of --overfit-n or --val-frames-json must be given.")

    import os
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    from PIL import Image
    from diffusers import (AutoencoderKL, ControlNetModel, DDIMScheduler, DDPMScheduler,
                           LCMScheduler, StableDiffusionControlNetImg2ImgPipeline,
                           UNet2DConditionModel)
    from transformers import CLIPTextModel, CLIPTokenizer

    from canny import extract_canny_control_image
    from metrics import score_aligned_pair
    from train_colour_translation_lora import build_pairs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("No CUDA device -- this script is meant for a GPU node.")

    all_pairs = build_pairs(args.pairs_dir)
    if args.overfit_n:
        pairs = all_pairs[: args.overfit_n]
        print(f"Scoring against {len(pairs)} pairs (legacy --overfit-n={args.overfit_n}).")
    else:
        with open(args.val_frames_json) as fh:
            val_frames = set(json.load(fh)["val_frames"])
        pairs = [p for p in all_pairs if p["frame_id"] in val_frames]
        print(f"Scoring against {len(pairs)} pairs reconstructed from "
              f"{args.val_frames_json}'s val_frames={sorted(val_frames)}.")
    if not pairs:
        raise SystemExit("No pairs selected -- check --pairs-dir / --val-frames-json.")

    teacher_dir = Path(args.teacher_dir)
    run_dir = Path(args.run_dir)

    # Discover checkpoints in a stable, numerically-sorted order.
    ckpt_dirs = []
    for d in sorted(run_dir.glob("checkpoint-*")):
        m = re.match(r"checkpoint-(\d+)$", d.name)
        if m and (d / "pytorch_lora_weights.safetensors").exists():
            ckpt_dirs.append((int(m.group(1)), d))
    ckpt_dirs.sort(key=lambda t: t[0])
    tagged = [(str(n), d) for n, d in ckpt_dirs]
    if (run_dir / "best").exists() and (run_dir / "best" / "pytorch_lora_weights.safetensors").exists():
        tagged.append(("best", run_dir / "best"))
    if (run_dir / "final" / "pytorch_lora_weights.safetensors").exists():
        tagged.append(("final", run_dir / "final"))
    if not tagged:
        raise SystemExit(f"No checkpoints with pytorch_lora_weights.safetensors found under {run_dir}")
    print(f"Found {len(tagged)} checkpoints x {len(args.student_guidances)} guidances = "
          f"{len(tagged) * len(args.student_guidances)} combos to score: "
          f"{[t for t, _ in tagged]}  x  {args.student_guidances}")

    print(f"Loading SD 1.5 components from cache ({args.model}) ...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")

    print(f"Loading frozen P1-10 ControlNet from {teacher_dir / 'controlnet'} ...")
    controlnet = ControlNetModel.from_pretrained(str(teacher_dir / "controlnet"))

    print(f"Loading frozen P1-10 colour LoRA from {teacher_dir} as adapter 'colour' ...")
    unet.load_lora_adapter(str(teacher_dir), prefix="unet", adapter_name="colour",
                           weight_name="pytorch_lora_weights.safetensors")

    adapter_names = []
    for tag, ckpt_dir in tagged:
        name = f"lcm_{tag}"
        unet.load_lora_adapter(str(ckpt_dir), prefix=None, adapter_name=name,
                               weight_name="pytorch_lora_weights.safetensors")
        adapter_names.append((tag, name))
    print(f"Loaded {len(adapter_names)} 'lcm' checkpoint adapters alongside frozen 'colour'.")

    for m in [vae, text_encoder, unet, controlnet]:
        m.to(device)
    vae.eval(); text_encoder.eval(); unet.eval(); controlnet.eval()

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

    prompt = "H&E stained histopathology tissue"

    # Generate each pair's teacher output ONCE and load the real Hamamatsu
    # ground truth ONCE -- both reused across every checkpoint x guidance
    # combo, since neither depends on either axis. Real-H is the raw paired
    # crop, unregistered -- consistent with how these training-domain pairs
    # are used everywhere else in this project (coordinate-corresponding,
    # not pixel-exact; registration is a held-out-only step, per CLAUDE.md).
    print("Generating teacher (50-step DDIM) reference + loading real-H ground truth, once per pair ...")
    teacher_outputs, real_h_images, control_tensors = [], [], []
    for p in pairs:
        src_rgb = np.asarray(Image.open(p["aperio_path"]).convert("RGB"))
        real_h = np.asarray(Image.open(p["hamamatsu_path"]).convert("RGB"))
        real_h_images.append(real_h)
        control_tensor = build_control_tensor(src_rgb)
        control_tensors.append(control_tensor)

        unet.set_adapters(["colour"])
        eval_pipe.scheduler = DDIMScheduler.from_config(noise_scheduler.config)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        teacher_out = np.asarray(eval_pipe(
            prompt=prompt, image=Image.fromarray(src_rgb), strength=args.teacher_strength,
            num_inference_steps=args.teacher_steps, guidance_scale=args.teacher_guidance,
            control_image=control_tensor, generator=gen).images[0])
        teacher_outputs.append(teacher_out)

    # combo key: "<tag>@g<guidance>"
    results = {}
    for tag, name in adapter_names:
        for guidance in args.student_guidances:
            combo = f"{tag}@g{guidance}"
            print(f"Scoring {combo} ...")
            tf_scores, rh_scores = [], []
            for p, control_tensor, teacher_out, real_h in zip(
                    pairs, control_tensors, teacher_outputs, real_h_images):
                src_rgb = np.asarray(Image.open(p["aperio_path"]).convert("RGB"))
                unet.set_adapters(["colour", name])
                eval_pipe.scheduler = LCMScheduler.from_config(noise_scheduler.config)
                gen = torch.Generator(device=device).manual_seed(args.seed)
                student_out = np.asarray(eval_pipe(
                    prompt=prompt, image=Image.fromarray(src_rgb), strength=args.student_strength,
                    num_inference_steps=args.student_steps, guidance_scale=guidance,
                    control_image=control_tensor, generator=gen).images[0])
                tf_scores.append(score_aligned_pair(student_out, teacher_out))
                rh_scores.append(score_aligned_pair(student_out, real_h))
            keys = tf_scores[0].keys()
            tf_mean = {k: sum(d[k] for d in tf_scores) / len(tf_scores) for k in keys}
            rh_mean = {k: sum(d[k] for d in rh_scores) / len(rh_scores) for k in keys}
            results[combo] = {"tag": tag, "guidance": guidance,
                              "teacher_fidelity": tf_mean, "real_h": rh_mean}
            print(f"  teacher_fidelity: ssim={tf_mean['ssim']:.4f} lab_total={tf_mean['lab_total']:.2f}  |  "
                  f"real_h: ssim={rh_mean['ssim']:.4f} lab_total={rh_mean['lab_total']:.2f}")

    # ------------------------------------------------------------------
    # Pareto frontiers (real-H is the one that matters for selection;
    # teacher-fidelity frontier reported for reference/diagnostic only).
    # ------------------------------------------------------------------
    def pareto_frontier(combos, metric_key):
        def dominates(a, b):
            a_s, a_l = results[a][metric_key]["ssim"], results[a][metric_key]["lab_total"]
            b_s, b_l = results[b][metric_key]["ssim"], results[b][metric_key]["lab_total"]
            not_worse = a_s >= b_s and a_l <= b_l
            strictly_better = a_s > b_s or a_l < b_l
            return not_worse and strictly_better
        frontier = [c for c in combos if not any(dominates(o, c) for o in combos if o != c)]
        frontier.sort(key=lambda c: results[c][metric_key]["lab_total"])
        return frontier

    def floor_select(combos, metric_key):
        max_ssim = max(results[c][metric_key]["ssim"] for c in combos)
        floor = args.ssim_floor_frac * max_ssim
        eligible = [c for c in combos if results[c][metric_key]["ssim"] >= floor]
        best = min(eligible, key=lambda c: results[c][metric_key]["lab_total"])
        return best, floor, eligible

    all_combos = list(results.keys())
    real_h_frontier = pareto_frontier(all_combos, "real_h")
    teacher_frontier = pareto_frontier(all_combos, "teacher_fidelity")
    selected, floor, eligible = floor_select(all_combos, "real_h")

    print(f"\n=== Real-H Pareto frontier ({len(real_h_frontier)}/{len(all_combos)}) ===")
    for c in real_h_frontier:
        m = results[c]["real_h"]
        print(f"  {c:>18s}  ssim={m['ssim']:.4f}  lab_total={m['lab_total']:.2f}")

    print(f"\n=== Teacher-fidelity Pareto frontier ({len(teacher_frontier)}/{len(all_combos)}, reference only) ===")
    for c in teacher_frontier:
        m = results[c]["teacher_fidelity"]
        print(f"  {c:>18s}  ssim={m['ssim']:.4f}  lab_total={m['lab_total']:.2f}")

    print(f"\nSelection rule: real-H ssim >= {args.ssim_floor_frac} x best real-H ssim ({floor:.4f}) "
          f"-> eligible: {eligible}")
    print(f"Selected: '{selected}' real_h=(ssim={results[selected]['real_h']['ssim']:.4f}, "
          f"lab_total={results[selected]['real_h']['lab_total']:.2f})  "
          f"teacher_fidelity=(ssim={results[selected]['teacher_fidelity']['ssim']:.4f}, "
          f"lab_total={results[selected]['teacher_fidelity']['lab_total']:.2f}) "
          f"[teacher-fidelity reported as a diagnostic only, not the selection criterion]")

    # Best configuration at each guidance value (same floor+lowest-lab_total
    # rule, restricted to that guidance's own combos) -- shows whether
    # guidance=1.0 materially changes the achievable result.
    print("\n=== Best configuration per guidance value (real-H, same selection rule) ===")
    per_guidance_best = {}
    for g in args.student_guidances:
        g_combos = [c for c in all_combos if results[c]["guidance"] == g]
        g_best, g_floor, g_eligible = floor_select(g_combos, "real_h")
        per_guidance_best[g] = g_best
        m = results[g_best]["real_h"]
        print(f"  guidance={g}: best='{g_best}'  real_h ssim={m['ssim']:.4f}  lab_total={m['lab_total']:.2f}")

    with open(args.out, "w") as fh:
        json.dump({"combos": results, "real_h_pareto_frontier": real_h_frontier,
                   "teacher_fidelity_pareto_frontier": teacher_frontier,
                   "ssim_floor_frac": args.ssim_floor_frac, "ssim_floor_value": floor,
                   "eligible_by_ssim_floor": eligible, "selected": selected,
                   "per_guidance_best": {str(k): v for k, v in per_guidance_best.items()}},
                  fh, indent=2)
    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()
