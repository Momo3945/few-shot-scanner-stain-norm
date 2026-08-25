#!/usr/bin/env python3
"""
infer_p1_13_lcm_controls.py -- P1-13's mandatory post-training controls:
inference for the task-specific LCM-LoRA (checkpoint-625, selected via the
5%-SSIM-floor rule -- see tickets/PHASE1-TICKETS.md P1-13) composed with
the frozen P1-10 colour LoRA + ControlNet.

Covers, in one script (Control 4 -- few-step DDIM reference -- already
exists from the earlier diagnostic, job 45654/45700, and needs no new code):

  --variant task_specific   Control 2 (source ablation) when combined with
                            --source-mode {correct,zero,shuffled}: the
                            selected P1-13 "lcm" adapter + frozen "colour",
                            LCMScheduler, few steps.
  --variant adapter_disabled  Control 3 (adapter isolation): "lcm" IS loaded
                            (same checkpoint as task_specific) but excluded
                            from set_adapters -- tests DISABLING, not mere
                            absence. DDIMScheduler at full steps -- must
                            reproduce ordinary P1-10 behaviour
                            (infer_colour_translation.py's own defaults),
                            confirming the adapter composition doesn't leak
                            into the baseline path.
  --variant generic_lcm     Control 1 comparison arm: the official pretrained
                            latent-consistency/lcm-lora-sdv1-5 + "colour",
                            LCMScheduler, same few-step settings as
                            task_specific -- for a fair, matched comparison.

Deliberately a NEW script, not an edit to infer_colour_translation.py (that
script is validated, in-use code for P1-10/P1-11 elsewhere) -- crop
selection / registration / source-mode-ablation construction below is
adapted from it (P1-10's ticket precedent: each new ticket gets its own
script, never edits a prior validated one).

Manifest schema matches infer_colour_translation.py's exactly (seed,
source_mode, crop_id, slide, frame, x, y, output_path, reference_path,
aperio_path), so the EXISTING score_p1_10_ablation.py scores it unchanged --
no new scorer needed. The "source_mode" column is written as
"<variant>_<source_mode>" (e.g. "task_specific_correct",
"adapter_disabled_correct", "generic_lcm_correct") so every control arm gets
its own bucket in that scorer's per-mode aggregation and paired win-rate,
even though several arms are physically separate --out directories /
separate invocations of this script.

Usage
-----
    # Control 2 -- source ablation (task-specific adapter), 3 separate runs:
    python infer_p1_13_lcm_controls.py --variant task_specific --source-mode correct \
        --lcm-adapter <out>/best_by_validation_b --root <data>/mitos_heldout \
        --heldout <data>/pairs/heldout_frames.csv --out <data>/eval/p1_13_ctrl_task_specific_correct --limit 20
    python infer_p1_13_lcm_controls.py --variant task_specific --source-mode zero     ... --out .../p1_13_ctrl_task_specific_zero
    python infer_p1_13_lcm_controls.py --variant task_specific --source-mode shuffled ... --out .../p1_13_ctrl_task_specific_shuffled

    # Control 3 -- adapter isolation ("lcm" loaded but excluded from set_adapters):
    python infer_p1_13_lcm_controls.py --variant adapter_disabled --source-mode correct \
        --lcm-adapter <out>/best_by_validation_b --root <data>/mitos_heldout \
        --heldout <data>/pairs/heldout_frames.csv --out <data>/eval/p1_13_ctrl_adapter_disabled --limit 20

    # Control 1 comparison arm -- generic pretrained LCM-LoRA:
    python infer_p1_13_lcm_controls.py --variant generic_lcm --source-mode correct \
        --root <data>/mitos_heldout --heldout <data>/pairs/heldout_frames.csv \
        --out <data>/eval/p1_13_ctrl_generic_lcm --limit 20

    # Score everything together (reuses the EXISTING, unmodified scorer):
    python score_p1_10_ablation.py --eval-dirs <data>/eval/p1_13_ctrl_* --out <data>/eval/p1_13_ctrl_summary

Dependencies: torch, diffusers, numpy, Pillow, opencv-python-headless,
tifffile. Requires registration.py and canny.py on the path (same folder).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="P1-13: post-training controls for the task-specific LCM-LoRA.")
    ap.add_argument("--variant", choices=["task_specific", "adapter_disabled", "generic_lcm"],
                    required=True)
    ap.add_argument("--lora", default=None,
                    help="P1-10 colour LoRA checkpoint dir (e.g. lora/a2h_cond_r8/best). "
                         "Required for every variant -- 'colour' is always active.")
    ap.add_argument("--controlnet", default=None,
                    help="P1-10 ControlNet dir (e.g. lora/a2h_cond_r8/best/controlnet). "
                         "Defaults to <lora>/controlnet if omitted.")
    ap.add_argument("--lcm-adapter", default=None,
                    help="Task-specific P1-13 'lcm' adapter checkpoint dir (e.g. "
                         "lora/a2h_cond_r8_lcm_distilled_lr0.00003_ga4/best_by_validation_b). "
                         "Required for --variant task_specific (active) and adapter_disabled "
                         "(loaded but excluded, to test disabling not absence). Ignored for "
                         "generic_lcm.")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--root", default=None)
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--pairs-dir", default=None,
                    help="Run against training pairs instead of held-out frames. Mutually "
                         "exclusive with --root/--heldout.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--direction", choices=["A2H", "H2A"], default="A2H")
    ap.add_argument("--source-mode", choices=["correct", "zero", "shuffled"], default="correct",
                    help="Control 2's source-conditioning ablation. Only meaningfully varied "
                         "for --variant task_specific; other variants typically run 'correct' "
                         "only, as a fixed reference point.")
    ap.add_argument("--steps", type=int, default=None,
                    help="Defaults: 8 for task_specific/generic_lcm (LCM few-step), 50 for "
                         "adapter_disabled (ordinary P1-10 DDIM quality path).")
    ap.add_argument("--strength", type=float, default=None,
                    help="Defaults: 0.70 for task_specific/generic_lcm, 0.50 for "
                         "adapter_disabled (matches infer_colour_translation.py's own default).")
    ap.add_argument("--guidance", type=float, default=None,
                    help="Defaults: 2.0 for adapter_disabled (ordinary DDIM-quality CFG), "
                         "1.0 for task_specific/generic_lcm (no external CFG -- the LCM "
                         "student/adapter is never trained to expect diffusers' internal "
                         "cond/uncond doubling that guidance_scale > 1.0 triggers; see "
                         "tickets/PHASE1-TICKETS.md P1-13 guidance-semantics mismatch note).")
    ap.add_argument("--prompt", default="H&E stained histopathology tissue")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tissue-thresh", type=float, default=0.30)
    ap.add_argument("--ecc-min", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-crops-per-frame", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--online", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.lora:
        raise SystemExit("--lora is required for every --variant.")
    if args.variant in ("task_specific", "adapter_disabled") and not args.lcm_adapter:
        raise SystemExit(
            "--lcm-adapter is required for --variant task_specific AND for "
            "adapter_disabled -- Control 3 (adapter isolation) needs the 'lcm' adapter "
            "actually LOADED and then explicitly disabled via set_adapters, not merely "
            "never attached, to test disabling rather than absence.")
    if args.steps is None:
        args.steps = 50 if args.variant == "adapter_disabled" else 8
    if args.strength is None:
        args.strength = 0.50 if args.variant == "adapter_disabled" else 0.70
    if args.guidance is None:
        args.guidance = 2.0 if args.variant == "adapter_disabled" else 1.0
    controlnet_dir = args.controlnet or str(Path(args.lora) / "controlnet")

    using_pairs_dir = bool(args.pairs_dir)
    using_heldout = bool(args.root or args.heldout)
    if using_pairs_dir and using_heldout:
        raise SystemExit("--pairs-dir is mutually exclusive with --root/--heldout.")
    if not using_pairs_dir and not (args.root and args.heldout):
        raise SystemExit("Either --pairs-dir, or both --root and --heldout, must be given.")
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    from PIL import Image

    from canny import extract_canny_control_image
    if using_pairs_dir:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
        from train_colour_translation_lora import build_pairs
    else:
        from registration import read_rgb, register_h_to_a

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("No CUDA device -- inference must run on a GPU node.")

    out_dir = Path(args.out)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)

    # ---- tissue + crop helpers, identical convention to infer_colour_translation.py ----
    def tissue_fraction(rgb):
        a = rgb.astype(np.int16); mx = a.max(2); mn = a.min(2)
        return float(((mx < 235) & ((mx - mn) > 12)).mean())

    def grid_offsets(length, crop):
        if length <= crop:
            return [0]
        offs = list(range(0, length - crop + 1, crop))
        if offs[-1] != length - crop:
            offs.append(length - crop)
        return sorted(set(offs))

    crops = []
    if using_pairs_dir:
        pairs = build_pairs(args.pairs_dir)
        if args.limit:
            pairs = pairs[: args.limit]
        src_key, ref_key = ("aperio_path", "hamamatsu_path") if args.direction == "A2H" \
            else ("hamamatsu_path", "aperio_path")
        for p in pairs:
            src_c = np.asarray(Image.open(p[src_key]).convert("RGB"))
            ref_c = np.asarray(Image.open(p[ref_key]).convert("RGB"))
            crops.append({"slide": p["slide"], "frame": p["frame_id"], "x": 0, "y": 0,
                         "tag_id": p["pair_id"], "src": src_c, "ref": ref_c,
                         "aperio_path": p["aperio_path"]})
        print(f"  {args.pairs_dir}: {len(crops)} training pairs")
    else:
        with open(args.heldout, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if args.limit:
            rows = rows[: args.limit]
        for r in rows:
            r["aperio_path"] = r["aperio_path"].replace("\\", "/")
            r["hamamatsu_path"] = r["hamamatsu_path"].replace("\\", "/")

        for r in rows:
            a_rgb = read_rgb(Path(args.root) / r["aperio_path"])
            h_rgb = read_rgb(Path(args.root) / r["hamamatsu_path"])
            reg = register_h_to_a(a_rgb, h_rgb, ecc_min=args.ecc_min)
            h_reg = reg.h_registered_rgb
            H, W = a_rgb.shape[:2]
            src_frame, ref_frame = (a_rgb, h_reg) if args.direction == "A2H" else (h_reg, a_rgb)

            crops_done = 0
            for y in grid_offsets(H, args.crop):
                for x in grid_offsets(W, args.crop):
                    if args.max_crops_per_frame and crops_done >= args.max_crops_per_frame:
                        break
                    src_c = src_frame[y:y + args.crop, x:x + args.crop]
                    ref_c = ref_frame[y:y + args.crop, x:x + args.crop]
                    if (ref_c.max(2) < 6).mean() > 0.10:
                        continue
                    if tissue_fraction(src_c) < args.tissue_thresh:
                        continue
                    crops.append({"slide": r["aperio_slide"], "frame": r["frame_id"],
                                 "x": x, "y": y,
                                 "tag_id": f"{r['aperio_slide']}_{r['frame_id']}_x{x}_y{y}",
                                 "src": src_c, "ref": ref_c, "aperio_path": r["aperio_path"]})
                    crops_done += 1
            print(f"  {r['aperio_slide']}_{r['frame_id']}: {crops_done} crops")
    if not crops:
        raise SystemExit("No tissue crops selected -- check --tissue-thresh / heldout / pairs-dir inventory.")

    rng = np.random.default_rng(0)
    shuffled_idx = rng.permutation(len(crops))
    for i in range(len(crops)):
        if shuffled_idx[i] == i:
            j = (i + 1) % len(crops)
            shuffled_idx[i], shuffled_idx[j] = shuffled_idx[j], shuffled_idx[i]

    def control_source_for(i):
        if args.source_mode == "correct":
            return crops[i]["src"]
        if args.source_mode == "zero":
            return None
        return crops[shuffled_idx[i]]["src"]

    def build_control_tensor(rgb):
        canny = extract_canny_control_image(rgb)
        rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        canny_t = torch.from_numpy(canny.astype(np.float32) / 255.0).permute(2, 0, 1)
        return torch.cat([rgb_t, canny_t], dim=0).unsqueeze(0)

    def seed_for(pair_key: str, seed: int) -> int:
        h = hashlib.sha256(f"{pair_key}|{seed}".encode()).hexdigest()
        return int(h[:8], 16)

    # ------------------------------------------------------------------
    # Pipeline: "colour" always active; "lcm"/"lcm_generic" added on top
    # depending on --variant. Scheduler follows the variant (DDIM for
    # adapter_disabled, LCM otherwise) -- same multi-adapter + scheduler-swap
    # pattern already validated in infer_colour_lora.py.
    # ------------------------------------------------------------------
    from diffusers import (ControlNetModel, DDIMScheduler, LCMScheduler,
                           StableDiffusionControlNetImg2ImgPipeline)
    print(f"Loading P1-10 pipeline: LoRA={args.lora}  ControlNet={controlnet_dir}  "
          f"variant={args.variant}  source_mode={args.source_mode} ...")
    controlnet = ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.model, controlnet=controlnet, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False)
    pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors",
                           adapter_name="colour")

    if args.variant == "adapter_disabled":
        # Load "lcm" (same as task_specific) but then explicitly leave it OUT
        # of set_adapters -- tests that DISABLING the adapter reproduces
        # ordinary P1-10 behaviour, not merely that it's absent.
        pipe.unet.load_lora_adapter(args.lcm_adapter, prefix=None, adapter_name="lcm",
                                    weight_name="pytorch_lora_weights.safetensors")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_adapters(["colour"])
    elif args.variant == "task_specific":
        # NOT pipe.load_lora_weights() -- this checkpoint was saved via
        # unet.save_lora_adapter() (train_p1_10_lcm_lora.py), which writes
        # native PEFT-format keys (no "unet." prefix, "lora_A"/"lora_B"
        # naming), not the diffusers-prefixed format pipe.load_lora_weights()
        # expects. Load directly on pipe.unet, matching the exact call
        # already proven working in that training script's own Validation B.
        pipe.unet.load_lora_adapter(args.lcm_adapter, prefix=None, adapter_name="lcm",
                                    weight_name="pytorch_lora_weights.safetensors")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.set_adapters(["colour", "lcm"])
    else:  # generic_lcm
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5",
                               weight_name="pytorch_lora_weights.safetensors",
                               adapter_name="lcm_generic")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.set_adapters(["colour", "lcm_generic"])

    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    def run_crop(target_src_rgb, control_src_rgb, seed):
        control_tensor = (torch.zeros(1, 6, args.crop, args.crop, dtype=torch.float32)
                          if control_src_rgb is None else build_control_tensor(control_src_rgb))
        gen = torch.Generator(device=device).manual_seed(seed)
        out = pipe(prompt=args.prompt, image=Image.fromarray(target_src_rgb), strength=args.strength,
                   num_inference_steps=args.steps, guidance_scale=args.guidance,
                   control_image=control_tensor, generator=gen).images[0]
        return np.asarray(out)

    # ------------------------------------------------------------------
    man_path = out_dir / "eval_manifest.csv"
    man = open(man_path, "w", newline="")
    mw = csv.writer(man)
    mw.writerow(["seed", "source_mode", "crop_id", "slide", "frame", "x", "y",
                 "output_path", "reference_path", "aperio_path"])

    manifest_mode = f"{args.variant}_{args.source_mode}"
    n_out = 0
    for i, c in enumerate(crops):
        tag = c["tag_id"]
        ref_path = out_dir / "reference" / f"{tag}.png"
        if not ref_path.exists():
            Image.fromarray(c["ref"]).save(ref_path)

        for s in args.seeds:
            seed_val = seed_for(tag, s)
            out = run_crop(c["src"], control_source_for(i), seed_val)
            out_path = out_dir / "outputs" / f"{tag}_{manifest_mode}_seed{s}.png"
            Image.fromarray(out).save(out_path)
            mw.writerow([s, manifest_mode, tag, c["slide"], c["frame"], c["x"], c["y"],
                        os.path.relpath(out_path, out_dir), os.path.relpath(ref_path, out_dir),
                        c["aperio_path"]])
            n_out += 1

    man.close()
    print(f"\nWrote {n_out} output crops across {len(crops)} locations x {len(args.seeds)} seed(s), "
          f"variant={args.variant}, source_mode={args.source_mode} (manifest mode='{manifest_mode}').")
    print(f"Manifest: {man_path}")
    print("Next: score_p1_10_ablation.py --eval-dirs <this dir + others> --out <summary dir>")


if __name__ == "__main__":
    main()
