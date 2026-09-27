#!/usr/bin/env python3
"""
probe_pixcell_pipeline.py -- PXC-01a: the smallest possible wiring/smoke
check for PixCell before any LoRA training or held-out scoring code is
written. See tickets/PROBE-PIXCELL-TICKETS.md.

This is NOT part of the graded proposal scope -- see that ticket file's
own caveat. This script's only job is to answer, empirically, on this
cluster, with these exact cached weights: does the documented API
(DiffusionPipeline + trust_remote_code custom pipeline, UNI2-h via timm,
uni_embeds conditioning) actually work as documented? Several real
unknowns going in (all verified only from partial/summarised web
documentation, never a full primary-source code read):
  - the exact uni_embeds tensor shape PixCell-256 expects for a SINGLE
    256x256 image (known: (B, N, D) with D=1536; N=16 is confirmed for a
    1024x1024 image patchified into 16x256px tiles in PixCell's own
    virtual-staining code, but N for a native 256x256 input is NOT
    confirmed -- plausibly 1, unverified);
  - whether the whole pipeline needs to run in bf16 (SD3.5's own VAE is
    known from this project's own SD3.5-probe experience, PR-02, to be
    unstable in fp16 -- "switched to loading the whole pipeline, VAE
    included, uniformly in bf16" was that ticket's actual fix; applying
    the same lesson here pre-emptively rather than re-discovering it);
  - whether trust_remote_code's custom pipeline file needs live network
    access on every call, or caches after the first (matters for this
    project's usual HF_HUB_OFFLINE=1 convention).
No training, no LoRA, no held-out metrics here -- just prove the wiring,
print every real shape/dtype encountered, save one output image, and log
peak VRAM (same measurement discipline as the SD3.5 probe's PR-01).

Usage
-----
    python probe_pixcell_pipeline.py \
        --pixcell-repo StonyBrook-CVLab/PixCell-256 \
        --source-crop <pairs>/train/A03_00A_c000_aperio.png \
        --out <out_dir>

Dependencies: torch, diffusers, transformers, timm, Pillow, numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="PXC-01a: PixCell pipeline + UNI2-h wiring smoke check.")
    ap.add_argument("--pixcell-repo", default="StonyBrook-CVLab/PixCell-256")
    ap.add_argument("--pixcell-pipeline-repo", default="StonyBrook-CVLab/PixCell-pipeline")
    ap.add_argument("--sd35-vae-repo", default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--uni2h-repo", default="hf-hub:MahmoodLab/UNI2-h")
    ap.add_argument("--source-crop", required=True, help="One real RGB crop (any real Aperio PNG).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20, help="PixCell's own documented default.")
    ap.add_argument("--guidance-scale", type=float, default=1.5, help="PixCell's own documented default.")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--online", action="store_true",
                    help="Allow HF network access -- trust_remote_code's custom pipeline file "
                         "may need to fetch from the Hub on first use even with weights cached.")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("No CUDA device -- this script is meant for a GPU node.")
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"steps_completed": [], "shapes": {}, "errors": []}

    def checkpoint(name, **info):
        report["steps_completed"].append(name)
        if info:
            report["shapes"][name] = {k: str(v) for k, v in info.items()}
        print(f"[OK] {name}: {info}")

    # ------------------------------------------------------------------
    # 1. UNI2-h embedding model (timm) -- frozen, eval.
    # ------------------------------------------------------------------
    import timm
    print(f"timm version: {timm.__version__}")
    print(f"Loading UNI2-h ({args.uni2h_repo}) via timm ...")
    # UNI2-h's own documented timm_kwargs (its HF model card / GitHub) --
    # required explicitly: "hf-hub:" loading in this cluster's installed
    # timm version does NOT auto-populate these from the repo's own config,
    # and omitting them (confirmed empirically, job 61105) makes timm's
    # position-embedding resampler guess a mismatched grid size and crash
    # with "shape '[1, 15, 15, -1]' is invalid for input of size 391680" --
    # not a network/weights problem, purely a missing-kwargs one.
    uni2h_kwargs = dict(
        img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
        embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
        reg_tokens=8, dynamic_img_size=True,
    )
    uni_model = timm.create_model(args.uni2h_repo, pretrained=True, **uni2h_kwargs)
    uni_model.eval().to(device)
    uni_config = timm.data.resolve_data_config(uni_model.pretrained_cfg)
    uni_transform = timm.data.create_transform(**uni_config)
    checkpoint("uni2h_loaded", pretrained_cfg_input_size=uni_config.get("input_size"))

    # ------------------------------------------------------------------
    # 2. Extract the source crop's UNI2-h embedding, exact shape unknown
    #    for a native 256x256 input until this line actually runs.
    # ------------------------------------------------------------------
    src_img = Image.open(args.source_crop).convert("RGB")
    src_tensor = uni_transform(src_img).unsqueeze(0).to(device)
    with torch.inference_mode():
        raw_embed = uni_model(src_tensor)
    checkpoint("uni2h_embedding_extracted", input_shape=tuple(src_tensor.shape),
              raw_output_shape=tuple(raw_embed.shape), dtype=raw_embed.dtype)

    # ------------------------------------------------------------------
    # 3. Load the SD3.5 VAE (bf16 -- pre-emptively applying this
    #    project's own PR-02 lesson: SD3.5's VAE is unstable in fp16).
    # ------------------------------------------------------------------
    from diffusers import AutoencoderKL, DiffusionPipeline
    print(f"Loading SD3.5 VAE ({args.sd35_vae_repo}, bf16) ...")
    sd35_vae = AutoencoderKL.from_pretrained(
        args.sd35_vae_repo, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
    checkpoint("sd35_vae_loaded", latent_channels=sd35_vae.config.latent_channels,
              scaling_factor=sd35_vae.config.scaling_factor)

    # ------------------------------------------------------------------
    # 4. Load PixCell's custom diffusers pipeline (trust_remote_code).
    #    Whole pipeline in bf16, VAE included, matching PR-02's fix.
    # ------------------------------------------------------------------
    print(f"Loading PixCell pipeline ({args.pixcell_repo}, custom_pipeline="
          f"{args.pixcell_pipeline_repo}) ...")
    pipe = DiffusionPipeline.from_pretrained(
        args.pixcell_repo, vae=sd35_vae,
        custom_pipeline=args.pixcell_pipeline_repo,
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)
    checkpoint("pixcell_pipeline_loaded", transformer_dtype=next(pipe.transformer.parameters()).dtype)

    # Introspect the transformer's own expected conditioning shape BEFORE
    # guessing -- print whatever config attributes exist, so a shape
    # mismatch below is diagnosable from this project's own log, not a
    # bare stack trace.
    cfg = pipe.transformer.config
    cfg_dict = dict(cfg) if hasattr(cfg, "keys") else vars(cfg)
    interesting = {k: v for k, v in cfg_dict.items()
                   if "token" in k.lower() or "caption" in k.lower() or "cross_attention" in k.lower()}
    checkpoint("pixcell_transformer_config_inspected", relevant_fields=interesting)

    # ------------------------------------------------------------------
    # 5. Shape the embedding for the pipeline's uni_embeds argument.
    #    Try the most likely shape (B, 1, 1536) first; if the pipeline
    #    rejects it, the raised error should name the expected N -- catch
    #    and report rather than guess further blindly.
    # ------------------------------------------------------------------
    uni_embeds = raw_embed.unsqueeze(1).to(torch.bfloat16) if raw_embed.dim() == 2 else raw_embed.to(torch.bfloat16)
    checkpoint("uni_embeds_shaped", shape=tuple(uni_embeds.shape))

    # guidance_scale > 1.0 (PixCell's own documented default, 1.5) requires
    # a negative_uni_embeds for classifier-free guidance -- confirmed via
    # direct read of the cached pipeline.py's check_inputs()/get_
    # unconditional_embedding() (a learned embedding on the transformer's
    # own caption_projection module, not something to construct by hand).
    negative_uni_embeds = pipe.get_unconditional_embedding(batch_size=uni_embeds.shape[0]).to(
        device=device, dtype=torch.bfloat16)
    checkpoint("negative_uni_embeds_obtained", shape=tuple(negative_uni_embeds.shape))

    # ------------------------------------------------------------------
    # 6. One generation call. This is the actual pass/fail gate.
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        result = pipe(
            uni_embeds=uni_embeds,
            negative_uni_embeds=negative_uni_embeds,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.resolution, width=args.resolution,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )
        image = result.images[0]
        elapsed = time.time() - t0
        checkpoint("generation_completed", elapsed_sec=round(elapsed, 2), output_size=image.size)
        image.save(out_dir / "probe_output.png")
        src_img.resize(image.size).save(out_dir / "probe_source_input.png")
        print(f"Saved output -> {out_dir / 'probe_output.png'}")
    except Exception as e:
        report["errors"].append({"step": "generation_call", "error": str(e),
                                 "traceback": traceback.format_exc()})
        print(f"[FAIL] generation_call: {e}")
        print(traceback.format_exc())

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    report["peak_vram_gb"] = round(peak_vram_gb, 2)
    print(f"\nPeak VRAM: {peak_vram_gb:.2f} GB")

    with open(out_dir / "probe_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"Report written -> {out_dir / 'probe_report.json'}")
    if report["errors"]:
        raise SystemExit(f"Probe completed with {len(report['errors'])} error(s) -- see probe_report.json.")
    print("\nAll steps completed cleanly.")


if __name__ == "__main__":
    main()
