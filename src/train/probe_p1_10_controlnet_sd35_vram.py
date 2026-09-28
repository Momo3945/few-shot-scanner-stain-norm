#!/usr/bin/env python3
"""
probe_p1_10_controlnet_sd35_vram.py -- feasibility check: does a P1-10-style
source-conditioning ControlNet + colour LoRA fit on SD3.5's transformer within
bigbatch's 24GB RTX 3090?

NOT a trainer -- no data loading, no real training loop, no text encoders.
Builds the frozen transformer (bf16, matching train_colour_lora_sd35.py's
established pattern), clones a fresh SD3ControlNetModel via from_transformer()
-- the direct MMDiT analog to the UNet-based ControlNetModel.from_unet() that
P1-10's train_colour_translation_lora.py actually uses for its source-
conditioning branch -- adds a colour LoRA to the transformer, freezes
everything in the ControlNet clone except its zero-initialised new parts
(pos_embed_input, controlnet_blocks -- mirroring P1-10's own
TRAINABLE_CONTROLNET_PREFIXES convention for the UNet ControlNet), then runs
ONE dummy forward+backward step with random tensors at the correct real
shapes/dtypes and reports peak VRAM via torch.cuda.max_memory_allocated().
Random tensors (not real text-encoder output / real VAE-encoded images) are
deliberate: PR-01 already proved text encoders can be freed before the heavy
part of training, so the open question this probe answers is specifically
whether transformer + ControlNet + LoRA + activations + optimizer state fit
together -- loading real text encoders would only add irrelevant VRAM noise
to that question.

Real transformer config (verified from the cached config.json):
num_layers=38, num_attention_heads=38, attention_head_dim=64 (inner_dim=2432),
joint_attention_dim=4096, pooled_projection_dim=2048. A 12-layer ControlNet
clone is therefore ~12/38 ~= 31% of the transformer's own block count -- NOT
negligible, unlike SD1.5/SDXL's much smaller UNets where P1-10/P3-05's
ControlNet additions were a minor VRAM fraction. This is exactly why this is
measured empirically rather than assumed.

Conditioning design: SD3's own ControlNet pipeline VAE-encodes its control
image into the SAME 16-channel latent space (confirmed from
pipeline_stable_diffusion_3_controlnet.py: `control_image =
self.vae.encode(control_image).latent_dist.sample()`) -- NOT a dedicated
pixel-space conv encoder like the UNet ControlNetModel's
controlnet_cond_embedding. Porting P1-10's 6-channel raw source-RGB + Canny
conditioning therefore means VAE-encoding source RGB and Canny (each 3ch)
SEPARATELY to 16ch latents and concatenating -> 32 total channels, i.e.
extra_conditioning_channels=16 (the base 16 + 16 extra). This probe fabricates
that shape directly via a random 32-channel latent-sized tensor -- no real VAE
encode needed for a VRAM-only check.

Usage
-----
    python probe_p1_10_controlnet_sd35_vram.py [--num-layers 12] [--rank 8] [--resolution 1024]
"""

from __future__ import annotations

import argparse
import os


def parse_args():
    ap = argparse.ArgumentParser(description="VRAM feasibility probe: P1-10-style ControlNet on SD3.5.")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--num-layers", type=int, default=12,
                    help="ControlNet clone depth (P1-10-analog default; SD3's official Canny "
                         "ControlNet also uses a partial-depth clone, not the full 38 layers).")
    ap.add_argument("--extra-conditioning-channels", type=int, default=16,
                    help="16 = dual-VAE-encode design (source RGB + Canny, each 3ch->16ch latent, "
                         "concatenated to 32ch total alongside the base 16ch noisy latent).")
    ap.add_argument("--rank", type=int, default=8, help="Colour LoRA rank (parity with PR-01).")
    ap.add_argument("--resolution", type=int, default=1024,
                    help="Pixel resolution -- determines the 128x128 (at 1024, /8 VAE downsample) "
                         "latent spatial size used for the dummy tensors.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--mixed-precision", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--online", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    local_only = not args.online

    import torch
    import torch.nn.functional as F
    from diffusers import SD3ControlNetModel, SD3Transformer2DModel
    from peft import LoraConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("No CUDA device -- this probe must run on a GPU node.")
    torch.cuda.reset_peak_memory_stats(device)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.mixed_precision]

    print(f"Loading frozen SD3.5 transformer ({args.model}) in {args.mixed_precision} ...")
    transformer = SD3Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=dtype, local_files_only=local_only)
    transformer.requires_grad_(False)
    print(f"  transformer: {sum(p.numel() for p in transformer.parameters()):,} params, "
          f"num_layers={transformer.config.num_layers}")

    print(f"Cloning SD3ControlNetModel via from_transformer(num_layers={args.num_layers}, "
          f"num_extra_conditioning_channels={args.extra_conditioning_channels}) ...")
    controlnet = SD3ControlNetModel.from_transformer(
        transformer, num_layers=args.num_layers,
        num_extra_conditioning_channels=args.extra_conditioning_channels)
    controlnet.to(dtype)

    # Freeze everything cloned from the pretrained transformer; only the
    # genuinely NEW, zero-initialised parts train -- pos_embed_input and
    # controlnet_blocks (the zero-conv residual connections), mirroring
    # P1-10's TRAINABLE_CONTROLNET_PREFIXES convention for the UNet
    # ControlNetModel (conditioning encoder + zero-conv residuals only).
    controlnet.requires_grad_(False)
    n_cn_trainable = 0
    for name, p in controlnet.named_parameters():
        if name.startswith("pos_embed_input") or name.startswith("controlnet_blocks"):
            p.requires_grad_(True)
            n_cn_trainable += p.numel()
    n_cn_total = sum(p.numel() for p in controlnet.parameters())
    print(f"  controlnet: {n_cn_total:,} params total, {n_cn_trainable:,} trainable "
          f"({n_cn_trainable / n_cn_total * 100:.3f}%)")

    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(lora_config)
    transformer.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()

    lora_trainable = [p for p in transformer.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in lora_trainable)
    print(f"  colour LoRA rank {args.rank}: {n_lora:,} trainable params")

    transformer.to(device)
    controlnet.to(device)
    transformer.train()
    controlnet.train()

    trainable = lora_trainable + [p for p in controlnet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    # ---- dummy forward + backward at real shapes (no data/text-encoder loading) ----
    lat = args.resolution // 8  # VAE downsample factor
    bsz = args.batch_size
    noisy = torch.randn(bsz, 16, lat, lat, device=device, dtype=dtype)
    control_cond = torch.randn(bsz, 16 + args.extra_conditioning_channels, lat, lat,
                                device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(bsz, 77 + 256, transformer.config.joint_attention_dim,
                                        device=device, dtype=dtype)
    pooled = torch.randn(bsz, transformer.config.pooled_projection_dim, device=device, dtype=dtype)
    timestep = torch.tensor([500.0] * bsz, device=device, dtype=dtype)
    target = torch.randn_like(noisy)

    print("Running one dummy forward + backward step ...")
    with torch.autocast(device_type="cuda", dtype=dtype):
        control_block_samples = controlnet(
            hidden_states=noisy, timestep=timestep,
            encoder_hidden_states=encoder_hidden_states, pooled_projections=pooled,
            controlnet_cond=control_cond, conditioning_scale=1.0, return_dict=False,
        )[0]
        model_pred = transformer(
            hidden_states=noisy, timestep=timestep,
            encoder_hidden_states=encoder_hidden_states, pooled_projections=pooled,
            block_controlnet_hidden_states=control_block_samples, return_dict=False,
        )[0]
        loss = F.mse_loss(model_pred.float(), target.float())

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"\nOne dummy step completed cleanly (loss={loss.item():.4f}).")
    print(f"Peak VRAM allocated: {peak_gb:.2f} GB")
    print(f"bigbatch RTX 3090 = 24GB -> {'FITS' if peak_gb < 24 else 'DOES NOT FIT'} "
          f"({24 - peak_gb:+.2f} GB headroom).")


if __name__ == "__main__":
    main()
