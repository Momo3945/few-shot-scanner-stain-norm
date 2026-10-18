#!/usr/bin/env python3
"""
vae_stage0_checks.py -- P1-18 Stage 0 CPU checks (tickets/P1-18_vae_decoder_finetune.md).

--weight-diff  Stage 0.1: is sd-vae-ft-mse's ENCODER identical to the stock SD1.5 VAE's?
               Per top-level module (encoder / quant_conv / decoder / post_quant_conv):
               tensor count, tensors bit-identical, max |diff|, relative L2 diff.
               If the encoder is identical, ft_mse is already a decoder-only fine-tune and
               P1-15 was effectively "Stage 1 with a pretrained decoder".
--lpips        Stage 0.4: confirm the `lpips` package imports, its VGG backbone weights are
               cached under TORCH_HOME (verified by BYTE SIZE, not exit code), and a forward
               pass gives a finite, sensible distance. Reports torch version so the caller
               can confirm it was not changed by `pip install --no-deps lpips`.

CPU-only, standalone, touches no validated script. Run via slurm/stage0_vae_checks.slurm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def weight_diff(out_dir: Path) -> dict:
    import torch
    from diffusers import AutoencoderKL

    stock = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                          subfolder="vae", torch_dtype=torch.float32)
    ftmse = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32)
    sa, sb = stock.state_dict(), ftmse.state_dict()
    if set(sa) != set(sb):
        only_a, only_b = sorted(set(sa) - set(sb)), sorted(set(sb) - set(sa))
        raise SystemExit(f"State-dict keys differ: only stock={only_a[:5]} only ft_mse={only_b[:5]}")

    modules = ["encoder", "quant_conv", "decoder", "post_quant_conv"]
    report = {}
    for m in modules:
        keys = [k for k in sa if k == m or k.startswith(m + ".")]
        n_ident, max_abs, num, den = 0, 0.0, 0.0, 0.0
        for k in keys:
            a, b = sa[k].float(), sb[k].float()
            if torch.equal(a, b):
                n_ident += 1
            d = (a - b)
            max_abs = max(max_abs, d.abs().max().item())
            num += d.pow(2).sum().item()
            den += a.pow(2).sum().item()
        report[m] = {
            "n_tensors": len(keys), "n_bit_identical": n_ident,
            "all_identical": n_ident == len(keys), "max_abs_diff": max_abs,
            "relative_l2_diff": (num / den) ** 0.5 if den else None,
        }
        print(f"{m:16s} tensors={len(keys):4d} identical={n_ident:4d} "
              f"max|diff|={max_abs:.3e} relL2={report[m]['relative_l2_diff']:.3e}")
    report["verdict"] = {
        "encoder_identical": report["encoder"]["all_identical"] and report["quant_conv"]["all_identical"],
        "decoder_identical": report["decoder"]["all_identical"] and report["post_quant_conv"]["all_identical"],
    }
    print("VERDICT:", json.dumps(report["verdict"]))
    (out_dir / "weight_diff.json").write_text(json.dumps(report, indent=2))
    return report


def lpips_check(out_dir: Path) -> dict:
    import torch
    report = {"torch_version": torch.__version__}
    import lpips  # noqa: F401  (ImportError here == package missing)
    report["lpips_version"] = getattr(lpips, "__version__", "unknown")

    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    # Building the model triggers the torchvision VGG16 download into TORCH_HOME if absent.
    net = lpips.LPIPS(net="vgg", verbose=False).eval()
    ckpts = sorted((torch_home / "hub" / "checkpoints").glob("vgg*"))
    report["vgg_checkpoints"] = [{"file": p.name, "bytes": p.stat().st_size} for p in ckpts]
    # torchvision vgg16 is ~528 MB; anything tiny is a stub / failed download.
    report["vgg_weights_real"] = any(p.stat().st_size > 400_000_000 for p in ckpts)

    g = torch.Generator().manual_seed(0)
    x = torch.rand(2, 3, 128, 128, generator=g) * 2 - 1
    with torch.no_grad():
        d_same = net(x, x).flatten().tolist()
        d_diff = net(x, torch.rand(2, 3, 128, 128, generator=g) * 2 - 1).flatten().tolist()
    report["lpips_identical_images"] = d_same
    report["lpips_random_pair"] = d_diff
    report["forward_ok"] = (all(abs(v) < 1e-6 for v in d_same)
                            and all(v > 0.1 and v == v for v in d_diff))
    print(json.dumps(report, indent=2))
    (out_dir / "lpips_check.json").write_text(json.dumps(report, indent=2))
    if not (report["vgg_weights_real"] and report["forward_ok"]):
        raise SystemExit("LPIPS check FAILED (weights stub or bad forward) -- see report above.")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-diff", action="store_true")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not (args.weight_diff or args.lpips):
        raise SystemExit("Pass --weight-diff and/or --lpips.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.weight_diff:
        weight_diff(out_dir)
    if args.lpips:
        lpips_check(out_dir)


if __name__ == "__main__":
    main()
