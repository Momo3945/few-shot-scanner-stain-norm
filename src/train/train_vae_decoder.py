#!/usr/bin/env python3
"""
train_vae_decoder.py -- P1-18: fine-tune the SD1.5-compatible VAE on H&E self-reconstruction
(input == target; no scanner pairs, so registration/pair-correspondence is irrelevant).
See tickets/P1-18_vae_decoder_finetune.md.

--mode decoder   Stage 1. Encoder + quant_conv FROZEN (latents bit-identical to today's, so P1-10 /
                 P1-11 stay valid with zero retraining); only decoder + post_quant_conv train.
                 The encoder state-dict hash is recorded before/after and must match.
--mode full      Stage 2a (implemented, NOT yet exercised -- do not run before Stage 1 is
                 reported). Encoder + decoder train, with a latent anchor
                 lambda * MSE(E_new(x), E_orig(x)) against a frozen copy of the original encoder.

Data (all 512px, none touches held-out slides A06/A08/A09/A13/A16 -- leak-checked, hard abort):
  hist_lora_pool/ (MITOS Aperio+Hamamatsu, PanNuke, TCGA)  +  pairs/train (A03/H03 crops).
  Validation is split by SLIDE (MITOS: whole slides, both scanners; TCGA: whole slides) or random
  (PanNuke has no slide id), never by crop within a slide.

Training loss is NOT a success signal (project guardrail). Validation SSIM/PSNR/MAE use the same
metrics.py functions as benchmark_vae_reconstruction.py; the real verdict comes from that
benchmark on the 496 held-out crops.

Env-style overrides live in the .slurm (SMOKE=1, OVERFIT_N=N); here they are --smoke / --overfit-n.
Resumable: state.pt in the run dir is picked up automatically.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

HELDOUT_SLIDES = {"A06", "A08", "A09", "A13", "A16", "H06", "H08", "H09", "H13", "H16"}
_HELD_RE = re.compile(r"(?<![A-Za-z0-9])[AH](?:06|08|09|13|16)(?![0-9])")

# Fixed, logged validation slides (chosen once, before any training; see ticket Stage 0.2).
VAL_MITOS_SLIDES = {"A10", "H10", "A15", "H15"}
VAL_TCGA_SLIDES = {"TCGA-BH-A0DZ-01Z-00-DX1", "TCGA-E2-A1BD-01Z-00-DX1"}
VAL_PANNUKE_N = 100


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["decoder", "full"], default="decoder")
    ap.add_argument("--init", choices=["ft_mse", "stock"], default="ft_mse",
                    help="Starting weights. ft_mse = current best decoder (stock encoder, identical).")
    ap.add_argument("--pool-dir", required=True, help="hist_lora_pool/ (manifest.csv + patches/).")
    ap.add_argument("--pairs-dir", required=True, help="pairs/train (A03/H03 *_aperio.png/_hamamatsu.png).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train-steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256, choices=[256, 512],
                    help="Training crop (px). 256 matches how ft_mse itself was trained; "
                         "validation is always full 512.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--l1-weight", type=float, default=1.0)
    ap.add_argument("--lpips-weight", type=float, default=0.3)
    ap.add_argument("--ssim-weight", type=float, default=0.0, help="Optional 1-SSIM term (default off).")
    ap.add_argument("--anchor-lambda", type=float, default=0.0, help="--mode full only.")
    ap.add_argument("--mitos-weight", type=float, default=3.0,
                    help="Sampling weight of MITOS content (pool MITOS + pair crops) vs PanNuke/TCGA.")
    ap.add_argument("--pannuke-weight", type=float, default=1.0,
                    help="Sampling weight of PanNuke patches. PanNuke images here are UPSCALED (source "
                         "folder upscale_results_train) and trivially easy (init SSIM ~0.95 vs ~0.6 on "
                         "MITOS), so they can teach the decoder that blur is fine. Lower this (0.0-0.3) "
                         "if the validation log shows MITOS gaining less than expected.")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--no-augment", action="store_true",
                    help="Disable random rot90/flip (needed for a true memorisation / capacity control).")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--overfit-n", type=int, default=0,
                    help="Capacity control: train on N crops and validate on those SAME crops.")
    ap.add_argument("--smoke", action="store_true", help="5 steps, 8 val images, env/loop check only.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--online", action="store_true")
    return ap.parse_args()


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def _domain_of_pool_row(row):
    if row["source"] == "mitos":
        return "aperio" if row["slide_id_or_tissue_type"].startswith("A") else "hamamatsu"
    return "tcga" if row["source"] == "tcga" else "pannuke"


DOMAINS = ("aperio", "hamamatsu", "tcga", "pannuke")
MITOS_DOMAINS = ("aperio", "hamamatsu")


def build_lists(pool_dir: Path, pairs_dir: Path, seed: int):
    """Return (train, val) lists of dicts {path, domain, mitos(bool), id}."""
    train, val = [], []
    rows = list(csv.DictReader(open(pool_dir / "manifest.csv", encoding="utf-8-sig")))
    rng = random.Random(seed)
    pannuke = [r for r in rows if r["source"] == "pannuke"]
    pannuke_val = {r["filename"] for r in rng.sample(pannuke, VAL_PANNUKE_N)}
    for r in rows:
        slide = r["slide_id_or_tissue_type"]
        item = {"path": str(pool_dir / "patches" / r["filename"]), "domain": _domain_of_pool_row(r),
                "mitos": r["source"] == "mitos", "id": r["filename"], "slide": slide}
        is_val = ((r["source"] == "mitos" and slide in VAL_MITOS_SLIDES)
                  or (r["source"] == "tcga" and slide in VAL_TCGA_SLIDES)
                  or (r["source"] == "pannuke" and r["filename"] in pannuke_val))
        (val if is_val else train).append(item)
    # A03/H03 pair crops: always TRAIN (A03/H03 are not validation slides).
    for p in sorted(pairs_dir.glob("A03_*_aperio.png")) + sorted(pairs_dir.glob("A03_*_hamamatsu.png")):
        dom = "aperio" if p.name.endswith("_aperio.png") else "hamamatsu"
        train.append({"path": str(p), "domain": dom, "mitos": True, "id": p.name, "slide": "A03"})
    return train, val


def leak_check(train, val):
    for name, lst in (("train", train), ("val", val)):
        # Regex only on MITOS ids: TCGA barcodes like TCGA-A8-A06O would false-positive on it.
        bad = [x["id"] for x in lst
               if x["slide"] in HELDOUT_SLIDES or (x["mitos"] and _HELD_RE.search(x["id"]))]
        if bad:
            raise SystemExit(f"LEAK CHECK FAILED: held-out slide content in {name}: {bad[:5]} ...")
    overlap = {x["path"] for x in train} & {x["path"] for x in val}
    if overlap:
        raise SystemExit(f"Train/val path overlap: {sorted(overlap)[:3]}")
    print(f"Leak check OK: no held-out slide in train ({len(train)}) or val ({len(val)}); no overlap.")


def load_rgb(path):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def sha_of_modules(vae, prefixes=("encoder", "quant_conv")):
    h = hashlib.sha256()
    for k, v in sorted(vae.state_dict().items()):
        if any(k == p or k.startswith(p + ".") for p in prefixes):
            h.update(k.encode()); h.update(v.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def gaussian_ssim_loss(x, y, win=11, sigma=1.5):
    import torch
    import torch.nn.functional as F
    c = x.shape[1]
    ax = torch.arange(win, device=x.device, dtype=x.dtype) - win // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :]).expand(c, 1, win, win).contiguous()
    x, y = (x + 1) / 2, (y + 1) / 2
    mu_x, mu_y = F.conv2d(x, k, groups=c), F.conv2d(y, k, groups=c)
    sxx = F.conv2d(x * x, k, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, k, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, k, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return 1 - ssim.mean()


def main():
    args = parse_args()
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.mode == "full" and args.anchor_lambda <= 0:
        print("WARNING: --mode full with anchor_lambda<=0 is the UNANCHORED variant (Stage 2b territory).")

    import numpy as np
    import torch
    import lpips
    from diffusers import AutoencoderKL
    from metrics import grayscale_ssim, mae as mae_fn, psnr as psnr_fn

    if not torch.cuda.is_available():
        raise SystemExit("ABORT: no CUDA device -- this script is meant for a GPU node.")
    device = torch.device("cuda")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data --------------------------------------------------------------------------------
    train, val = build_lists(Path(args.pool_dir), Path(args.pairs_dir), args.seed)
    leak_check(train, val)
    if args.overfit_n:
        # Capacity control on the project's target domains: half Aperio, half Hamamatsu MITOS crops
        # (a uniform draw would be ~80% TCGA/PanNuke, which says nothing about H&E-scanner headroom).
        rng = random.Random(args.seed)
        half = max(1, args.overfit_n // 2)
        pick = []
        for dom, n in (("aperio", half), ("hamamatsu", args.overfit_n - half)):
            pool_d = [x for x in train if x["domain"] == dom]
            rng.shuffle(pool_d); pick += pool_d[:n]
        train = pick; val = list(train)
        print(f"OVERFIT control: train == val == {len(train)} MITOS crops "
              f"({half} Aperio / {args.overfit_n - half} Hamamatsu).")
    if args.smoke:  # 2 per domain, so every domain's validation path is exercised
        val = [x for d in DOMAINS for x in [v for v in val if v["domain"] == d][:2]]
    counts = lambda lst: {d: sum(1 for x in lst if x["domain"] == d) for d in DOMAINS}
    print(f"train {len(train)} {counts(train)} | val {len(val)} {counts(val)}")
    weights = np.array([args.mitos_weight if x["mitos"] else
                        (args.pannuke_weight if x["domain"] == "pannuke" else 1.0) for x in train],
                       dtype=np.float64)
    weights /= weights.sum()
    rng_np = np.random.default_rng(args.seed)

    # ---- model -------------------------------------------------------------------------------
    repo, sub = (("stabilityai/sd-vae-ft-mse", None) if args.init == "ft_mse"
                 else ("stable-diffusion-v1-5/stable-diffusion-v1-5", "vae"))
    kw = {"subfolder": sub} if sub else {}
    vae = AutoencoderKL.from_pretrained(repo, torch_dtype=torch.float32, **kw).to(device)
    scaling = vae.config.scaling_factor
    print(f"init={args.init} ({repo}) scaling_factor={scaling} "
          f"params={sum(p.numel() for p in vae.parameters()):,}")
    enc_hash_before = sha_of_modules(vae)
    orig_vae = None
    if args.mode == "decoder":
        vae.encoder.requires_grad_(False); vae.quant_conv.requires_grad_(False)
        vae.decoder.requires_grad_(True); vae.post_quant_conv.requires_grad_(True)
    else:
        vae.requires_grad_(True)
        if args.anchor_lambda > 0:
            import copy
            orig_vae = copy.deepcopy(vae).eval().requires_grad_(False)  # frozen reference encoder
    if args.grad_checkpointing:
        vae.enable_gradient_checkpointing()
    params = [p for p in vae.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params):,}")
    lpips_net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    lr_at = lambda s: args.lr * min(1.0, (s + 1) / max(1, args.warmup))

    # ---- resume ------------------------------------------------------------------------------
    step, best_val = 0, -1.0
    state_path = out_dir / "state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        vae.load_state_dict(st["vae"]); opt.load_state_dict(st["opt"])
        step, best_val = st["step"], st["best_val"]
        rng_np = np.random.default_rng(args.seed + step)
        print(f"RESUMED from step {step}")

    # ---- helpers -----------------------------------------------------------------------------
    def to_tensor(arr):
        return torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)

    def augment(arr):
        if args.no_augment:
            return arr
        k = int(rng_np.integers(4))
        arr = np.rot90(arr, k)
        if rng_np.random() < 0.5:
            arr = arr[:, ::-1]
        return np.ascontiguousarray(arr)

    def make_batch():
        idx = rng_np.choice(len(train), size=args.batch_size, p=weights)
        return torch.stack([to_tensor(augment(load_rgb(train[i]["path"]))) for i in idx]).to(device)

    def decode_target(x):
        """Return (recon_crop, target_crop, latent_new, latent_orig_or_None) for one training batch."""
        crop = args.crop
        if args.mode == "decoder" and crop < 512:
            # Encode FULL 512 (latent context identical to inference), decode an aligned window.
            with torch.no_grad(), torch.autocast("cuda", enabled=False):  # fp32 frozen encoder
                z = vae.encode(x).latent_dist.mode()
            lc = crop // 8
            oy, ox = (int(rng_np.integers(0, 512 // 8 - lc + 1)) for _ in range(2))
            zc = z[:, :, oy:oy + lc, ox:ox + lc]
            rec = vae.decode(zc).sample
            tgt = x[:, :, oy * 8:oy * 8 + crop, ox * 8:ox * 8 + crop]
            return rec, tgt, None, None
        if args.mode == "decoder":
            with torch.no_grad(), torch.autocast("cuda", enabled=False):  # fp32 frozen encoder
                z = vae.encode(x).latent_dist.mode()
            return vae.decode(z).sample, x, None, None
        # full mode: encode the pixel crop with grad
        if crop < 512:
            oy, ox = (int(rng_np.integers(0, 512 - crop + 1)) // 8 * 8 for _ in range(2))
            x = x[:, :, oy:oy + crop, ox:ox + crop]
        z = vae.encode(x).latent_dist.mode()
        z_orig = None
        if orig_vae is not None:
            with torch.no_grad():
                z_orig = orig_vae.encode(x).latent_dist.mode()
        return vae.decode(z).sample, x, z, z_orig

    @torch.no_grad()
    def validate(tag):
        vae.eval()
        per = {d: [] for d in DOMAINS}
        for it in val:
            arr = load_rgb(it["path"])
            x = to_tensor(arr).unsqueeze(0).to(device)
            rec = vae.decode(vae.encode(x).latent_dist.mode()).sample
            out = ((rec[0].float().cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            per[it["domain"]].append((grayscale_ssim(out, arr), psnr_fn(out, arr), mae_fn(out, arr)))
        res = {}
        for d, v in per.items():
            if v:
                a = np.asarray(v)
                res[d] = {"n": len(v), "ssim": float(a[:, 0].mean()), "psnr": float(a[:, 1].mean()),
                          "mae": float(a[:, 2].mean())}
        # Model selection uses the MITOS domains only (the project's target domains); the TCGA/PanNuke
        # numbers are reported but PanNuke is near-saturated and would dilute the signal. Falls back to
        # all domains present (e.g. the overfit control, which has a random mix).
        sel = [res[d]["ssim"] for d in MITOS_DOMAINS if d in res] or [r["ssim"] for r in res.values()]
        res["select_ssim"] = float(np.mean(sel))
        print(f"[val {tag}] " + " | ".join(
            f"{d}: ssim={r['ssim']:.4f} psnr={r['psnr']:.2f} mae={r['mae']:.2f} (n={r['n']})"
            for d, r in res.items() if isinstance(r, dict)) + f" | select_ssim={res['select_ssim']:.4f}")
        if args.mode == "decoder":
            vae.decoder.train(); vae.post_quant_conv.train()
        else:
            vae.train()
        return res

    def save(tag):
        d = out_dir / tag
        vae.save_pretrained(d)
        print(f"saved {d}")

    log_path = out_dir / "val_log.jsonl"
    total = 5 if args.smoke else args.train_steps
    if step == 0:
        res0 = validate("init (step 0)")
        with open(log_path, "a") as fh:
            fh.write(json.dumps({"step": 0, **res0}) + "\n")
        best_val = res0["select_ssim"]
    vae.eval()
    if args.mode == "decoder":
        vae.decoder.train(); vae.post_quant_conv.train()
    else:
        vae.train()
    json.dump({**vars(args), "n_train": len(train), "n_val": len(val), "enc_hash_before": enc_hash_before},
              open(out_dir / "run_config.json", "w"), indent=2)

    # ---- train loop --------------------------------------------------------------------------
    t0 = time.time()
    while step < total:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x = make_batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rec, tgt, z_new, z_orig = decode_target(x)
        rec = rec.float(); tgt = tgt.float()
        l1 = (rec - tgt).abs().mean()
        loss = args.l1_weight * l1
        lp = torch.zeros((), device=device)
        if args.lpips_weight > 0:
            lp = lpips_net(rec, tgt).mean()
            loss = loss + args.lpips_weight * lp
        if args.ssim_weight > 0:
            loss = loss + args.ssim_weight * gaussian_ssim_loss(rec, tgt)
        if z_orig is not None:
            loss = loss + args.anchor_lambda * (z_new.float() - z_orig.float()).pow(2).mean()
        if not torch.isfinite(loss):
            raise SystemExit(f"ABORT: non-finite loss at step {step}.")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % 25 == 0 or step <= 5:
            print(f"step {step}/{total} loss={loss.item():.4f} l1={l1.item():.4f} lpips={lp.item():.4f} "
                  f"lr={lr_at(step):.2e} {(time.time() - t0) / step:.2f}s/step")
        if step % args.val_every == 0 or step == total:
            res = validate(f"step {step}")
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"step": step, **res}) + "\n")
            if res["select_ssim"] > best_val and not args.smoke:
                best_val = res["select_ssim"]; save("best_val")
        if step % args.save_every == 0 and not args.smoke:
            torch.save({"vae": vae.state_dict(), "opt": opt.state_dict(), "step": step,
                        "best_val": best_val}, state_path)

    save("final")
    enc_hash_after = sha_of_modules(vae)
    verdict = {"enc_hash_before": enc_hash_before, "enc_hash_after": enc_hash_after,
               "encoder_unchanged": enc_hash_before == enc_hash_after}
    json.dump(verdict, open(out_dir / "encoder_hash_check.json", "w"), indent=2)
    print("ENCODER CHECK:", json.dumps(verdict))
    if args.mode == "decoder" and not verdict["encoder_unchanged"]:
        raise SystemExit("FAIL: encoder changed during decoder-only training (G1.0).")
    print(f"DONE. best_val select_ssim={best_val:.4f}; final + best_val in {out_dir}")


if __name__ == "__main__":
    main()
